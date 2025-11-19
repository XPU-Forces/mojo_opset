import torch

from ..mojo_function import MojoFuncBase, mojo_func_dispatcher


@mojo_func_dispatcher
class MojoFlashAttnFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(
        ctx, 
        q, 
        k, 
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=0.0,
        causal=False,
        softmax_scale=1.0,
        block_q_len=128,
        block_k_len=512,
    ):
        pass

    @staticmethod
    def forward_ref(
        ctx, 
        q, 
        k, 
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=0.0,
        causal=False,
        softmax_scale=1.0,
        block_q_len=128,
        block_k_len=512,
    ):
        assert q.ndim == k.ndim == v.ndim == 3, f"Input shape should be [total_S, H, D], but get q{q.shape}, k{k.shape}, v{v.shape}"
        Sq, Nq, D = q.shape
        Sk, Nk, Dk = k.shape
        Sv, Nv, Dv = v.shape
        assert Sk == Sv and Nk == Nv, f"Seq length and head number of k, v should be same, but get k{k.shape}, v{v.shape}"
        assert D == Dk, f"Head dim of q,k should be same, but get q{q.shape}, k{k.shape}"
        assert Nq % Nk == 0, f"Head number of q should be divisible by k and v, but get q{q.shape}, k{k.shape}, v{v.shape}"
        G = Nq // Nk

        cu_seqlens_q = cu_seqlens_q.to("cpu").tolist()
        cu_seqlens_k = cu_seqlens_k.to("cpu").tolist()

        B = len(cu_seqlens_q) - 1
        assert len(cu_seqlens_k) == B + 1

        O = q.new_zeros((Sq, Nq, Dv))
        L = torch.zeros((Sq, Nq), device=q.device, dtype=q.dtype)

        q = q.permute(1, 0, 2).reshape(G, Nk, Sq, D) # [Sq, Nq, D] -> [G, Nk, Sq, D]
        k = k.permute(1, 0, 2) # [Sk, Nk, D] -> [Nk, Sk, D]
        v = v.permute(1, 0, 2) # [Sk, Nk, D] -> [Nk, Sk, D]

        dropout_scale = 1 / (1 - dropout_p) if dropout_p > 0.0 else None
        dropout_masks = [] if dropout_p > 0.0 else None

        for b in range(B):
            cur_q_len = cu_seqlens_q[b+1] - cu_seqlens_q[b]
            cur_k_len = cu_seqlens_k[b+1] - cu_seqlens_k[b]
            cur_q = q[:, :, cu_seqlens_q[b]:cu_seqlens_q[b+1], :] # [G, Nk, cu_seqlens_k, D]
            cur_k = k[:, cu_seqlens_k[b]:cu_seqlens_k[b+1], :] # [Nk, cu_seqlens_k, D]
            cur_v = v[:, cu_seqlens_k[b]:cu_seqlens_k[b+1], :] # [Nk, cu_seqlens_k, D]

            cur_O = torch.zeros((cur_q_len, Nq, Dv), device=q.device, dtype=q.dtype)
            cur_L = torch.zeros((cur_k_len, Nq), device=q.device, dtype=q.dtype)

            if dropout_p > 0.0:
                dropout_mask = torch.empty(G, Nk, cur_q_len, cur_k_len, dtype=torch.bool, device=q.device).bernoulli_(1 - dropout_p)
                dropout_masks.append(dropout_mask)
            else:
                dropout_mask = None

            for q0 in range(0, cur_q_len, block_q_len):
                ql = min(block_q_len, cur_q_len - q0)
                q_block = cur_q[:, :, q0:q0+ql, :].contiguous() # [G, Nk, ql, D]

                m = torch.full((G, Nk, ql), float("-inf"), dtype=q.dtype, device=q.device)
                l = torch.zeros((G, Nk, ql), dtype=q.dtype, device=q.device)
                n = torch.zeros((G, Nk, ql, Dv), dtype=q.dtype, device=q.device)

                for k0 in range(0, cur_k_len, block_k_len):
                    kl = min(block_k_len, cur_k_len - k0)
                    k_block = cur_k[:, k0:k0+kl, :].contiguous() # [Nk, kl, D]
                    v_block = cur_v[:, k0:k0+kl, :].contiguous() # [Nk, kl, D]

                    # q @ k.T: [G, Nk, ql, D] @ [Nk, kl, D] -> [G, Nk, ql, kl]
                    scores = torch.einsum("g h q d, h k d -> g h q k", q_block, k_block) * softmax_scale

                    # attention mask
                    if causal and Sq == Sk:
                        q_idx = torch.arange(q0, q0+ql, device=q.device)[:, None] # [ql, 1]
                        k_idx = torch.arange(k0, k0+kl, device=q.device)[None, :] # [1, kl]
                        mask = k_idx > q_idx # [ql, kl]
                        scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))
                    
                    # softmax
                    m_curr = scores.amax(dim=-1) # [G, Nk, ql, kl] -> [G, Nk, ql]
                    m_new = torch.maximum(m, m_curr) # [G, Nk, ql]
                    exp_m_mnew = torch.exp(m - m_new) # [G, Nk, ql]
                    exp_s_mnew = torch.exp(scores - m_new[..., None]) # [G, Nk, ql] -> [G, Nk, ql, kl]
                    l_new = exp_m_mnew * l + exp_s_mnew.sum(dim=-1) # [G, Nk, ql, kl] -> [G, Nk, ql]

                    if dropout_mask is not None:
                        exp_s_mnew = exp_s_mnew * dropout_mask[:, :, q0:q0+ql, k0:k0+kl] * dropout_scale

                    # p @ v: [G, Nk, ql, kl] @ [Nk, kl, D] -> [G, Nk, ql, D]
                    n_new = (exp_m_mnew[..., None] * n) + torch.einsum(
                        "g h q k, h k d -> g h q d", exp_s_mnew, v_block
                    )

                    m, l, n = m_new, l_new, n_new
                
                y_block = n / l[..., None] # [G, Nk, ql, D]

                # [G, Nk, ql, D] -> [ql, Nq, D]
                cur_O[q0:q0+ql, :, :] = y_block.reshape(Nq, ql, Dv).permute(1, 0, 2)

                # [G, Nk, ql] + log([G, Nk, ql]) -> [ql, Nq]
                cur_L[q0:q0+ql, :] =(m + torch.log(l)).reshape(Nq, ql).permute(1, 0)

            O[cu_seqlens_q[b]:cu_seqlens_q[b+1], :, :] = cur_O
            L[cu_seqlens_q[b]:cu_seqlens_q[b+1], :] = cur_L

        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.dropout_p = dropout_p
        ctx.dropout_masks = dropout_masks
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.block_q_len = block_q_len
        ctx.block_k_len = block_k_len
        ctx.save_for_backward(q, k, v, O, L)

        return O

    @staticmethod
    def backward_dump(ctx, grad_o):
        pass

    @staticmethod
    def backward_ref(ctx, grad_o):
        cu_seqlens_q = ctx.cu_seqlens_q
        cu_seqlens_k = ctx.cu_seqlens_k
        dropout_p = ctx.dropout_p
        dropout_masks = ctx.dropout_masks
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        block_q_len = ctx.block_q_len
        block_k_len = ctx.block_k_len
        q, k, v, O, L = ctx.saved_tensors
        G, _, Sq, Dq = q.shape
        Nk, Sk, Dk = k.shape
        Nv, Sv, Dv = v.shape
        Nq = G * Nk

        grad_q = torch.zeros((Sq, Nq, Dq), device=q.device, dtype=q.dtype) # [Sq, Nq, D]
        grad_k = torch.zeros((Sk, Nk, Dk), device=q.device, dtype=q.dtype) # [Sk, Nk, D]
        grad_v = torch.zeros((Sv, Nv, Dv), device=q.device, dtype=q.dtype) # [Sv, Nv, D]

        O = O.permute(1, 0, 2).reshape(G, Nk, Sq, Dv) # [Sq, Nq, D]-> [G, Nk, Sq, D]
        L = L.permute(1, 0).reshape(G, Nk, Sq) # [Sq, Nq] -> [G, Nk, Sq]
        grad_o = grad_o.permute(1, 0, 2).reshape(G, Nk, Sq, Dv) # [Sq, Nq, D] -> [G, Nk, Sq, D]

        B = len(cu_seqlens_q) - 1

        dropout_scale = 1 / (1 - dropout_p) if dropout_p > 0.0 else None

        for b in range(B):
            cur_q_len = cu_seqlens_q[b+1] - cu_seqlens_q[b]
            cur_k_len = cu_seqlens_k[b+1] - cu_seqlens_k[b]
            cur_q = q[:, :, cu_seqlens_q[b]:cu_seqlens_q[b+1], :] # [G, Nk, cu_seqlens_k, D]
            cur_k = k[:, cu_seqlens_k[b]:cu_seqlens_k[b+1], :] # [Nk, cu_seqlens_k, D]
            cur_v = v[:, cu_seqlens_k[b]:cu_seqlens_k[b+1], :] # [Nk, cu_seqlens_k, D]
            cur_o = O[:, :, cu_seqlens_q[b]:cu_seqlens_q[b+1], :] # [G, Nk, cu_seqlens_q, D]
            cur_do = grad_o[:, :, cu_seqlens_q[b]:cu_seqlens_q[b+1], :] # [G, Nk, cu_seqlens_q, D]
            cur_l = L[:, :, cu_seqlens_q[b]:cu_seqlens_q[b+1]] # [G, Nk, cu_seqlens_q]

            D = (cur_do * cur_o).sum(dim=-1) # [G, Nk, cu_seqlens_q, D] -> [G, Nk, cu_seqlens_q]
            cur_dq = torch.zeros_like(cur_q) # [G, Nk, cu_seqlens_k, D]
            cur_dk = torch.zeros_like(cur_k) # [Nk, cu_seqlens_k, D]
            cur_dv = torch.zeros_like(cur_v) # [Nk, cu_seqlens_k, D]

            dropout_mask = dropout_masks[b] if dropout_masks is not None else None

            for k0 in range(0, cur_k_len, block_k_len):
                kl = min(block_k_len, cur_k_len - k0)
                k_block = cur_k[:, k0:k0+kl, :].contiguous() # [Nk, kl, D]
                v_block = cur_v[:, k0:k0+kl, :].contiguous() # [Nk, kl, D]

                for q0 in range(0, cur_q_len, block_q_len):
                    ql = min(block_q_len, cur_q_len - q0)
                    q_block = cur_q[:, :, q0:q0+ql, :] # [G, Nk, ql, D]
                    grad_o_block = cur_do[:, :, q0:q0+ql, :] # [G, Nk, ql, D]
                    L_block = cur_l[:, :, q0:q0+ql] # [G, Nk, ql]
                    D_block = D[:, :, q0:q0+ql] # [G, Nk, ql]

                    # q @ k.T: [G, Nk, ql, D] @ [Nk, kl, D] -> [G, Nk, ql, kl]
                    scores = torch.einsum("g h q d, h k d -> g h q k", q_block, k_block) * softmax_scale

                    # attention mask
                    if causal and Sq == Sk:
                        q_idx = torch.arange(q0, q0+ql, device=q.device)[:, None] # [ql, 1]
                        k_idx = torch.arange(k0, k0+kl, device=q.device)[None, :] # [1, kl]
                        mask = k_idx > q_idx # [ql, kl]
                        scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))

                    exp_s_mnew = torch.exp(scores - L_block[..., None]) # [G, Nk, ql] -> [G, Nk, ql, kl]

                    # dropout mask
                    if dropout_mask is not None:
                        exp_s_mnew = exp_s_mnew * dropout_mask[:, :, q0:q0+ql, k0:k0+kl] * dropout_scale

                    # dV += P.T @ dY, [G, Nk, ql, kl] @ [G, Nk, ql, D] -> [Nk, kl, D]
                    cur_dv[:, k0:k0+kl, :] += torch.einsum("g h q k, g h q d -> h k d", exp_s_mnew, grad_o_block)

                    # [G, Nk, ql, D] @ [Nk, kl, D] ->  [G, Nk, ql, kl]
                    grad_exp_s_mnew = torch.einsum("g h q d, h k d -> g h q k", grad_o_block, v_block)

                    if dropout_mask is not None:
                        grad_exp_s_mnew = grad_exp_s_mnew * dropout_mask[:, :, q0:q0+ql, k0:k0+kl] * dropout_scale

                    grad_score = exp_s_mnew * (grad_exp_s_mnew - D_block[..., None]) # [G, Nk, ql] -> [G, Nk, ql, kl]

                    # [G, Nk, ql, kl] @ [Nk, kl, D] -> [G, Nk, ql, D]
                    cur_dq[:, :, q0:q0+ql, :] += torch.einsum("g h q k, h k d -> g h q d", grad_score, k_block) * softmax_scale

                    #  [G, Nk, ql, kl] @ [G, Nk, ql, D] -> [Nk, kl, D]
                    cur_dk[:, k0:k0+kl, :] += torch.einsum("g h q k, g h q d -> h k d", grad_score, q_block) * softmax_scale

            # [G, Nk, cu_seqlens_q, D] -> [cu_seqlens_q, Nq, D]
            grad_q[cu_seqlens_q[b]:cu_seqlens_q[b+1], :, :] = cur_dq.reshape(Nq, cur_q_len, Dq).permute(1, 0, 2)
            
            # [Nk, cu_seqlens_k, D] -> [cu_seqlens_k, Nk, D]
            grad_k[cu_seqlens_k[b]:cu_seqlens_k[b+1], :, :] = cur_dk.permute(1, 0, 2)
            grad_v[cu_seqlens_k[b]:cu_seqlens_k[b+1], :, :] = cur_dv.permute(1, 0, 2)

        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None
