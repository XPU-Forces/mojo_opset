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
        causal=False,
        softmax_scale=1.0,
        block_q_len=128,
        block_k_len=512,
    ):
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, f"Input shape should be [B,S,N,D], but get q{q.shape}, k{k.shape}, v{v.shape}"
        B, Sq, Nq, D = q.shape
        Bk, Sk, Nk, Dk = k.shape
        Bv, Sv, Nv, Dv = v.shape
        assert B == Bk == Bv, f"Batch size of q, k, v should be same, but get q{q.shape}, k{k.shape}, v{v.shape}"
        assert Sk == Sv and Nk == Nv, f"Seq length and head number of k, v should be same, but get k{k.shape}, v{v.shape}"
        assert D == Dk, f"Head dim of q,k should be same, but get q{q.shape}, k{k.shape}"
        assert Nq % Nk == 0, f"Head number of q should be divisible by k and v, but get q{q.shape}, k{k.shape}, v{v.shape}"
        G = Nq // Nk

        O = q.new_zeros((B, Sq, Nq, Dv))
        L = torch.zeros((B, Sq, Nq), device=q.device, dtype=q.dtype)

        q = q.permute(0, 2, 1, 3).reshape(B, G, Nk, Sq, D) # [B, S, N, D] -> [B, G, Nk, S, D]
        k = k.permute(0, 2, 1, 3) # [B, S, Nk, D] -> [B, Nk, S, D]
        v = v.permute(0, 2, 1, 3) # [B, S, Nk, D] -> [B, Nk, S, D]

        for q0 in range(0, Sq, block_q_len):
            ql = min(block_q_len, Sq - q0)
            q_block = q[:, :, :, q0:q0+ql, :].contiguous() # [B, G, Nk, ql, D]

            m = torch.full((B, G, Nk, ql), float("-inf"), dtype=q.dtype, device=q.device)
            l = torch.zeros((B, G, Nk, ql), dtype=q.dtype, device=q.device)
            n = torch.zeros((B, G, Nk, ql, Dv), dtype=q.dtype, device=q.device)

            for k0 in range(0, Sk, block_k_len):
                kl = min(block_k_len, Sk - k0)
                k_block = k[:, :, k0:k0+kl, :].contiguous() # [B, Nk, kl, D]
                v_block = v[:, :, k0:k0+kl, :].contiguous() # [B, Nk, kl, D]

                # q @ k.T: [B, G, Nk, ql, D] @ [B, Nk, kl, D] -> [B, G, Nk, ql, kl]
                scores = torch.einsum("b g h q d, b h k d -> b g h q k", q_block, k_block) * softmax_scale

                # attention mask
                if causal and Sq == Sk:
                    q_idx = torch.arange(q0, q0+ql, device=q.device)[:, None] # [ql, 1]
                    k_idx = torch.arange(k0, k0+kl, device=q.device)[None, :] # [1, kl]
                    mask = k_idx > q_idx # [ql, kl]
                    scores = scores.masked_fill(mask[None, None, None, :, :], float('-inf'))
                
                # softmax
                m_curr = scores.amax(dim=-1) # [B, G, Nk, ql, kl] -> [B, G, Nk, ql]
                m_new = torch.maximum(m, m_curr) # [B, G, Nk, ql]
                exp_m_mnew = torch.exp(m - m_new) # [B, G, Nk, ql]
                exp_s_mnew = torch.exp(scores - m_new[..., None]) # [B, G, Nk, ql] -> [B, G, Nk, ql, kl]
                l_new = exp_m_mnew * l + exp_s_mnew.sum(dim=-1) # [B, G, Nk, ql, kl] -> [B, G, Nk, ql]

                # p @ v: [B, G, Nk, ql, kl] @ [B, Nk, kl, D] -> [B, G, Nk, ql, D]
                n_new = (exp_m_mnew[..., None] * n) + torch.einsum(
                    "b g h q k, b h k d -> b g h q d", exp_s_mnew, v_block
                )

                m, l, n = m_new, l_new, n_new
            
            y_block = n / l[..., None] # [B, G, Nk, ql, D]

            # [B, G, Nk, ql, D] -> [B, ql, G, Nk, D] -> [B, ql, Nq, D]
            O[:, q0:q0+ql, :, :] = y_block.permute(0, 3, 1, 2, 4).reshape(B, ql, Nq, Dv)

            # [B, G, Nk, ql] + log([B, G, Nk, ql]) -> [B, ql, G, Nk] -> [B, ql, Nq]
            L[:, q0:q0+ql, :] =(m + torch.log(l)).permute(0, 3, 1, 2).reshape(B, ql, Nq)

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
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        block_q_len = ctx.block_q_len
        block_k_len = ctx.block_k_len
        q, k, v, O, L = ctx.saved_tensors
        B, G, _, Sq, Dq = q.shape
        Bk, Nk, Sk, Dk = k.shape
        Bv, Nv, Sv, Dv = v.shape
        Nq = G * Nk

        grad_q = torch.zeros_like(q) # [B, G, Nk, S, D]
        grad_k = torch.zeros_like(k) # [B, Nk, S, D]
        grad_v = torch.zeros_like(v) # [B, Nk, S, D]

        O = O.permute(0, 2, 1, 3).reshape(B, G, Nk, Sq, Dv) # [B, S, Nq, D] -> [B, G, Nk, S, D]
        L = L.permute(0, 2, 1).reshape(B, G, Nk, Sq) # [B, ql, Nq] -> [B, G, Nk, S]
        grad_o = grad_o.permute(0, 2, 1, 3).reshape(B, G, Nk, Sq, Dv) # [B, S, Nq, D] -> [B, G, Nk, S, D]

        D = (grad_o * O).sum(dim=-1) # [B, G, Nk, S, D] -> [B, G, Nk, S]

        for k0 in range(0, Sk, block_k_len):
            kl = min(block_k_len, Sk - k0)
            k_block = k[:, :, k0:k0+kl, :].contiguous() # [B, Nk, kl, D]
            v_block = v[:, :, k0:k0+kl, :].contiguous() # [B, Nk, kl, D]

            for q0 in range(0, Sq, block_q_len):
                ql = min(block_q_len, Sq - q0)
                q_block = q[:, :, :, q0:q0+ql, :] # [B, G, Nk, ql, D]
                grad_o_block = grad_o[:, :, :, q0:q0+ql, :] # [B, G, Nk, ql, D]
                L_block = L[:, :, :, q0:q0+ql] # [B, G, Nk, ql]
                D_block = D[:, :, :, q0:q0+ql] # [B, G, Nk, ql]

                # q @ k.T: [B, G, Nk, ql, D] @ [B, Nk, kl, D] -> [B, G, Nk, ql, kl]
                scores = torch.einsum("b g h q d, b h k d -> b g h q k", q_block, k_block) * softmax_scale

                # attention mask
                if causal and Sq == Sk:
                    q_idx = torch.arange(q0, q0+ql, device=q.device)[:, None] # [ql, 1]
                    k_idx = torch.arange(k0, k0+kl, device=q.device)[None, :] # [1, kl]
                    mask = k_idx > q_idx # [ql, kl]
                    scores = scores.masked_fill(mask[None, None, None, :, :], float('-inf'))

                exp_s_mnew = torch.exp(scores - L_block[..., None]) # [B, G, Nk, ql] -> [B, G, Nk, ql, kl]

                # dV += P.T @ dY, [B, G, Nk, ql, kl] @ [B, G, Nk, ql, D] -> [B, Nk, kl, D]
                grad_v[:, :, k0:k0+kl, :] += torch.einsum("b g h q k, b g h q d -> b h k d", exp_s_mnew, grad_o_block)

                # [B, G, Nk, ql, D] @ [B, Nk, kl, D] ->  [B, G, Nk, ql, kl]
                grad_exp_s_mnew = torch.einsum("b g h q d, b h k d -> b g h q k", grad_o_block, v_block)

                grad_score = exp_s_mnew * (grad_exp_s_mnew - D_block[..., None]) # [B, G, Nk, ql] -> [B, G, Nk, ql, kl]

                # [B, G, Nk, ql, kl] @ [B, Nk, kl, D] -> [B, G, Nk, ql, D]
                grad_q[:, :, :, q0:q0+ql, :] += torch.einsum("b g h q k, b h k d -> b g h q d", grad_score, k_block) * softmax_scale

                #  [B, G, Nk, ql, kl] @ [B, G, Nk, ql, D] -> [B, Nk, kl, D]
                grad_k[:, :, k0:k0+kl, :] += torch.einsum("b g h q k, b g h q d -> b h k d", grad_score, q_block) * softmax_scale

        grad_q = grad_q.reshape(B, Nq, Sq, Dq).permute(0, 2, 1, 3)
        grad_k = grad_k.permute(0, 2, 1, 3)
        grad_v = grad_v.permute(0, 2, 1, 3)

        return grad_q, grad_k, grad_v, None, None, None, None
