import torch
import math

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
        softmax_scale=None,
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
        softmax_scale=None,
    ):
        block_q_len = 128
        block_k_len = 128
        assert q.ndim == k.ndim == v.ndim == 3
        Sq, Nq, D = q.shape
        Sk, Nk, Dk = k.shape
        Sv, Nv, Dv = v.shape
        assert Sk == Sv and Nk == Nv
        assert D == Dk
        assert Nq % Nk == 0
        G = Nq // Nk

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        cu_seqlens_q_cpu = cu_seqlens_q.to("cpu").tolist()
        cu_seqlens_k_cpu = cu_seqlens_k.to("cpu").tolist()
        B = len(cu_seqlens_q_cpu) - 1

        O = torch.zeros((Sq, Nq, Dv), device=q.device, dtype=q.dtype)

        L = torch.zeros((Sq, Nq), device=q.device, dtype=torch.float32)

        q_ref = q.permute(1, 0, 2).reshape(Nk, G, Sq, D).permute(1, 0, 2, 3)
        k_ref = k.permute(1, 0, 2)
        v_ref = v.permute(1, 0, 2)

        dropout_scale = 1 / (1 - dropout_p) if dropout_p > 0.0 else None
        dropout_masks = [] if dropout_p > 0.0 else None

        for b in range(B):
            seq_q_start, seq_q_end = cu_seqlens_q_cpu[b], cu_seqlens_q_cpu[b + 1]
            seq_k_start, seq_k_end = cu_seqlens_k_cpu[b], cu_seqlens_k_cpu[b + 1]

            cur_q_len = seq_q_end - seq_q_start
            cur_k_len = seq_k_end - seq_k_start

            cur_q = q_ref[:, :, seq_q_start:seq_q_end, :]
            cur_k = k_ref[:, seq_k_start:seq_k_end, :]
            cur_v = v_ref[:, seq_k_start:seq_k_end, :]

            cur_O = torch.zeros((cur_q_len, Nq, Dv), device=q.device, dtype=torch.float32)

            m_prev = torch.full((G, Nk, cur_q_len), float("-inf"), device=q.device, dtype=torch.float32)
            l_prev = torch.zeros((G, Nk, cur_q_len), device=q.device, dtype=torch.float32)
            acc_o = torch.zeros((G, Nk, cur_q_len, Dv), device=q.device, dtype=torch.float32)

            if dropout_p > 0.0:
                mask_shape = (G, Nk, cur_q_len, cur_k_len)
                d_mask = torch.empty(mask_shape, dtype=torch.bool, device=q.device).bernoulli_(1 - dropout_p)
                dropout_masks.append(d_mask)
            else:
                d_mask = None

            for q0 in range(0, cur_q_len, block_q_len):
                ql = min(block_q_len, cur_q_len - q0)
                q_block = cur_q[:, :, q0 : q0 + ql, :]

                m_i = m_prev[:, :, q0 : q0 + ql]
                l_i = l_prev[:, :, q0 : q0 + ql]
                acc_i = acc_o[:, :, q0 : q0 + ql, :]

                for k0 in range(0, cur_k_len, block_k_len):
                    kl = min(block_k_len, cur_k_len - k0)
                    k_block = cur_k[:, k0 : k0 + kl, :]
                    v_block = cur_v[:, k0 : k0 + kl, :]

                    s_block = torch.einsum("g h q d, h k d -> g h q k", q_block, k_block) * softmax_scale

                    if causal:
                        q_idx = torch.arange(q0, q0 + ql, device=q.device)[:, None] + seq_q_start
                        k_idx = torch.arange(k0, k0 + kl, device=q.device)[None, :] + seq_k_start
                        mask = k_idx > q_idx
                        s_block = s_block.masked_fill(mask[None, None, :, :], float("-inf"))

                    m_curr = s_block.amax(dim=-1)
                    m_new = torch.maximum(m_i, m_curr)

                    alpha = torch.exp(m_i - m_new)

                    p = torch.exp(s_block - m_new[..., None])

                    if d_mask is not None:
                        p = p * d_mask[:, :, q0 : q0 + ql, k0 : k0 + kl] * dropout_scale

                    row_sum = p.sum(dim=-1)
                    l_new = alpha * l_i + row_sum

                    pv = torch.einsum("g h q k, h k d -> g h q d", p.to(v.dtype), v_block)
                    acc_new = alpha[..., None] * acc_i + pv

                    m_i = m_new
                    l_i = l_new
                    acc_i = acc_new

                m_prev[:, :, q0 : q0 + ql] = m_i
                l_prev[:, :, q0 : q0 + ql] = l_i
                acc_o[:, :, q0 : q0 + ql, :] = acc_i

            block_out = acc_o / l_prev[..., None]

            cur_O_final = block_out.permute(1, 0, 2, 3).reshape(Nq, cur_q_len, Dv).permute(1, 0, 2)

            O[seq_q_start:seq_q_end] = cur_O_final.to(q.dtype)

            cur_L_final = (m_prev + torch.log(l_prev)).permute(1, 0, 2).reshape(Nq, cur_q_len).t()
            L[seq_q_start:seq_q_end] = cur_L_final

        ctx.cu_seqlens_q = cu_seqlens_q_cpu
        ctx.cu_seqlens_k = cu_seqlens_k_cpu
        ctx.dropout_p = dropout_p
        ctx.dropout_masks = dropout_masks
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.block_q_len = block_q_len
        ctx.block_k_len = block_k_len

        ctx.save_for_backward(q_ref, k_ref, v_ref, O, L)

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

        G, Nk, Sq, Dq = q.shape
        Nk_k, Sk, Dk = k.shape
        Nq = G * Nk

        grad_q_accum = torch.zeros_like(q)
        grad_k_accum = torch.zeros_like(k)
        grad_v_accum = torch.zeros_like(v)

        grad_o = grad_o.permute(1, 0, 2).reshape(Nk, G, Sq, Dq).permute(1, 0, 2, 3)

        O = O.permute(1, 0, 2).reshape(Nk, G, Sq, Dq).permute(1, 0, 2, 3)

        L = L.t().reshape(Nk, G, Sq).permute(1, 0, 2)

        B = len(cu_seqlens_q) - 1
        dropout_scale = 1 / (1 - dropout_p) if dropout_p > 0.0 else None

        for b in range(B):
            seq_q_start, seq_q_end = cu_seqlens_q[b], cu_seqlens_q[b + 1]
            seq_k_start, seq_k_end = cu_seqlens_k[b], cu_seqlens_k[b + 1]

            cur_q_len = seq_q_end - seq_q_start
            cur_k_len = seq_k_end - seq_k_start

            cur_q = q[:, :, seq_q_start:seq_q_end, :]
            cur_k = k[:, seq_k_start:seq_k_end, :]
            cur_v = v[:, seq_k_start:seq_k_end, :]
            cur_o = O[:, :, seq_q_start:seq_q_end, :]
            cur_do = grad_o[:, :, seq_q_start:seq_q_end, :]
            cur_l = L[:, :, seq_q_start:seq_q_end]

            D_vec = (cur_do * cur_o).sum(dim=-1)

            d_mask = dropout_masks[b] if dropout_masks else None

            for q0 in range(0, cur_q_len, block_q_len):
                ql = min(block_q_len, cur_q_len - q0)
                q_blk = cur_q[:, :, q0 : q0 + ql, :]
                do_blk = cur_do[:, :, q0 : q0 + ql, :]
                l_blk = cur_l[:, :, q0 : q0 + ql]
                d_blk = D_vec[:, :, q0 : q0 + ql]

                for k0 in range(0, cur_k_len, block_k_len):
                    kl = min(block_k_len, cur_k_len - k0)
                    k_blk = cur_k[:, k0 : k0 + kl, :]
                    v_blk = cur_v[:, k0 : k0 + kl, :]

                    s_blk = torch.einsum("g h q d, h k d -> g h q k", q_blk, k_blk) * softmax_scale

                    if causal:
                        q_idx = torch.arange(q0, q0 + ql, device=q.device)[:, None] + seq_q_start
                        k_idx = torch.arange(k0, k0 + kl, device=q.device)[None, :] + seq_k_start
                        mask = k_idx > q_idx
                        s_blk = s_blk.masked_fill(mask[None, None, :, :], float("-inf"))

                    p_blk = torch.exp(s_blk - l_blk[..., None])

                    if d_mask is not None:
                        p_blk = p_blk * d_mask[:, :, q0 : q0 + ql, k0 : k0 + kl] * dropout_scale

                    dv_contribution = torch.einsum("g h q k, g h q d -> h k d", p_blk, do_blk)
                    grad_v_accum[:, seq_k_start + k0 : seq_k_start + k0 + kl, :] += dv_contribution

                    dp_blk = torch.einsum("g h q d, h k d -> g h q k", do_blk, v_blk)
                    if d_mask is not None:
                        dp_blk = dp_blk * d_mask[:, :, q0 : q0 + ql, k0 : k0 + kl] * dropout_scale

                    ds_blk = p_blk * (dp_blk - d_blk[..., None])
                    ds_blk = ds_blk * softmax_scale

                    dq_contribution = torch.einsum("g h q k, h k d -> g h q d", ds_blk, k_blk)
                    grad_q_accum[:, :, seq_q_start + q0 : seq_q_start + q0 + ql, :] += dq_contribution

                    dk_contribution = torch.einsum("g h q k, g h q d -> h k d", ds_blk, q_blk)
                    grad_k_accum[:, seq_k_start + k0 : seq_k_start + k0 + kl, :] += dk_contribution

        grad_q = grad_q_accum.permute(1, 0, 2, 3).reshape(Nq, Sq, Dq).permute(1, 0, 2)
        grad_k = grad_k_accum.permute(1, 0, 2)
        grad_v = grad_v_accum.permute(1, 0, 2)

        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None
