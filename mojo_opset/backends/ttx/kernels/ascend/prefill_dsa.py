import torch
import triton
import triton.language as tl
from triton.runtime.libentry import libentry


@libentry()
@triton.jit
def dsa_prefill_kernel(
    Q,
    K,
    V,
    O,
    Mask,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    stride_mask_b,
    stride_mask_s,
    softmax_scale,
    SEQLEN: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_batch = pid // HEAD_NUM
    cur_head = pid % HEAD_NUM

    offset_d_qk = tl.arange(0, QK_HEAD_DIM)
    offset_d_v = tl.arange(0, V_HEAD_DIM)
    offset_seq = tl.arange(0, BLOCK_N)

    for seq_start_q in range(0, SEQLEN, BLOCK_N):
        acc = tl.zeros([BLOCK_N, V_HEAD_DIM], dtype=tl.float32)
        lse = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")

        cur_seq_q = seq_start_q + offset_seq
        offset_q = cur_batch * stride_q_b + cur_head * stride_q_h + cur_seq_q[:, None] * stride_q_s + offset_d_qk[None, :]
        q = tl.load(Q + offset_q, mask=cur_seq_q[:, None] < SEQLEN, other=0.0)

        for seq_start_k in range(0, seq_start_q + 1, BLOCK_N):
            cur_seq_kv = seq_start_k + offset_seq
            offset_k = cur_batch * stride_k_b + cur_head * stride_k_h + cur_seq_kv[:, None] * stride_k_s + offset_d_qk[None, :]
            k = tl.load(K + offset_k, mask=cur_seq_kv[:, None] < SEQLEN, other=0.0)
            k = tl.trans(k)

            qk = tl.dot(q, k.to(q.dtype))
            qk *= softmax_scale

            offset_mask = cur_batch * stride_mask_b + cur_seq_q[:, None] * stride_mask_s + cur_seq_kv[None, :]
            qk_mask = tl.load(Mask + offset_mask, mask=(cur_seq_q[:, None] < SEQLEN) & (cur_seq_kv[None, :] < SEQLEN), other=False)

            qk = tl.where(qk_mask, qk, float("-inf"))

            offset_v = cur_batch * stride_v_b + cur_head * stride_v_h + cur_seq_kv[:, None] * stride_v_s + offset_d_v[None, :]
            v = tl.load(V + offset_v, mask=cur_seq_kv[:, None] < SEQLEN, other=0.0)

            local_max = tl.max(qk, 1)
            local_exp = tl.exp(qk - local_max[:, None])
            local_sum = tl.sum(local_exp, 1)
            local_lse = local_max + tl.log(local_sum)
            new_lse = tl.log(tl.exp(lse) + tl.exp(local_lse))
            new_lse = tl.where(new_lse != new_lse, lse, new_lse)

            rescale = tl.exp(lse - new_lse)
            rescale = tl.where(local_max == float("-inf"), 1.0, rescale)[:, None]
            acc *= rescale
            current_weight = tl.exp(local_lse - new_lse)
            acc_temp = tl.dot((local_exp / local_sum[:, None]).to(v.dtype), v) * current_weight[:, None]
            acc_temp = tl.where(acc_temp != acc_temp, 0.0, acc_temp)
            acc += acc_temp

            lse = new_lse

        acc = tl.where(acc != acc, 0.0, acc)
        offset_o = cur_batch * stride_o_b + cur_head * stride_o_h + cur_seq_q[:, None] * stride_o_s + offset_d_v[None, :]
        tl.store(O + offset_o, acc)


def ttx_prefill_dsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    bsz, seqlen, n_heads, qk_head_dim = q.shape
    v_head_dim = v.shape[-1]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    softmax_scale = qk_head_dim ** -0.5

    causal_mask = torch.tril(torch.ones((seqlen, seqlen), device=q.device, dtype=torch.bool))
    sparse_mask = torch.zeros((bsz, seqlen, seqlen), device=q.device, dtype=torch.bool))
    sparse_mask.scatter_(2, topk_indices, True)
    combined_mask = causal_mask.unsqueeze(0) & sparse_mask

    output = torch.zeros((bsz, n_heads, seqlen, v_head_dim), dtype=v.dtype, device=v.device)

    BLOCK_N = 16
    grid = (bsz * n_heads, )
    dsa_prefill_kernel[grid](
        q,
        k,
        v,
        output,
        combined_mask,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        combined_mask.stride(0),
        combined_mask.stride(1),
        softmax_scale,
        SEQLEN=seqlen,
        QK_HEAD_DIM=qk_head_dim,
        V_HEAD_DIM=v_head_dim,
        HEAD_NUM=n_heads,
        BLOCK_N=BLOCK_N,
    )
    return output.transpose(1, 2).contiguous()
