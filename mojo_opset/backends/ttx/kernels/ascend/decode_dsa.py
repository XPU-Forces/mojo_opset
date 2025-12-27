import torch
import triton
import triton.language as tl
from triton.runtime.libentry import libentry


@libentry()
@triton.jit
def dsa_decode_kernel(
    Q_nope,
    Q_pe,
    Wkv_b,
    Kv_cache,
    Pe_cache,
    Topk_indices,
    O,
    stride_q_nope_b,
    stride_q_nope_h,
    stride_q_pe_b,
    stride_q_pe_h,
    stride_wkv_b,
    stride_kv_nope_b,
    stride_kv_nope_s,
    stride_k_pe_b,
    stride_k_pe_s,
    stride_indices_b,
    stride_o_b,
    stride_o_h,
    softmax_scale: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM_NOPE: tl.constexpr,
    HEAD_DIM_ROPE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_batch = pid // HEAD_NUM
    cur_head = pid % HEAD_NUM

    offset_d_nope = tl.arange(0, HEAD_DIM_NOPE)
    offset_d_rope = tl.arange(0, HEAD_DIM_ROPE)
    offset_kv_lora = tl.arange(0, KV_LORA_RANK)

    # wkv_b = wkv_b.view(n_local_head, -1, kv_lora_rank)
    # q_nope = q_nope * wkv_b[:, :qk_nope_head_dim].T
    offset_q_nope = cur_batch * stride_q_nope_b + cur_head * stride_q_nope_h + offset_d_nope
    q_nope = tl.load(Q_nope + offset_q_nope)

    offset_wkv_b_q = cur_head * (HEAD_DIM_NOPE + HEAD_DIM_V) * stride_wkv_b + offset_d_nope[:, None] * stride_wkv_b + offset_kv_lora[None, :]
    wkv_b_q = tl.load(Wkv_b + offset_wkv_b_q)

    q_nope = tl.dot(q_nope[None, :], wkv_b_q.to(q_nope.dtype))

    # Load q_pe and topk_indices
    offset_q_pe = cur_batch * stride_q_pe_b + cur_head * stride_q_pe_h + offset_d_rope
    q_pe = tl.load(Q_pe + offset_q_pe)

    topk_indices = tl.load(Topk_indices + cur_batch * stride_indices_b + tl.arange(0, TOP_K))

    # Initilize acc and lse
    acc = tl.zeros([1, KV_LORA_RANK], dtype=tl.float32)
    lse = tl.zeros([1], dtype=tl.float32) - float("inf")

    for start_s in range(0, TOP_K, BLOCK_N):
        kv_cache_buf = tl.full((BLOCK_N, KV_LORA_RANK), 0.0, dtype=q_pe.dtype)
        pe_cache_buf = tl.full((BLOCK_N, HEAD_DIM_ROPE), 0.0, dtype=q_pe.dtype)
        for i in range(0, min(BLOCK_N, TOP_K - start_s)):
            offset_cache = tl.get_element(topk_indices, (start_s + i,))
            offset_kv_cache = cur_batch * stride_kv_nope_b + offset_cache * stride_kv_nope_s + offset_kv_lora
            kv_cache_temp = tl.load(Kv_cache + offset_kv_cache)
            kv_cache_buf = tl.insert_slice(kv_cache_buf, kv_cache_temp[None, :], offsets=(i, 0), sizes=(1, KV_LORA_RANK), strides=(1, 1))
            offset_pe_cache = cur_batch * stride_k_pe_b + offset_cache * stride_k_pe_s + offset_d_rope
            pe_cache_temp = tl.load(Pe_cache + offset_pe_cache)
            pe_cache_buf = tl.insert_slice(pe_cache_buf, pe_cache_temp[None, :], offsets=(i, 0), sizes=(1, HEAD_DIM_ROPE), strides=(1, 1))

        kv_cache = tl.trans(kv_cache_buf + 1e-6)
        pe_cache = tl.trans(pe_cache_buf + 1e-6)
        qk = tl.dot(q_pe[None, :], pe_cache)
        qk += tl.dot(q_nope, kv_cache.to(q_nope.dtype))
        qk = qk * softmax_scale

        # score = scores.softmax(dim=-1); o = scores * kv_cache[:bsz, :end_pos]
        local_max = tl.max(qk, 1)
        local_exp = tl.exp(qk - local_max[:, None])
        local_sum = tl.sum(local_exp, 1)
        local_lse = local_max + tl.log(local_sum)
        mean_lse = (lse + local_lse) / 2
        new_lse = mean_lse + tl.log(tl.exp(lse - mean_lse) + tl.exp(local_lse - mean_lse))
        new_lse = tl.where(new_lse != new_lse, local_lse, new_lse)

        acc = acc * tl.exp(lse - new_lse)[:, None]
        current_weight = tl.exp(local_lse - new_lse)
        acc += tl.dot((local_exp / local_sum[:, None]), tl.trans(kv_cache).to(q_nope.dtype)) * current_weight[:, None]
        lse = new_lse

    # o = o * wkv_b[:, -v_head_dim:].T
    offset_v = tl.arange(0, HEAD_DIM_V // 2)
    for i in range(0, 2):
        offset_wkv_b_o = (cur_head * (HEAD_DIM_NOPE + HEAD_DIM_V) + HEAD_DIM_NOPE) * stride_wkv_b + \
                        offset_v[:, None] * stride_wkv_b + offset_kv_lora[None, :]                                                                                                      :]
        wkv_b_o = tl.load(Wkv_b + offset_wkv_b_o)
        output = tl.dot(acc, tl.trans(wkv_b_o.to(acc.dtype)))

        offset_o = cur_batch * stride_o_b + cur_head * stride_o_h + offset_v[None, :]
        tl.store(O + offset_o, output)

        offset_v += HEAD_DIM_V // 2


def ttx_decode_dsa(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    wkv_b: torch.Tensor,
    kv_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
    bsz, _, n_heads, qk_nope_head_dim = q_nope.shape
    qk_rope_head_dim = q_pe.shape[-1]
    kv_lora_rank = kv_cache.shape[-1]
    v_head_dim = wkv_b.shape[0] // n_heads - qk_nope_head_dim
    top_k = topk_indices.shape[-1]
    softmax_scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

    q_nope = q_nope.transpose(1, 2).contiguous()
    q_pe = q_pe.transpose(1, 2).contiguous()
    output = torch.zeros((bsz, n_heads, 1, v_head_dim), device=q_nope.device)

    BLOCK_N = 16
    grid = (bsz * n_heads,)
    dsa_decode_kernel[grid](
        q_nope,
        q_pe,
        wkv_b,
        kv_cache,
        pe_cache,
        topk_indices,
        output,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        wkv_b.stride(0),,
        kv_cache.stride(0),
        kv_cache.stride(1),
        pe_cache.stride(0),
        pe_cache.stride(1),
        topk_indices.stride(0),
        output.stride(0),
        output.stride(1),
        softmax_scale = softmax_scale,
        HEAD_NUM = n_heads,
        HEAD_DIM_NOPE = qk_nope_head_dim,
        HEAD_DIM_ROPE = qk_rope_head_dim,
        HEAD_DIM_V = v_head_dim,
        KV_LORA_RANK = kv_lora_rank,
        BLOCK_N = BLOCK_N,
        TOP_K = top_k,
    )
    return output.transpose(1, 2).contiguous()
