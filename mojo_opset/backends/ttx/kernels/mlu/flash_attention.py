import torch

from typing import Optional

import triton
import triton.language as tl
from mojo_opset.backends.ttx.kernels.mlu.utils import get_mlu_total_cores

@triton.jit
def paged_attention_decode_kernel(
    q_ptr,
    kcache_ptr,
    vcache_ptr,
    o_ptr,
    block_table_ptr,
    context_lens_ptr,
    softmax_scale,
    stride_q_batch,
    stride_q_head,
    stride_k_nblk,
    stride_k_head,
    stride_k_blksz,
    stride_v_nblk,
    stride_v_head,
    stride_v_blksz,
    stride_o_batch,
    stride_o_head,
    stride_bt_batch,
    stride_bt_nblk,
    gqa_interleave,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_SEQ: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_SIZE_QK: tl.constexpr,
    HEAD_SIZE_VO: tl.constexpr,
    TILE_K: tl.constexpr,
):
    Q_DIV_K: tl.constexpr = Q_HEADS // KV_HEADS
    TILE_Q: tl.constexpr = Q_DIV_K
    SEG_SIZE: tl.constexpr = TILE_K if TILE_K < BLOCK_SIZE else BLOCK_SIZE  # block_size is too large ???
    BLOCKS_ON_CORE: tl.constexpr = (TILE_K + BLOCK_SIZE - 1) // BLOCK_SIZE
    BLOCKS_PAD_UP: tl.constexpr = (BLOCKS_PER_SEQ + BLOCKS_ON_CORE - 1 ) // BLOCKS_ON_CORE * BLOCKS_ON_CORE + \
                                   (1 if BLOCKS_PER_SEQ == 0 else 0)  # blocks_pad_up == 0 ???
    
    total_tasks = batch_size * KV_HEADS
    core_nums = tl.num_programs(0)

    for task_id in range(tl.program_id(0), total_tasks, core_nums):
        batch_id, kv_head_id = task_id // KV_HEADS, task_id % KV_HEADS
        if gqa_interleave:
            q_head_ids = tl.arange(0, Q_DIV_K) * KV_HEADS + kv_head_id
        else:
            q_head_ids = tl.arange(0, Q_DIV_K) + kv_head_id * Q_DIV_K
        # load q
        q_offset = batch_id * stride_q_batch + q_head_ids * stride_q_head
        q_seg = tl.arange(0, HEAD_SIZE_QK)
        q_ptrs = q_ptr + q_offset[:, None] + q_seg[None, :]
        q_data = tl.load(q_ptrs)

        # seq_len
        context_len = tl.load(context_lens_ptr + batch_id)

        # load block_table
        block_table_offset = stride_bt_batch * batch_id
        block_table_seg = tl.arange(0, BLOCKS_PAD_UP)
        block_table_ptrs = block_table_ptr + block_table_offset + block_table_seg
        block_ids = tl.load(block_table_ptrs, mask=block_table_seg < BLOCKS_PER_SEQ, other=0)

        # output
        o = tl.zeros((TILE_Q, HEAD_SIZE_VO), dtype=tl.float32)
        rmax, cmax = tl.full((TILE_Q,), -float('inf'), dtype=tl.float32), tl.zeros((TILE_Q,), dtype=tl.float32)
        rsum, csum = tl.zeros((TILE_Q,), dtype=tl.float32), tl.zeros((TILE_Q,), dtype=tl.float32)

        for k_begin in range(0, context_len, TILE_K):
            seq_k = TILE_K if context_len > k_begin + TILE_K else context_len - k_begin

            block_begin = k_begin // BLOCK_SIZE
            block_id = block_ids[block_begin: block_begin + BLOCKS_ON_CORE]
            block_k_begin = k_begin % BLOCK_SIZE

            # load kcache, tile_k = seg_size * blocks_on_core
            kcache_offset = block_id * stride_k_nblk + kv_head_id * stride_k_head + block_k_begin * stride_k_blksz
            kcache_seg = tl.arange(0, SEG_SIZE * HEAD_SIZE_QK)
            kcache_ptrs = kcache_ptr + kcache_offset[:, None] + kcache_seg[None, :]
            kcache_data = tl.load(kcache_ptrs).reshape(TILE_K, HEAD_SIZE_QK)

            qk = tl.dot(q_data, tl.trans(kcache_data), out_dtype=tl.float32) * softmax_scale
            cut_off_mask = tl.arange(0, TILE_K) < seq_k
            neg_inf = tl.full((TILE_K,), -float("inf"), dtype=tl.float32).cast(dtype=tl.int32, bitcast=True)
            neg_inf = (neg_inf * (1 - cut_off_mask)).cast(tl.float32, bitcast=True)
            qk = qk * cut_off_mask.to(tl.float32)[None, :] + neg_inf[None, :]

            cmax = tl.maximum(rmax, tl.max(qk, axis=1))
            p = tl.exp(qk - cmax[:, None])
            p_cast = p.to(kcache_data.dtype)
            csum = tl.sum(p, axis=1)

            alpha = tl.exp(rmax - cmax)
            rsum = rsum * alpha + csum
            rmax = cmax

            # load vcache
            vcache_offset = block_id * stride_v_nblk + kv_head_id * stride_v_head + block_k_begin * stride_v_blksz
            vcache_seg = tl.arange(0, SEG_SIZE * HEAD_SIZE_VO)
            vcache_ptrs = vcache_ptr + vcache_offset[:, None] + vcache_seg[None, :]
            vcache_data = tl.load(vcache_ptrs).reshape(TILE_K, HEAD_SIZE_VO)

            qkv = tl.dot(p_cast, vcache_data, out_dtype=tl.float32)

            o = o * alpha[:, None] + qkv
        # end for
        lse = rmax
        if context_len > 0:
            o = o / rsum[:, None]
            lse += tl.log(rsum)

        # store output
        o_offset = batch_id * stride_o_batch + q_head_ids * stride_o_head
        o_seg = tl.arange(0, HEAD_SIZE_VO)
        o_ptrs = o_ptr + o_offset[:, None] + o_seg[None, :]
        tl.store(o_ptrs, o.to(q_data.dtype))
    # end for


def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    grid_size = (get_mlu_total_cores(),)
    
    bs, q_heads, head_size_qk = q.shape
    _, kv_heads, block_size, head_size_vo = key_cache.shape
    _, blocks_per_seq = block_tables.shape

    o = torch.empty((bs, q_heads, head_size_vo), dtype=q.dtype)

    stride_q_batch, stride_q_head, _ = q.stride()
    stride_k_nblk, stride_k_head, stride_k_blksz, stride_k_hsz = key_cache.stride()
    stride_v_nblk, stride_v_head, stride_v_blksz, stride_v_hsz = value_cache.stride()
    stride_o_batch, stride_o_head, _ = o.stride()
    stride_bt_batch, stride_bt_nblk = block_tables.stride()

    tile_k = 512

    paged_attention_decode_kernel[grid_size](
        q,
        key_cache,
        value_cache,
        o,
        block_tables,
        seqlens,
        softmax_scale,
        stride_q_batch,
        stride_q_head,
        stride_k_nblk,
        stride_k_head,
        stride_k_blksz,
        stride_v_nblk,
        stride_v_head,
        stride_v_blksz,
        stride_o_batch,
        stride_o_head,
        stride_bt_batch,
        stride_bt_nblk,
        gqa_interleave,
        bs,
        block_size,
        blocks_per_seq,
        q_heads,
        kv_heads,
        head_size_qk,
        head_size_vo,
        tile_k
    )

    return o
