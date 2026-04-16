import torch

from typing import Optional

import triton
import triton.language as tl
from mojo_opset.backends.ttx.kernels.mlu.utils import get_mlu_total_cores

import pdb


"""
    load q

    loop:
        load k
        q: [tile_q, head_size_qk] @ k: [tile_k, head_size_qk]
        
        cmax = max(qk, rmax)
        p = exp(qk - cmax)
        csum = sum(p)

        alpha = exp(rmax - max)
        rsum = rsum * alpha + csum

        load v
        qkv = p: [tile_q, tile_k] @ v: [head_size_qk, tile_k]
        o = o * alpha + qkv
    end

    o = o / rsum
"""
"""
    # task partition
    total_tasks = bs * kv_heads
    core_nums = tl.num_programs(0)
    tasks_per_core = total_tasks // core_nums
    remained_tasks = total_tasks - tasks_per_core * core_nums
    if tl.program_id(0) < remained_tasks:
        tasks_per_core += 1
    task_begin = tl.program_id(0) * tasks_per_core + (remained_tasks if tl.program_id(0) >= remained_tasks else 0)
    task_end = task_begin + tasks_per_core

    # task_id, batch_id, kv_head_id
    task_ids = tl.arange(0, tasks_per_core) + task_begin
    batch_ids = task_ids // kv_heads
    kv_head_ids = task_ids % kv_heads
"""

@triton.jit
def paged_attention_decode_kernel(
    q_ptr,
    kcache_ptr,
    vcache_ptr,
    o_ptr,
    block_table_ptr,
    context_lens_ptr,
    softmax_scale,
    block_size: tl.constexpr,
    blocks_per_seq: tl.constexpr,
    bs,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_size_qk: tl.constexpr,
    head_size_vo: tl.constexpr,
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
    gqa_interleave,
    ):

    q_div_k : tl.constexpr = q_heads // kv_heads
    tile_q  : tl.constexpr = q_div_k
    tile_k  : tl.constexpr = 128
    seg_size: tl.constexpr = tile_k if tile_k < block_size else block_size # block_size is too large ???
    block_num_per_core: tl.constexpr = (tile_k + block_size - 1) // block_size
    block_num_pad_up  : tl.constexpr = (blocks_per_seq + block_num_per_core - 1 ) // block_num_per_core * block_num_per_core

    total_tasks = bs * kv_heads
    core_nums = tl.num_programs(0)

    for task_id in range(tl.program_id(0), total_tasks, core_nums):
        # load q
        batch_id, kv_head_id = task_id // kv_heads, task_id % kv_heads
        if gqa_interleave:
            q_head_ids = tl.arange(0, q_div_k) * kv_heads + kv_head_id
        else:
            q_head_ids = tl.arange(0, q_div_k) + kv_head_id * q_div_k

        q_offset = batch_id * stride_q_batch + q_head_ids * stride_q_head
        q_seg = tl.arange(0, head_size_qk)
        q_ptrs = q_ptr + q_offset[:, None] + q_seg[None, :]
        q_data = tl.load(q_ptrs)
        print(q_data.dtype)

        # seq_len
        context_len = tl.load(context_lens_ptr + batch_id)
        
        # load block_table
        block_table_offset = stride_bt_batch * batch_id
        block_table_seg = tl.arange(0, block_num_pad_up)
        block_table_ptrs = block_table_ptr + block_table_offset + block_table_seg
        block_ids = tl.load(block_table_ptrs, mask=block_table_seg < blocks_per_seq, other=0)

        # output
        o = tl.zeros((tile_q, head_size_vo), dtype=tl.float32)
        rmax, cmax = tl.full((tile_q,), -float('inf'), dtype=tl.float32), tl.zeros((tile_q,), dtype=tl.float32)
        rsum, csum = tl.zeros((tile_q,), dtype=tl.float32), tl.zeros((tile_q,), dtype=tl.float32)
        for k_begin in range(0, context_len, tile_k): # tile_k = m * block_size
            block_begin = k_begin // block_size

            for block_k_begin in range(0, block_size, tile_k): # block_k_begin > 0, when tile_k < block_size
                block_ids_per_core = block_ids[block_begin: block_begin + block_num_per_core]

                k_begin_real = k_begin + block_k_begin
                seq_k = tile_k if context_len > k_begin_real + tile_k else context_len - k_begin_real

                # load kcache
                kcache_offset = block_ids_per_core * stride_k_nblk + kv_head_id * stride_k_head + block_k_begin * stride_k_blksz
                kcache_seg = tl.arange(0, seg_size * head_size_qk)
                kcache_ptrs = kcache_ptr + kcache_offset[:, None] + kcache_seg[None, :]
                kcache_data = tl.load(kcache_ptrs)
                kcache_data = tl.reshape(kcache_data, (tile_k, head_size_qk))

                qk = tl.dot(q_data, tl.trans(kcache_data), out_dtype=tl.float32)
                cut_off_mask = tl.arange(0, tile_k) < seq_k
                neg_inf = tl.full((tile_k,), -float("inf"), dtype=tl.float32).cast(dtype=tl.int32, bitcast=True)
                neg_inf = neg_inf * (1 - cut_off_mask)
                neg_inf = neg_inf.cast(tl.float32, bitcast=True)
                qk = qk * cut_off_mask.to(tl.float32)[None, :] + neg_inf[None, :]
                qk *= softmax_scale

                cmax = tl.maximum(rmax, tl.max(qk, axis=1))
                p = tl.exp(qk - cmax[:, None])
                p_cast = p.to(kcache_data.dtype)
                csum = tl.sum(p, axis=1)

                alpha = tl.exp(rmax - cmax)
                rsum = rsum * alpha + csum
                rmax = cmax

                # load vcache
                vcache_offset = block_ids_per_core * stride_v_nblk + kv_head_id * stride_v_head + block_k_begin * stride_v_blksz
                vcache_seg = tl.arange(0, seg_size * head_size_vo)
                vcache_ptrs = vcache_ptr + vcache_offset[:, None] + vcache_seg[None, :]
                vcache_data = tl.load(vcache_ptrs)
                vcache_data = tl.reshape(vcache_data, (tile_k, head_size_vo))

                qkv = tl.dot(p_cast, vcache_data, out_dtype=tl.float32)

                o = o * alpha[:, None] + qkv
            # end for
        # end for
        if context_len > 0:
            o = o / rsum[:, None]

        lse = rmax + tl.log(rsum)
        # store output
        o_offset = batch_id * stride_o_batch + q_head_ids * stride_o_head
        o_seg = tl.arange(0, head_size_vo)
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

    print("\ncontext_len=", seqlens)
    print("blocks_per_seq=", blocks_per_seq)

    o = torch.empty((bs, q_heads, head_size_vo), dtype=q.dtype)

    stride_q_batch, stride_q_head, _ = q.stride()
    stride_k_nblk, stride_k_head, stride_k_blksz, stride_k_hsz = key_cache.stride()
    stride_v_nblk, stride_v_head, stride_v_blksz, stride_v_hsz = key_cache.stride()
    stride_o_batch, stride_o_head, _ = o.stride()
    stride_bt_batch, stride_bt_nblk = block_tables.stride()
    
    paged_attention_decode_kernel[grid_size](
        q,
        key_cache,
        value_cache,
        o,
        block_tables,
        seqlens,
        softmax_scale,
        block_size,
        blocks_per_seq,
        bs,
        q_heads,
        kv_heads,
        head_size_qk,
        head_size_vo,
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
        gqa_interleave,
    )

    return o