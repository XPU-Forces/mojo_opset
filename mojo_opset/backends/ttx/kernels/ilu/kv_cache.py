# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Dense KV cache write from fused QKV (aligned with trike store_kv_cache).

from typing import Tuple

import torch
import triton
import triton.language as tl

from .utils import libentry


@libentry()
@triton.jit
def _store_kv_cache_fwd_kernel(
    k_cache_ptr,
    v_cache_ptr,
    qkv_ptr,
    kv_len_ptr,
    kv_idx_ptr,
    seq_len,
    qkv_stride_s,
    qkv_stride_h,
    qkv_stride_d,
    cache_stride_b,
    cache_stride_h,
    cache_stride_s,
    cache_stride_d,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
):
    s_id = tl.program_id(0)
    bz_id = s_id // seq_len
    kv_idx = tl.load(kv_idx_ptr + bz_id)

    if kv_idx < 0:
        return

    seq_id = s_id % seq_len
    pos_id = tl.load(kv_len_ptr + bz_id) + seq_id
    pad_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = pad_block < HEAD_DIM

    base_offs_k = s_id * qkv_stride_s + pad_block * qkv_stride_d
    base_offs_o = kv_idx * cache_stride_b + pos_id * cache_stride_s + pad_block * cache_stride_d
    for off_h in range(0, NUM_KV_HEADS):
        offs_k = (NUM_Q_HEADS + off_h) * qkv_stride_h
        k = tl.load(qkv_ptr + base_offs_k + offs_k, mask=mask, other=0.0)

        offs_v = base_offs_k + offs_k + NUM_KV_HEADS * qkv_stride_h
        v = tl.load(qkv_ptr + offs_v, mask=mask, other=0.0)

        offs_o = base_offs_o + off_h * cache_stride_h
        tl.store(k_cache_ptr + offs_o, k, mask=mask)
        tl.store(v_cache_ptr + offs_o, v, mask=mask)


def store_kv_cache_impl(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qkv: torch.Tensor,
    kv_len: torch.Tensor,
    kv_idx: torch.Tensor,
    num_q_head: int,
    num_kv_head: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bz, seq_len = qkv.shape[0], qkv.shape[1]
    head_dim = k_cache.shape[-1]
    grid = (bz * seq_len, 1, 1)

    qkv_2d = qkv.view(bz * seq_len, -1, head_dim)
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    _store_kv_cache_fwd_kernel[grid](
        k_cache,
        v_cache,
        qkv_2d,
        kv_len,
        kv_idx,
        seq_len,
        qkv_2d.stride(0),
        qkv_2d.stride(1),
        qkv_2d.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        num_q_head,
        num_kv_head,
        head_dim,
        padded_head_dim,
    )

    return k_cache, v_cache, qkv


# --- Paged KV cache ---


@libentry()
@triton.jit
def _store_paged_kv_cache_kernel(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cu_seqlens_ptr,
    kv_lens_ptr,
    stride_k_tok,
    stride_k_head,
    stride_k_dim,
    stride_v_tok,
    stride_v_head,
    stride_v_dim,
    stride_kc_blk,
    stride_kc_head,
    stride_kc_tok,
    stride_kc_dim,
    stride_vc_blk,
    stride_vc_head,
    stride_vc_tok,
    stride_vc_dim,
    stride_bt_batch,
    stride_bt_blk,
    num_kv_heads,
    batch_size,
    max_chunks_per_seq,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    chunk_id_linear = tl.program_id(0)
    batch_idx = chunk_id_linear // max_chunks_per_seq
    chunk_in_seq = chunk_id_linear % max_chunks_per_seq

    if batch_idx >= batch_size:
        return

    seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
    seq_end_tok = tl.load(cu_seqlens_ptr + batch_idx + 1)
    seq_len_curr = seq_end_tok - seq_start_tok

    token_offset_in_seq = chunk_in_seq * CHUNK_SIZE
    if token_offset_in_seq >= seq_len_curr:
        return

    if HAS_KV_LENS:
        logical_kv_start = tl.load(kv_lens_ptr + batch_idx).to(tl.int32)
        if logical_kv_start < 0:
            return
        logical_kv_start += token_offset_in_seq
    else:
        logical_kv_start = token_offset_in_seq

    global_token_idx = seq_start_tok + token_offset_in_seq
    valid_len = seq_len_curr - token_offset_in_seq

    curr_log_pos = logical_kv_start
    curr_kv_pos = global_token_idx

    remain_chunk_len = CHUNK_SIZE
    remain_chunk_len = tl.minimum(remain_chunk_len, valid_len)

    processed = 0
    while processed < remain_chunk_len:
        block_table_idx = curr_log_pos // block_size
        block_inner_off = curr_log_pos % block_size

        physical_block_id = tl.load(block_table_ptr + batch_idx * stride_bt_batch + block_table_idx * stride_bt_blk)
        valid_block = physical_block_id >= 0
        physical_block_id = tl.maximum(physical_block_id, 0)

        space_in_block = block_size - block_inner_off
        sub_len = tl.minimum(remain_chunk_len - processed, space_in_block).to(tl.int32)

        offs_sub = tl.arange(0, CHUNK_SIZE)
        mask_sub = offs_sub < sub_len

        offs_d = tl.arange(0, head_dim)

        for h in range(num_kv_heads):
            src_k_ptr = (
                k_ptr
                + (curr_kv_pos + offs_sub[:, None]) * stride_k_tok
                + h * stride_k_head
                + offs_d[None, :] * stride_k_dim
            )

            k_val = tl.load(src_k_ptr, mask=mask_sub[:, None], other=0.0)

            dst_k_ptr = (
                key_cache_ptr
                + physical_block_id * stride_kc_blk
                + h * stride_kc_head
                + (block_inner_off + offs_sub[:, None]) * stride_kc_tok
                + offs_d[None, :] * stride_kc_dim
            )

            tl.store(dst_k_ptr, k_val, mask=valid_block & mask_sub[:, None])

            src_v_ptr = (
                v_ptr
                + (curr_kv_pos + offs_sub[:, None]) * stride_v_tok
                + h * stride_v_head
                + offs_d[None, :] * stride_v_dim
            )

            v_val = tl.load(src_v_ptr, mask=mask_sub[:, None], other=0.0)

            dst_v_ptr = (
                value_cache_ptr
                + physical_block_id * stride_vc_blk
                + h * stride_vc_head
                + (block_inner_off + offs_sub[:, None]) * stride_vc_tok
                + offs_d[None, :] * stride_vc_dim
            )

            tl.store(dst_v_ptr, v_val, mask=valid_block & mask_sub[:, None])

        processed += sub_len
        curr_log_pos += sub_len
        curr_kv_pos += sub_len


def store_paged_kv_impl(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kv_lens_before_store: torch.Tensor,
):
    assert k_states.is_contiguous() and v_states.is_contiguous()

    if cu_seqlens is None:
        cu_seqlens = torch.arange(k_states.shape[0] + 1, device=k_states.device, dtype=torch.int32)

    num_kv_heads = k_states.shape[1]
    head_dim = k_states.shape[2]

    block_size = key_cache.shape[2]

    batch_size: int = block_table.shape[0]
    max_chunks_per_seq = block_table.shape[1]

    grid = (batch_size * max_chunks_per_seq,)

    _store_paged_kv_cache_kernel[grid](
        k_states,
        v_states,
        key_cache,
        value_cache,
        block_table,
        cu_seqlens,
        kv_lens_before_store,
        k_states.stride(0),
        k_states.stride(1),
        k_states.stride(2),
        v_states.stride(0),
        v_states.stride(1),
        v_states.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        num_kv_heads,
        batch_size,
        max_chunks_per_seq,
        head_dim,
        block_size,
        CHUNK_SIZE=block_size,
        HAS_KV_LENS=kv_lens_before_store is not None,
    )

    return key_cache, value_cache
