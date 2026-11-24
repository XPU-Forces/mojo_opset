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

from .utils import ilu_grid_dim_from_row_tasks, libentry

_MAX_CHUNK_TILE_ELEMS = 16384


def _floor_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)


def _get_chunk_size(block_size: int, head_dim: int) -> int:
    if head_dim <= 0:
        return 1

    chunk_cap = max(1, _MAX_CHUNK_TILE_ELEMS // head_dim)
    return min(block_size, _floor_power_of_two(chunk_cap))


def _get_num_subchunks(block_size: int, chunk_size: int) -> int:
    return max(1, triton.cdiv(block_size, chunk_size))


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
def _store_paged_kv_cache_chunk_kernel(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    chunk_meta_ptr,
    num_chunks,
    stride_cm_row,
    stride_cm_col,
    num_kv_heads,
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
    head_dim: tl.constexpr,
    num_subchunks: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offs_sub = tl.arange(0, CHUNK_SIZE)
    offs_d = tl.arange(0, head_dim)

    for chunk_idx in range(pid, num_chunks, num_programs):
        chunk_base = chunk_meta_ptr + chunk_idx * stride_cm_row
        src_token_start = tl.load(chunk_base + 0 * stride_cm_col)
        dst_block_id = tl.load(chunk_base + 1 * stride_cm_col)
        dst_block_offset = tl.load(chunk_base + 2 * stride_cm_col)
        chunk_len = tl.load(chunk_base + 3 * stride_cm_col)

        for subchunk_idx in range(num_subchunks):
            processed = subchunk_idx * CHUNK_SIZE
            token_offsets = processed + offs_sub
            mask_sub = token_offsets < chunk_len

            base_src_k_ptr = (
                k_ptr
                + (src_token_start + token_offsets[:, None]) * stride_k_tok
                + offs_d[None, :] * stride_k_dim
            )
            base_dst_k_ptr = (
                key_cache_ptr
                + dst_block_id * stride_kc_blk
                + (dst_block_offset + token_offsets[:, None]) * stride_kc_tok
                + offs_d[None, :] * stride_kc_dim
            )
            base_src_v_ptr = (
                v_ptr
                + (src_token_start + token_offsets[:, None]) * stride_v_tok
                + offs_d[None, :] * stride_v_dim
            )
            base_dst_v_ptr = (
                value_cache_ptr
                + dst_block_id * stride_vc_blk
                + (dst_block_offset + token_offsets[:, None]) * stride_vc_tok
                + offs_d[None, :] * stride_vc_dim
            )

            for h in range(num_kv_heads):
                k_val = tl.load(base_src_k_ptr + h * stride_k_head, mask=mask_sub[:, None], other=0.0)
                tl.store(base_dst_k_ptr + h * stride_kc_head, k_val, mask=mask_sub[:, None])

                v_val = tl.load(base_src_v_ptr + h * stride_v_head, mask=mask_sub[:, None], other=0.0)
                tl.store(base_dst_v_ptr + h * stride_vc_head, v_val, mask=mask_sub[:, None])

def _get_chunk_num_programs(num_chunks: int) -> int:
    return ilu_grid_dim_from_row_tasks(num_chunks)


def store_paged_kv_impl(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    chunk_metadata: torch.Tensor,
):
    assert k_states.is_contiguous() and v_states.is_contiguous()
    assert chunk_metadata.dim() == 2
    assert chunk_metadata.shape[1] == 4

    num_chunks = chunk_metadata.shape[0]
    if num_chunks == 0:
        return key_cache, value_cache

    num_kv_heads = k_states.shape[1]
    head_dim = k_states.shape[2]
    block_size = key_cache.shape[2]
    chunk_size = _get_chunk_size(block_size, head_dim)
    num_subchunks = _get_num_subchunks(block_size, chunk_size)

    grid = (_get_chunk_num_programs(num_chunks),)
    _store_paged_kv_cache_chunk_kernel[grid](
        k_states,
        v_states,
        key_cache,
        value_cache,
        chunk_metadata,
        num_chunks,
        chunk_metadata.stride(0),
        chunk_metadata.stride(1),
        num_kv_heads,
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
        head_dim=head_dim,
        num_subchunks=num_subchunks,
        CHUNK_SIZE=chunk_size,
    )

    return key_cache, value_cache
