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
# Same algorithm as NPU store_lowrank; grid[0] = ceil(n / BLOCK) programs.

import os

import torch
import triton
import triton.language as tl


def _store_label_cache_max_batch_bytes() -> int:
    """Upper bound on estimated per-program tensor footprint (see ``store_label_cache_infer_impl``).

    Compared against ``BLOCK * head_num * head_dim * element_size()`` (bytes). Not read from
    hardware; override with env ``MOJO_STORE_LABEL_CACHE_MAX_BATCH_BYTES`` if needed.
    """
    raw = os.environ.get("MOJO_STORE_LABEL_CACHE_MAX_BATCH_BYTES")
    if raw is not None:
        try:
            return max(1, int(raw, 10))
        except ValueError:
            pass
    return 192


@triton.jit
def _store_label_cache_triton_kernel(
    label_cache_ptr,
    key_lr_ptr,
    block_idx_list_ptr,
    token_idx_list_ptr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    token_num: tl.constexpr,
    l_stride_b: tl.constexpr,
    l_stride_h: tl.constexpr,
    l_stride_t: tl.constexpr,
    l_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    BATCH_BLOCK_NUM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    b_start = pid_b * BATCH_BLOCK_NUM
    b_end = tl.minimum(b_start + BATCH_BLOCK_NUM, token_num)
    b = tl.arange(0, BATCH_BLOCK_NUM) + b_start
    b_3d = b[:, None, None]
    h = tl.arange(0, head_num)
    h_3d = h[None, :, None]
    d = tl.arange(0, head_dim)
    d_3d = d[None, None, :]
    block_idx = tl.load(block_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)
    token_idx = tl.load(token_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)

    label_cache_addr = block_idx * l_stride_b + h_3d * l_stride_h + token_idx * l_stride_t + d_3d * l_stride_d
    key_lr_offset = b_3d * k_stride_s + h_3d * k_stride_h + d_3d * k_stride_d

    valid_mask = (b_3d < b_end) & (h_3d < head_num) & (d_3d < head_dim)

    key_lr_data = tl.load(key_lr_ptr + key_lr_offset, mask=valid_mask, other=0.0)
    tl.store(label_cache_ptr + label_cache_addr, key_lr_data, mask=valid_mask)


def store_label_cache_infer_impl(
    label_cache: torch.Tensor,
    key_lr: torch.Tensor,
    block_idxs: torch.Tensor,
    token_idxs: torch.Tensor,
    token_num: int,
):
    """
    Store label cache for each token (ILU Triton).

    Args:
        label_cache: [block_num, head_num, block_size, head_dim]
        key_lr: [seqlen, head_num, head_dim]
        block_idxs: [token_num]
        token_idxs: [token_num]
        token_num: number of tokens to store

    Returns:
        label_cache (updated in place)

    Heuristic:
        ``BLOCK`` is chosen so that ``BLOCK * head_num * head_dim * element_size`` (bytes) does
        not exceed a tunable bound (default ``192`` bytes via ``_store_label_cache_max_batch_bytes``).
        This is not derived from on-chip memory queries; set ``MOJO_STORE_LABEL_CACHE_MAX_BATCH_BYTES``
        to override.
    """
    block_num, head_num, block_size, head_dim = label_cache.shape
    assert label_cache.dtype == key_lr.dtype, "label_cache and key_lr must have the same dtype"

    n = token_num
    if n == 0:
        return label_cache

    max_batch_bytes = _store_label_cache_max_batch_bytes()

    # Initial tokens per program: ceil(n / K); keeps launch width ~O(K) without querying the device.
    TARGET_PROGRAMS = 256
    BLOCK = triton.cdiv(n, TARGET_PROGRAMS)

    # Rough per-program working-set estimate (bytes); if too large, cap BLOCK to reduce tile pressure.
    batch_data_size = BLOCK * head_num * head_dim * key_lr.element_size()
    if batch_data_size > max_batch_bytes:
        BLOCK = 16
    BLOCK = max(16, BLOCK)
    grid = (triton.cdiv(n, BLOCK),)

    BATCH_BLOCK_NUM = BLOCK

    _store_label_cache_triton_kernel[grid](
        label_cache_ptr=label_cache,
        key_lr_ptr=key_lr,
        block_idx_list_ptr=block_idxs,
        token_idx_list_ptr=token_idxs,
        head_num=head_num,
        head_dim=head_dim,
        token_num=token_num,
        l_stride_b=label_cache.stride(0),
        l_stride_h=label_cache.stride(1),
        l_stride_t=label_cache.stride(2),
        l_stride_d=label_cache.stride(3),
        k_stride_s=key_lr.stride(0),
        k_stride_h=key_lr.stride(1),
        k_stride_d=key_lr.stride(2),
        BATCH_BLOCK_NUM=BATCH_BLOCK_NUM,
    )
    return label_cache
