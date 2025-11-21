# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu, Hongmin Chen

from typing import Optional

import torch
import torch_npu
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mojo_opset.backends.ttx_kernels.src.ascend.linear_attn.utils import tensor_cache


@triton.autotune(
    configs=[triton.Config({})],
    key=["B"],
)
@triton.jit
def prepare_position_ids_kernel(y, cu_seqlens, B: tl.constexpr):
    i_n = tl.program_id(0)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o = tl.arange(0, B)
    for i in range(0, tl.cdiv(T, B) * B, B):
        o_i = o + i
        tl.store(y + bos + o_i, o_i, o_i < T)


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    return mask.sum(dim=-1, dtype=torch.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor, dtype: Optional[torch.dtype] = torch.int32
) -> torch.LongTensor:
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor, dtype: Optional[torch.dtype] = torch.int32
) -> torch.LongTensor:
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


@tensor_cache
def prepare_split_cu_seqlens(
    batch_size: int,
    seq_len: int,
    split_size: int,
    cu_seqlens: Optional[torch.LongTensor] = None,
    dtype: Optional[torch.dtype] = torch.int32,
    device: Optional[torch.device] = torch.device("cpu"),
) -> torch.LongTensor:
    if cu_seqlens is None:
        total_tokens = batch_size * seq_len
        cu_seqlens = list(range(0, total_tokens, seq_len)) + [total_tokens]
    else:
        cu_seqlens = cu_seqlens.tolist()
    return torch.tensor(
        [i for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:]) for i in range(bos, eos, split_size)]
        + [cu_seqlens[-1]],
        dtype=dtype,
        device=device,
    )


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat(
        [torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device) for n in prepare_lens(cu_seqlens).unbind()]
    )


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    position_ids = prepare_position_ids(cu_seqlens)
    return torch.stack([prepare_sequence_ids(cu_seqlens), position_ids], 1).to(cu_seqlens)


@triton.jit
def prepare_chunk_indices_and_offsets_fused_kernel(
    cu_seqlens, chunk_indices, chunk_offsets, batch_size, chunk_size, BLOCK_SIZE: tl.constexpr
):
    bos, chunk_num_offset = 0, 0

    tid = tl.arange(0, BLOCK_SIZE)
    tl.store(chunk_offsets + tid, 0, mask=tid < 1)

    for bid in range(batch_size):
        p_indices = chunk_indices + 2 * chunk_num_offset

        eos = tl.load(cu_seqlens + bid + 1).to(tl.int32)
        T = eos - bos
        bos = eos

        cur_chunk_num = (T + chunk_size - 1) // chunk_size
        chunk_num_offset = chunk_num_offset + cur_chunk_num

        tl.store(chunk_offsets + bid + tid + 1, chunk_num_offset, mask=tid < 1)

        iter_num = (cur_chunk_num + BLOCK_SIZE - 1) // BLOCK_SIZE
        for ii in range(iter_num):
            inner_chunk_start = ii * BLOCK_SIZE

            chunk_ids = tl.arange(0, BLOCK_SIZE) + inner_chunk_start
            store_id = chunk_ids * 2

            mask_ = chunk_ids < cur_chunk_num
            tl.store(p_indices + store_id, bid, mask=mask_)
            tl.store(p_indices + store_id + 1, chunk_ids, mask=mask_)


@tensor_cache
def prepare_chunk_indices_and_offsets(
    cu_seqlens: torch.LongTensor, cu_seqlens_host: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    host_chunk_num_list = triton.cdiv(prepare_lens(cu_seqlens_host), chunk_size).tolist()
    batch_size = cu_seqlens.shape[0] - 1
    chunk_num = 0
    for n in host_chunk_num_list:
        chunk_num = chunk_num + n
    indices = torch.empty([chunk_num, 2], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    offsets = torch.zeros_like(cu_seqlens)

    grid = (1, 1, 1)
    prepare_chunk_indices_and_offsets_fused_kernel[grid](
        cu_seqlens=cu_seqlens,
        chunk_indices=indices,
        chunk_offsets=offsets,
        batch_size=batch_size,
        chunk_size=chunk_size,
        BLOCK_SIZE=256,
    )
    return indices, offsets


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


@tensor_cache
def get_max_num_splits(cu_seqlens: torch.LongTensor, chunk_size: int) -> int:
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)
