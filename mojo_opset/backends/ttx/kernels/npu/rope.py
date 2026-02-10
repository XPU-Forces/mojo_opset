from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_lens
from mojo_opset.backends.ttx.kernels.utils import tensor_cache

ROPE_TOKEN_BLOCK_SIZE_TABLE = {
    (2, 1): 36,
    (4, 1): 16,
    (8, 1): 10,
    (16, 16): 5,
    (32, 32): 2,
    (64, 64): 1,
}


def _get_token_block_size(n_qh: int, n_kh: int) -> int:
    assert n_qh <= 84 and n_kh <= 84, "don't support head_num > 84, please raise an issue."

    if (n_qh, n_kh) in ROPE_TOKEN_BLOCK_SIZE_TABLE:
        return ROPE_TOKEN_BLOCK_SIZE_TABLE[(n_qh, n_kh)]

    for (q_thresh, k_thresh), block_size in sorted(
        ROPE_TOKEN_BLOCK_SIZE_TABLE.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        if n_qh <= q_thresh and n_kh <= k_thresh:
            return block_size

    return 1


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    kv_lens: Optional[torch.LongTensor] = None,
) -> torch.LongTensor:
    lens = prepare_lens(cu_seqlens)
    num_chunks = triton.cdiv(lens, chunk_size)
    total = num_chunks.sum()
    flat = torch.arange(total, device=cu_seqlens.device)
    seq_ids = torch.repeat_interleave(torch.arange(num_chunks.numel(), device=cu_seqlens.device), num_chunks)
    offsets = torch.cumsum(num_chunks, 0) - num_chunks
    chunk_indices = flat - offsets[seq_ids]

    seq_starts = cu_seqlens[:-1]
    seq_start_per_block = seq_starts[seq_ids]

    if kv_lens is not None:
        sin_cos_offset_per_block = kv_lens[seq_ids]
    else:
        sin_cos_offset_per_block = torch.zeros_like(seq_ids)

    return torch.stack([seq_ids, chunk_indices, seq_start_per_block, sin_cos_offset_per_block, lens[seq_ids]], dim=1)


@triton.jit
def _compute_rope(
    x,
    sin_tile,
    cos_tile,
    head_num: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    inverse: tl.constexpr,
):
    x1 = tl.extract_slice(x, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x2 = tl.extract_slice(x, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])

    if inverse:
        roped_x1 = x1 * cos_tile + x2 * sin_tile
        roped_x2 = x2 * cos_tile - x1 * sin_tile
    else:
        roped_x1 = x1 * cos_tile - x2 * sin_tile
        roped_x2 = x2 * cos_tile + x1 * sin_tile

    x = tl.insert_slice(x, roped_x1, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x = tl.insert_slice(
        x,
        roped_x2,
        [0, 0, half_rope_dim],
        [TOKEN_BLOCK_SIZE, head_num, half_rope_dim],
        [1, 1, 1],
    )

    return x


@triton.jit(do_not_specialize=["seq_len"])
def _rope_forward_kernel(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    num_seq_blocks,
    chunk_indices_ptr,
    kv_lens_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_blocks = bs * num_seq_blocks

    for block_id in range(pid, total_blocks, grid_size):
        if IS_VARLEN:
            chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
            seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
            sin_cos_offset = tl.load(chunk_indices_ptr + block_id * 5 + 3)
            actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

            block_start_seq_idx = chunk_idx * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < actual_seq_len

            global_seq_offsets = seq_start + seq_offsets

            sin_cos_seq_offsets = sin_cos_offset + seq_offsets
            cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
            sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride

            batch_idx = 0
        else:
            batch_idx = block_id // num_seq_blocks
            seq_block_id = block_id % num_seq_blocks

            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < seq_len

            global_seq_offsets = seq_offsets

            if HAS_KV_LENS:
                kv_len = tl.load(kv_lens_ptr + batch_idx)
                sin_cos_seq_offsets = kv_len + seq_offsets
                cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
                sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride
            else:
                sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
                cos_token_ptr = cos_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * cos_row_stride
                sin_token_ptr = sin_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * sin_row_stride

        half_rope_dim_offsets = tl.arange(0, half_rope_dim)
        half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim
        rope_dim_offsets = tl.arange(0, rope_dim)
        rope_dim_mask = rope_dim_offsets < rope_dim

        cos_block_2d = tl.load(
            cos_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        sin_block_2d = tl.load(
            sin_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )

        head_q_offsets = tl.arange(0, n_qh)
        head_k_offsets = tl.arange(0, n_kh)

        cos_tile = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
        sin_tile = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)

        q_offsets = (
            batch_idx * q_batch_stride
            + global_seq_offsets[:, None, None] * q_seq_stride
            + head_q_offsets[None, :, None] * hd
            + nope_dim
            + rope_dim_offsets[None, None, :]
        )
        q_mask = seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & rope_dim_mask[None, None, :]

        q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0).to(sin_block_2d.dtype)

        q_tile = _compute_rope(q_tile, sin_tile, cos_tile, n_qh, half_rope_dim, TOKEN_BLOCK_SIZE, False)

        tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

        k_offsets = (
            batch_idx * k_batch_stride
            + global_seq_offsets[:, None, None] * k_seq_stride
            + head_k_offsets[None, :, None] * hd
            + nope_dim
            + rope_dim_offsets[None, None, :]
        )
        k_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & rope_dim_mask[None, None, :]

        k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0).to(sin_block_2d.dtype)

        k_tile = _compute_rope(k_tile, sin_tile, cos_tile, n_kh, half_rope_dim, TOKEN_BLOCK_SIZE, False)

        tl.store(k_ptr + k_offsets, k_tile, mask=k_mask)


@triton.jit(do_not_specialize=["seq_len"])
def _rope_backward_kernel(
    dq_ptr,
    dq_batch_stride,
    dq_seq_stride,
    dk_ptr,
    dk_batch_stride,
    dk_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    num_seq_blocks,
    chunk_indices_ptr,
    kv_lens_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_blocks = bs * num_seq_blocks

    for block_id in range(pid, total_blocks, grid_size):
        if IS_VARLEN:
            chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
            seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
            sin_cos_offset = tl.load(chunk_indices_ptr + block_id * 5 + 3)
            actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

            block_start_seq_idx = chunk_idx * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < actual_seq_len

            global_seq_offsets = seq_start + seq_offsets

            sin_cos_seq_offsets = sin_cos_offset + seq_offsets
            cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
            sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride

            batch_idx = 0
        else:
            batch_idx = block_id // num_seq_blocks
            seq_block_id = block_id % num_seq_blocks

            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < seq_len

            global_seq_offsets = seq_offsets

            if HAS_KV_LENS:
                kv_len = tl.load(kv_lens_ptr + batch_idx)
                sin_cos_seq_offsets = kv_len + seq_offsets
                cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
                sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride
            else:
                sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
                cos_token_ptr = cos_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * cos_row_stride
                sin_token_ptr = sin_ptr + (sin_cos_batch_offset + seq_offsets[:, None]) * sin_row_stride

        half_rope_dim_offsets = tl.arange(0, half_rope_dim)
        half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim
        rope_dim_offsets = tl.arange(0, rope_dim)
        rope_dim_mask = rope_dim_offsets < rope_dim

        cos_block_2d = tl.load(
            cos_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        sin_block_2d = tl.load(
            sin_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )

        cos_tile = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
        sin_tile = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)

        head_q_offsets = tl.arange(0, n_qh)
        head_k_offsets = tl.arange(0, n_kh)

        dq_offsets = (
            batch_idx * dq_batch_stride
            + global_seq_offsets[:, None, None] * dq_seq_stride
            + head_q_offsets[None, :, None] * hd
            + nope_dim
            + rope_dim_offsets[None, None, :]
        )
        q_mask = seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & rope_dim_mask[None, None, :]

        dq_tile = tl.load(dq_ptr + dq_offsets, mask=q_mask, other=0).to(sin_block_2d.dtype)

        dq_tile = _compute_rope(dq_tile, sin_tile, cos_tile, n_qh, half_rope_dim, TOKEN_BLOCK_SIZE, True)

        tl.store(dq_ptr + dq_offsets, dq_tile, mask=q_mask)

        dk_offsets = (
            batch_idx * dk_batch_stride
            + global_seq_offsets[:, None, None] * dk_seq_stride
            + head_k_offsets[None, :, None] * hd
            + nope_dim
            + rope_dim_offsets[None, None, :]
        )
        k_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & rope_dim_mask[None, None, :]

        dk_tile = tl.load(dk_ptr + dk_offsets, mask=k_mask, other=0).to(sin_block_2d.dtype)

        dk_tile = _compute_rope(dk_tile, sin_tile, cos_tile, n_kh, half_rope_dim, TOKEN_BLOCK_SIZE, True)

        tl.store(dk_ptr + dk_offsets, dk_tile, mask=k_mask)


def rope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
    head_first: bool = True,
    rope_percentage: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    is_varlen = cu_seqlens is not None
    has_kv_lens = kv_lens is not None
    is_decode = False

    if is_varlen:
        assert q.dim() == 3 and k.dim() == 3, "q and k must be [total_seq_len, n_head, head_dim]."
        seq_len = q.shape[0]
        n_q_head, n_kv_head = q.shape[1], k.shape[1]
        head_dim = q.shape[2]
        batch_size = 1
        q_batch_stride, q_seq_stride = 0, q.stride(0)
        k_batch_stride, k_seq_stride = 0, k.stride(0)
    elif q.dim() == 3:
        batch_size = q.shape[0]
        n_q_head, n_kv_head = q.shape[1], k.shape[1]
        head_dim = q.shape[2]
        seq_len = 1
        is_decode = True
        q_batch_stride, q_seq_stride = q.stride(0), 0
        k_batch_stride, k_seq_stride = k.stride(0), 0
    else:
        assert q.dim() == 4 and k.dim() == 4, (
            "q and k must be [bs, n_head, seq_len, head_dim] if head_first else [bs, seq_len, n_head, head_dim]."
        )
        if head_first:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
        batch_size, seq_len, n_q_head, head_dim = q.shape
        n_kv_head = k.shape[2]
        q_batch_stride, q_seq_stride = q.stride(0), q.stride(1)
        k_batch_stride, k_seq_stride = k.stride(0), k.stride(1)

    rope_dim = int(head_dim * rope_percentage)
    assert rope_dim == cos.shape[-1]
    nope_dim = head_dim - rope_dim
    half_rope_dim = rope_dim // 2

    token_block_size = _get_token_block_size(n_q_head, n_kv_head)

    chunk_indices = prepare_chunk_indices(cu_seqlens, token_block_size, kv_lens) if is_varlen else None

    num_seq_blocks = chunk_indices.shape[0] if is_varlen else (seq_len + token_block_size - 1) // token_block_size

    num_programs = get_num_cores()

    grid = (num_programs,)

    cos_batch_size = cos.shape[0]
    cos = cos.contiguous()
    sin = sin.contiguous()

    _rope_forward_kernel[grid](
        q,
        q_batch_stride,
        q_seq_stride,
        k,
        k_batch_stride,
        k_seq_stride,
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        num_seq_blocks,
        chunk_indices,
        kv_lens,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        nope_dim,
        rope_dim,
        half_rope_dim,
        token_block_size,
        is_varlen,
        has_kv_lens,
    )

    if is_varlen or is_decode:
        return q, k
    elif head_first:
        return q.transpose(1, 2), k.transpose(1, 2)
    else:
        return q, k


def rope_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
    head_first: bool = True,
    rope_percentage: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    is_varlen = cu_seqlens is not None
    has_kv_lens = kv_lens is not None
    is_decode = False

    if is_varlen:
        assert dq.dim() == 3 and dk.dim() == 3, "dq and dk must be [total_seq_len, n_head, head_dim]."
        seq_len = dq.shape[0]
        n_q_head, n_kv_head = dq.shape[1], dk.shape[1]
        head_dim = dq.shape[2]
        batch_size = 1
        dq_batch_stride, dq_seq_stride = 0, dq.stride(0)
        dk_batch_stride, dk_seq_stride = 0, dk.stride(0)
    elif dq.dim() == 3:
        batch_size = dq.shape[0]
        n_q_head, n_kv_head = dq.shape[1], dk.shape[1]
        head_dim = dq.shape[2]
        seq_len = 1
        is_decode = True
        dq_batch_stride, dq_seq_stride = dq.stride(0), 0
        dk_batch_stride, dk_seq_stride = dk.stride(0), 0
    else:
        assert dq.dim() == 4 and dk.dim() == 4, (
            "dq and dk must be [bs, seq_len, n_head, head_dim]/[bs, n_head, seq_len, head_dim]."
        )
        if head_first:
            dq = dq.transpose(1, 2).contiguous()
            dk = dk.transpose(1, 2).contiguous()
        else:
            dq = dq.contiguous()
            dk = dk.contiguous()
        batch_size, seq_len, n_q_head, head_dim = dq.shape
        n_kv_head = dk.shape[2]
        dq_batch_stride, dq_seq_stride = dq.stride(0), dq.stride(1)
        dk_batch_stride, dk_seq_stride = dk.stride(0), dk.stride(1)

    rope_dim = int(head_dim * rope_percentage)
    nope_dim = head_dim - rope_dim
    half_rope_dim = rope_dim // 2

    token_block_size = _get_token_block_size(n_q_head, n_kv_head)

    chunk_indices = prepare_chunk_indices(cu_seqlens, token_block_size, kv_lens) if is_varlen else None

    num_seq_blocks = chunk_indices.shape[0] if is_varlen else (seq_len + token_block_size - 1) // token_block_size

    num_programs = get_num_cores()

    grid = (num_programs,)

    cos_batch_size = cos.shape[0]
    cos = cos.contiguous()
    sin = sin.contiguous()

    _rope_backward_kernel[grid](
        dq,
        dq_batch_stride,
        dq_seq_stride,
        dk,
        dk_batch_stride,
        dk_seq_stride,
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        num_seq_blocks,
        chunk_indices,
        kv_lens,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        nope_dim,
        rope_dim,
        half_rope_dim,
        token_block_size,
        is_varlen,
        has_kv_lens,
    )

    if is_varlen or is_decode:
        return dq, dk
    elif head_first:
        return dq.transpose(1, 2).contiguous(), dk.transpose(1, 2).contiguous()
    else:
        return dq, dk
