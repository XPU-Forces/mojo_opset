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

SRAM_ALIGNMENT = 32


# When the half RoPE dimension satisfies the SRAM byte-alignment requirement,
# we can leverage a more efficient extension API to perform the RoPE computation.
def _is_half_rope_dim_aligned(half_rope_dim: int, dtype_size: int = 2) -> bool:
    return (half_rope_dim * dtype_size) % SRAM_ALIGNMENT == 0


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


@triton.jit
def _compute_rope_separated(
    x1,
    x2,
    sin_tile,
    cos_tile,
    inverse: tl.constexpr,
):
    if inverse:
        roped_x1 = x1 * cos_tile + x2 * sin_tile
        roped_x2 = x2 * cos_tile - x1 * sin_tile
    else:
        roped_x1 = x1 * cos_tile - x2 * sin_tile
        roped_x2 = x2 * cos_tile + x1 * sin_tile
    return roped_x1, roped_x2


@triton.jit
def _rot_pos_embed_kernel(
    cos_table_ptr,
    cos_table_stride,
    sin_table_ptr,
    sin_table_stride,
    cos_out_ptr,
    cos_out_stride,
    sin_out_ptr,
    sin_out_stride,
    chunk_indices_ptr,
    total_blocks,
    ROPE_DIM: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    """Gather position-specific cos/sin from the full embedding table.

    Each program handles blocks of tokens, reading per-block metadata from
    chunk_indices (5-column format from prepare_chunk_indices):
      [seq_id, chunk_idx, seq_start, context_len, actual_seq_len]
    """
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)

    dim_offsets = tl.arange(0, ROPE_DIM)

    for block_id in range(pid, total_blocks, grid_size):
        chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
        seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
        context_len = tl.load(chunk_indices_ptr + block_id * 5 + 3)
        actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

        block_start = chunk_idx * TOKEN_BLOCK_SIZE
        seq_offsets = block_start + tl.arange(0, TOKEN_BLOCK_SIZE)
        mask = seq_offsets < actual_seq_len

        table_positions = context_len + seq_offsets
        out_positions = seq_start + seq_offsets

        cos_vals = tl.load(
            cos_table_ptr + table_positions[:, None] * cos_table_stride + dim_offsets[None, :],
            mask=mask[:, None],
            other=0.0,
        )
        sin_vals = tl.load(
            sin_table_ptr + table_positions[:, None] * sin_table_stride + dim_offsets[None, :],
            mask=mask[:, None],
            other=0.0,
        )

        tl.store(
            cos_out_ptr + out_positions[:, None] * cos_out_stride + dim_offsets[None, :],
            cos_vals,
            mask=mask[:, None],
        )
        tl.store(
            sin_out_ptr + out_positions[:, None] * sin_out_stride + dim_offsets[None, :],
            sin_vals,
            mask=mask[:, None],
        )


@triton.jit(do_not_specialize=["seq_len"])
def _rope_inplace_kernel(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_seq_stride,
    cos_ptr,
    cos_batch_stride,
    cos_seq_stride,
    sin_ptr,
    sin_batch_stride,
    sin_seq_stride,
    seq_len,
    num_seq_blocks,
    bs,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    ALIGNED: tl.constexpr,
    INVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_blocks = bs * num_seq_blocks

    for block_id in range(pid, total_blocks, grid_size):
        batch_idx = block_id // num_seq_blocks
        seq_block_id = block_id % num_seq_blocks

        block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
        seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
        seq_mask = seq_offsets < seq_len

        global_seq_offsets = seq_offsets

        cos_token_ptr = cos_ptr + batch_idx * cos_batch_stride + seq_offsets[:, None] * cos_seq_stride
        sin_token_ptr = sin_ptr + batch_idx * sin_batch_stride + seq_offsets[:, None] * sin_seq_stride

        half_rope_dim_offsets = tl.arange(0, half_rope_dim)
        half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim

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

        if ALIGNED:
            rope_dim_offsets = tl.arange(0, rope_dim)
            rope_dim_mask = rope_dim_offsets < rope_dim

            q_offsets = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * head_dim
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            q_mask = seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & rope_dim_mask[None, None, :]

            q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0).to(sin_block_2d.dtype)
            q_tile = _compute_rope(q_tile, sin_tile, cos_tile, n_qh, half_rope_dim, TOKEN_BLOCK_SIZE, INVERSE)
            tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

            k_offsets = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * head_dim
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            k_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & rope_dim_mask[None, None, :]

            k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0).to(sin_block_2d.dtype)
            k_tile = _compute_rope(k_tile, sin_tile, cos_tile, n_kh, half_rope_dim, TOKEN_BLOCK_SIZE, INVERSE)
            tl.store(k_ptr + k_offsets, k_tile, mask=k_mask)
        else:
            q_offsets_half1 = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * head_dim
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            q_offsets_half2 = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * head_dim
                + nope_dim
                + half_rope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            q_half_mask = (
                seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & half_rope_dim_mask[None, None, :]
            )

            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_q_1, new_q_2 = _compute_rope_separated(q_tile_1, q_tile_2, sin_tile, cos_tile, INVERSE)
            tl.store(q_ptr + q_offsets_half1, new_q_1, mask=q_half_mask)
            tl.store(q_ptr + q_offsets_half2, new_q_2, mask=q_half_mask)

            k_offsets_half1 = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * head_dim
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            k_offsets_half2 = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * head_dim
                + nope_dim
                + half_rope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            k_half_mask = (
                seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & half_rope_dim_mask[None, None, :]
            )

            k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_k_1, new_k_2 = _compute_rope_separated(k_tile_1, k_tile_2, sin_tile, cos_tile, INVERSE)
            tl.store(k_ptr + k_offsets_half1, new_k_1, mask=k_half_mask)
            tl.store(k_ptr + k_offsets_half2, new_k_2, mask=k_half_mask)


def _normalize_to_bsnd(
    q: torch.Tensor,
    k: torch.Tensor,
    head_first: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int, int, int, int, int, int]:
    """Normalize q/k to [B, S, N, D] layout, returning strides and metadata."""

    if q.dim() == 3:
        assert k.dim() == 3
        if head_first:
            # [N, T, D] -> [BS, N, D]
            seq_len = q.shape[0]
            q = q.transpose(0, 1).clone(memory_format=torch.contiguous_format)
            k = k.transpose(0, 1).clone(memory_format=torch.contiguous_format)
        else:
            q = q.clone(memory_format=torch.contiguous_format)
            k = k.clone(memory_format=torch.contiguous_format)
        batch_size = 1
        seq_len, n_q_head, head_dim = q.shape
        n_kv_head = k.shape[1]
        q_batch_stride, q_seq_stride = 0, q.stride(0)
        k_batch_stride, k_seq_stride = 0, k.stride(0)
    else:
        assert q.dim() == 4 and k.dim() == 4
        if head_first:
            q = q.transpose(1, 2).clone(memory_format=torch.contiguous_format)
            k = k.transpose(1, 2).clone(memory_format=torch.contiguous_format)
        else:
            q = q.clone(memory_format=torch.contiguous_format)
            k = k.clone(memory_format=torch.contiguous_format)

        batch_size, seq_len, n_q_head, head_dim = q.shape
        n_kv_head = k.shape[2]
        q_batch_stride, q_seq_stride = q.stride(0), q.stride(1)
        k_batch_stride, k_seq_stride = k.stride(0), k.stride(1)
        

    return (
        q, k, batch_size, seq_len, n_q_head, n_kv_head, head_dim,
        q_batch_stride, q_seq_stride, k_batch_stride, k_seq_stride,
    )


def rot_pos_embed_impl(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqlens_kv: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract position-specific cos/sin from the full embedding table.

    When cu_seqlens is given, a Triton kernel gathers cos/sin for each token
    using per-batch context offsets derived from seqlens_kv.
    """
    if position_ids is not None:
        return cos[position_ids], sin[position_ids]
    if cu_seqlens_q is None:
        return cos[:x.shape[1]], sin[:x.shape[1]]

    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    if seqlens_kv is not None:
        context_lens = seqlens_kv - seqlens_q
    else:
        context_lens = None

    token_block_size = _get_token_block_size(1, 1)
    chunk_indices = prepare_chunk_indices(cu_seqlens_q, token_block_size, context_lens)
    total_blocks = chunk_indices.shape[0]
    rope_dim = cos.shape[-1]

    cos_out = torch.empty(x.shape[0], rope_dim, device=cos.device, dtype=cos.dtype)
    sin_out = torch.empty(x.shape[0], rope_dim, device=sin.device, dtype=sin.dtype)

    num_programs = min(total_blocks, get_num_cores())
    grid = (num_programs,)
    assert cos.dtype == torch.float32, "cos must be float32"

    _rot_pos_embed_kernel[grid](
        cos, cos.stride(0),
        sin, sin.stride(0),
        cos_out, cos_out.stride(0),
        sin_out, sin_out.stride(0),
        chunk_indices, total_blocks,
        rope_dim, token_block_size,
    )
    return cos_out, sin_out


def rope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k with pre-extracted cos/sin.

    Supports:
    - 4D padded prefill: q [B, S, N, D] or [B, N, S, D], cos [S, rope_dim]
    - 3D varlen:  q [T, N, D] or [N, T, D], cos [T, rope_dim]
    - 3D decode:  q [B, N, D] or [N, B, D], cos [B, rope_dim]
    """
    orig_q_shape = q.shape
    orig_k_shape = k.shape
    (
        q, k, batch_size, seq_len, n_q_head, n_kv_head, head_dim,
        q_batch_stride, q_seq_stride, k_batch_stride, k_seq_stride,
    ) = _normalize_to_bsnd(q, k, head_first)

    rope_dim = cos.shape[-1]
    nope_dim = head_dim - rope_dim
    half_rope_dim = rope_dim // 2

    is_aligned = _is_half_rope_dim_aligned(half_rope_dim)
    token_block_size = _get_token_block_size(n_q_head, n_kv_head)
    num_seq_blocks = (seq_len + token_block_size - 1) // token_block_size

    num_programs = get_num_cores()
    grid = (num_programs,)

    cos = cos.contiguous()
    sin = sin.contiguous()
    if cos.dim() == 3:
        cos_batch_stride = cos.stride(0)
        sin_batch_stride = sin.stride(0)
    else:
        cos_batch_stride = 0
        sin_batch_stride = 0

    _rope_inplace_kernel[grid](
        q,
        q_batch_stride,
        q_seq_stride,
        k,
        k_batch_stride,
        k_seq_stride,
        cos,
        cos_batch_stride,
        cos.stride(-2),
        sin,
        sin_batch_stride,
        sin.stride(-2),
        seq_len,
        num_seq_blocks,
        batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        nope_dim,
        rope_dim,
        half_rope_dim,
        token_block_size,
        is_aligned,
        False,
    )

    if head_first:
        q = q.transpose(-2, -3).contiguous()
        k = k.transpose(-2, -3).contiguous()
    q = q.reshape(*orig_q_shape)
    k = k.reshape(*orig_k_shape)
    return q, k


def rope_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass of RoPE with pre-extracted cos/sin."""
    orig_q_shape = dq.shape
    orig_k_shape = dk.shape
    (
        dq, dk, batch_size, seq_len, n_q_head, n_kv_head, head_dim,
        dq_batch_stride, dq_seq_stride, dk_batch_stride, dk_seq_stride,
    ) = _normalize_to_bsnd(dq, dk, head_first)

    rope_dim = cos.shape[-1]
    nope_dim = head_dim - rope_dim
    half_rope_dim = rope_dim // 2

    is_aligned = _is_half_rope_dim_aligned(half_rope_dim)
    token_block_size = _get_token_block_size(n_q_head, n_kv_head)
    num_seq_blocks = (seq_len + token_block_size - 1) // token_block_size

    num_programs = get_num_cores()
    grid = (num_programs,)

    cos = cos.contiguous()
    sin = sin.contiguous()
    if cos.dim() == 3:
        cos_batch_stride = cos.stride(0)
        sin_batch_stride = sin.stride(0)
    else:
        cos_batch_stride = 0
        sin_batch_stride = 0

    _rope_inplace_kernel[grid](
        dq,
        dq_batch_stride,
        dq_seq_stride,
        dk,
        dk_batch_stride,
        dk_seq_stride,
        cos,
        cos_batch_stride,
        cos.stride(-2),
        sin,
        sin_batch_stride,
        sin.stride(-2),
        seq_len,
        num_seq_blocks,
        batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        nope_dim,
        rope_dim,
        half_rope_dim,
        token_block_size,
        is_aligned,
        True,
    )

    if head_first:
        dq = dq.transpose(-2, -3).contiguous()
        dk = dk.transpose(-2, -3).contiguous()
    dq = dq.reshape(*orig_q_shape)
    dk = dk.reshape(*orig_k_shape)
    return dq, dk
