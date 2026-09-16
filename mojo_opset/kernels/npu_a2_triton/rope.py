from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._triton_utils import prepare_lens
from mojo_opset.kernels._triton_utils import rope_head_first
from mojo_opset.kernels._triton_utils import tensor_cache

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
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    kv_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
    sin_tile_1,
    sin_tile_2,
    cos_tile_1,
    cos_tile_2,
    head_num: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    inverse: tl.constexpr,
):
    x1 = tl.extra.cann.extension.extract_slice(x, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x2 = tl.extra.cann.extension.extract_slice(x, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])

    if inverse:
        roped_x1 = x1 * cos_tile_1 + x2 * sin_tile_2
        roped_x2 = x2 * cos_tile_2 - x1 * sin_tile_1
    else:
        roped_x1 = x1 * cos_tile_1 - x2 * sin_tile_1
        roped_x2 = x2 * cos_tile_2 + x1 * sin_tile_2

    x = tl.extra.cann.extension.insert_slice(x, roped_x1, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x = tl.extra.cann.extension.insert_slice(
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
    sin_tile_1,
    sin_tile_2,
    cos_tile_1,
    cos_tile_2,
    inverse: tl.constexpr,
):
    if inverse:
        roped_x1 = x1 * cos_tile_1 + x2 * sin_tile_2
        roped_x2 = x2 * cos_tile_2 - x1 * sin_tile_1
    else:
        roped_x1 = x1 * cos_tile_1 - x2 * sin_tile_1
        roped_x2 = x2 * cos_tile_2 + x1 * sin_tile_2
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


@triton.jit(do_not_specialize=["seq_len", "num_seq_blocks", "bs"])
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
    CACHE_HALVES_REPEATED: tl.constexpr,
    CAST_CACHE_TO_FP32: tl.constexpr,
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

        cos_block_1 = tl.load(
            cos_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        sin_block_1 = tl.load(
            sin_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        if CAST_CACHE_TO_FP32:
            cos_block_1 = cos_block_1.to(tl.float32)
            sin_block_1 = sin_block_1.to(tl.float32)
        if CACHE_HALVES_REPEATED:
            cos_block_2 = cos_block_1
            sin_block_2 = sin_block_1
        else:
            cos_block_2 = tl.load(
                cos_token_ptr + half_rope_dim + half_rope_dim_offsets[None, :],
                mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
                other=0,
            )
            sin_block_2 = tl.load(
                sin_token_ptr + half_rope_dim + half_rope_dim_offsets[None, :],
                mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
                other=0,
            )
            if CAST_CACHE_TO_FP32:
                cos_block_2 = cos_block_2.to(tl.float32)
                sin_block_2 = sin_block_2.to(tl.float32)

        head_q_offsets = tl.arange(0, n_qh)
        head_k_offsets = tl.arange(0, n_kh)

        cos_tile_1 = tl.reshape(cos_block_1, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
        cos_tile_2 = tl.reshape(cos_block_2, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
        sin_tile_1 = tl.reshape(sin_block_1, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
        sin_tile_2 = tl.reshape(sin_block_2, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)

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

            q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0).to(sin_block_1.dtype)
            q_tile = _compute_rope(
                q_tile, sin_tile_1, sin_tile_2, cos_tile_1, cos_tile_2,
                n_qh, half_rope_dim, TOKEN_BLOCK_SIZE, INVERSE,
            )
            tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

            k_offsets = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * head_dim
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            k_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & rope_dim_mask[None, None, :]

            k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0).to(sin_block_1.dtype)
            k_tile = _compute_rope(
                k_tile, sin_tile_1, sin_tile_2, cos_tile_1, cos_tile_2,
                n_kh, half_rope_dim, TOKEN_BLOCK_SIZE, INVERSE,
            )
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

            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0).to(sin_block_1.dtype)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0).to(sin_block_1.dtype)
            new_q_1, new_q_2 = _compute_rope_separated(
                q_tile_1, q_tile_2, sin_tile_1, sin_tile_2, cos_tile_1, cos_tile_2, INVERSE,
            )
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

            k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_half_mask, other=0.0).to(sin_block_1.dtype)
            k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_half_mask, other=0.0).to(sin_block_1.dtype)
            new_k_1, new_k_2 = _compute_rope_separated(
                k_tile_1, k_tile_2, sin_tile_1, sin_tile_2, cos_tile_1, cos_tile_2, INVERSE,
            )
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
    cu_q_lens: Optional[torch.Tensor] = None,
    seqlens_kv: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract position-specific cos/sin from the full embedding table.

    When cu_seqlens is given, a Triton kernel gathers cos/sin for each token
    using per-batch context offsets derived from seqlens_kv.
    """
    if position_ids is not None:
        return cos[position_ids], sin[position_ids]
    if cu_q_lens is None:
        return cos[:x.shape[1]], sin[:x.shape[1]]

    assert cu_q_lens.dtype == torch.int32
    seqlens_q = cu_q_lens[1:] - cu_q_lens[:-1]
    if seqlens_kv is not None:
        assert seqlens_kv.dtype == torch.int32
        context_lens = seqlens_kv - seqlens_q
    else:
        context_lens = None

    token_block_size = _get_token_block_size(1, 1)
    chunk_indices = prepare_chunk_indices(cu_q_lens, token_block_size, context_lens)
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


def _rope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
    cache_halves_repeated: bool,
    cast_cache_to_fp32: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k with pre-extracted cos/sin.

    Supports:
    - 4D padded prefill: q [B, S, N, D] or [B, N, S, D], cos [S, rope_dim]
    - 3D varlen:  q [T, N, D] or [N, T, D], cos [T, rope_dim]
    - 3D decode:  q [B, N, D] or [N, B, D], cos [B, rope_dim]
    """
    head_first = rope_head_first(cos.dim(), unsqueeze_dim)
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
    # The table is tuned for repeated cache halves. General unaligned caches
    # keep four sin/cos tiles live and need a smaller A2 local-memory tile.
    if not is_aligned and not cache_halves_repeated:
        token_block_size = min(token_block_size, 16)
    num_seq_blocks = (seq_len + token_block_size - 1) // token_block_size

    num_programs = get_num_cores()
    grid = (num_programs,)

    cos = cos.contiguous()
    sin = sin.contiguous()
    if cos.dim() == 3 and cos.shape[0] > 1:
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
        cache_halves_repeated,
        cast_cache_to_fp32,
    )

    if head_first:
        q = q.transpose(-2, -3).contiguous()
        k = k.transpose(-2, -3).contiguous()
    q = q.reshape(*orig_q_shape)
    k = k.reshape(*orig_k_shape)
    return q, k


def rope_train_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
    cache_halves_repeated: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _rope_fwd_impl(q, k, cos, sin, unsqueeze_dim, cache_halves_repeated, True)


def rope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
    keep_cos_sin_dtype: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del keep_cos_sin_dtype
    unsqueeze_dim = cos.dim() - 2 if head_first else cos.dim() - 1
    return _rope_fwd_impl(q, k, cos, sin, unsqueeze_dim, True, False)


def _rope_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
    cache_halves_repeated: bool,
    cast_cache_to_fp32: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass of RoPE with pre-extracted cos/sin."""
    head_first = rope_head_first(cos.dim(), unsqueeze_dim)
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
    if not is_aligned and not cache_halves_repeated:
        token_block_size = min(token_block_size, 16)
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
        cache_halves_repeated,
        cast_cache_to_fp32,
    )

    if head_first:
        dq = dq.transpose(-2, -3).contiguous()
        dk = dk.transpose(-2, -3).contiguous()
    dq = dq.reshape(*orig_q_shape)
    dk = dk.reshape(*orig_k_shape)
    return dq, dk


def rope_train_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
    cache_halves_repeated: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _rope_bwd_impl(dq, dk, cos, sin, unsqueeze_dim, cache_halves_repeated, True)


def rope_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    unsqueeze_dim = cos.dim() - 2 if head_first else cos.dim() - 1
    return _rope_bwd_impl(dq, dk, cos, sin, unsqueeze_dim, True, False)


@torch.library.custom_op("mojo_npu_triton_a2::apply_rope_fwd", mutates_args=())
def apply_rope_fwd(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return rope_train_fwd_impl(q, k, cos, sin, unsqueeze_dim, cache_halves_repeated=False)


@apply_rope_fwd.register_fake
def _apply_rope_fwd_fake(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    del cos, sin, unsqueeze_dim
    return (torch.empty_like(q, memory_format=torch.contiguous_format),
            torch.empty_like(k, memory_format=torch.contiguous_format))


@torch.library.custom_op("mojo_npu_triton_a2::apply_rope_bwd", mutates_args=())
def apply_rope_bwd(
    grad_q: torch.Tensor, grad_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return rope_train_bwd_impl(grad_q, grad_k, cos, sin, unsqueeze_dim, cache_halves_repeated=False)


@apply_rope_bwd.register_fake
def _apply_rope_bwd_fake(
    grad_q: torch.Tensor, grad_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    del cos, sin, unsqueeze_dim
    return (torch.empty_like(grad_q, memory_format=torch.contiguous_format),
            torch.empty_like(grad_k, memory_format=torch.contiguous_format))

@torch.library.custom_op("mojo_npu_triton_a2::apply_rope_infer_fwd", mutates_args=())
def apply_rope_infer_fwd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         head_first: bool, keep_cos_sin_dtype: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    return rope_fwd_impl(q, k, cos, sin, head_first, keep_cos_sin_dtype)


@apply_rope_infer_fwd.register_fake
def _rope_infer_fake(q, k, cos, sin, head_first, keep_cos_sin_dtype):
    return (torch.empty_like(q, memory_format=torch.contiguous_format),
            torch.empty_like(k, memory_format=torch.contiguous_format))


@torch.library.custom_op("mojo_npu_triton_a2::rotary_embedding_fwd", mutates_args=())
def rotary_embedding_fwd(x: torch.Tensor, inv_freq: torch.Tensor, cos: Optional[torch.Tensor],
                          sin: Optional[torch.Tensor], cu_q_lens: Optional[torch.Tensor],
                          total_seq_lens: Optional[torch.Tensor], position_ids: Optional[torch.Tensor],
                          attention_scaling: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        raise ValueError("Triton rotary_embedding requires precomputed cos/sin; set init_max_length")
    outputs = rot_pos_embed_impl(x, cos, sin, cu_q_lens=cu_q_lens, seqlens_kv=total_seq_lens,
                                  position_ids=position_ids)
    return outputs[0].clone(), outputs[1].clone()


@rotary_embedding_fwd.register_fake
def _rotary_embedding_fake(x, inv_freq, cos, sin, cu_q_lens, total_seq_lens, position_ids, attention_scaling):
    shape = (*position_ids.shape, inv_freq.shape[0] * 2) if position_ids is not None else (
        x.shape[0] if cu_q_lens is not None else x.shape[1], inv_freq.shape[0] * 2)
    return inv_freq.new_empty(shape), inv_freq.new_empty(shape)
