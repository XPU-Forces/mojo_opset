from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_lens
from mojo_opset.backends.ttx.kernels.utils import tensor_cache


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
    sin_tile,
    cos_tile,
    head_num: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    inverse: tl.constexpr,
):
    x1 = tl.extra.cann.extension.extract_slice(x, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x2 = tl.extra.cann.extension.extract_slice(x, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])

    if inverse:
        roped_x1 = x1 * cos_tile + x2 * sin_tile
        roped_x2 = x2 * cos_tile - x1 * sin_tile
    else:
        roped_x1 = x1 * cos_tile - x2 * sin_tile
        roped_x2 = x2 * cos_tile + x1 * sin_tile

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
    token_block_size_offsets = tl.arange(0, TOKEN_BLOCK_SIZE)

    for block_id in range(pid, total_blocks, grid_size):
        chunk_indices_ptr_offset = chunk_indices_ptr + block_id * 5
        chunk_idx = tl.load(chunk_indices_ptr_offset + 1)
        seq_start = tl.load(chunk_indices_ptr_offset + 2)
        context_len = tl.load(chunk_indices_ptr_offset + 3)
        actual_seq_len = tl.load(chunk_indices_ptr_offset + 4)

        block_start = chunk_idx * TOKEN_BLOCK_SIZE
        seq_offsets = block_start + token_block_size_offsets
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


@triton.autotune(
    configs=[
        triton.Config({"TOKEN_BLOCK_SIZE": 1}),
        triton.Config({"TOKEN_BLOCK_SIZE": 2}),
        triton.Config({"TOKEN_BLOCK_SIZE": 3}),
        triton.Config({"TOKEN_BLOCK_SIZE": 4}),
        triton.Config({"TOKEN_BLOCK_SIZE": 8}),
        triton.Config({"TOKEN_BLOCK_SIZE": 16}),
        triton.Config({"TOKEN_BLOCK_SIZE": 32}),
    ],
    key=["n_qh", "n_kh", "half_rope_dim"],
)
@triton.jit(do_not_specialize=["seq_len"])
def _rope_kernel(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_seq_stride,
    q_out_ptr,
    k_out_ptr,
    cos_ptr,
    cos_batch_stride,
    cos_seq_stride,
    sin_ptr,
    sin_batch_stride,
    sin_seq_stride,
    seq_len,
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

    num_seq_blocks = (seq_len + TOKEN_BLOCK_SIZE -1) // TOKEN_BLOCK_SIZE
    total_blocks = bs * num_seq_blocks

    token_block_size_offsets = tl.arange(0, TOKEN_BLOCK_SIZE)
    half_rope_dim_offsets = tl.arange(0, half_rope_dim)
    half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim
    head_q_offsets = tl.arange(0, n_qh)
    head_k_offsets = tl.arange(0, n_kh)
    if ALIGNED:
        rope_dim_offsets = tl.arange(0, rope_dim)
        rope_dim_mask = rope_dim_offsets < rope_dim
    for block_id in range(pid, total_blocks, grid_size):
        batch_idx = block_id // num_seq_blocks
        seq_block_id = block_id % num_seq_blocks

        block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
        seq_offsets = block_start_seq_idx + token_block_size_offsets
        seq_mask = seq_offsets < seq_len

        global_seq_offsets = seq_offsets

        cos_token_ptr = cos_ptr + batch_idx * cos_batch_stride + seq_offsets[:, None] * cos_seq_stride
        sin_token_ptr = sin_ptr + batch_idx * sin_batch_stride + seq_offsets[:, None] * sin_seq_stride

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

        # Copy the nope_dim (non-rotary) part from input to output.
        # In the non-inplace design, q_out is freshly allocated (empty_like),
        # so we must explicitly copy the nope_dim region. When nope_dim == 0
        # (rope_dim == head_dim), this block is eliminated at compile time.
        if nope_dim > 0:
            nope_dim_offsets = tl.arange(0, nope_dim)
            nope_dim_mask = nope_dim_offsets < nope_dim

            q_nope_offsets = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * head_dim
                + nope_dim_offsets[None, None, :]
            )
            q_nope_mask = seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & nope_dim_mask[None, None, :]
            q_nope_tile = tl.load(q_ptr + q_nope_offsets, mask=q_nope_mask, other=0.0)
            tl.store(q_out_ptr + q_nope_offsets, q_nope_tile, mask=q_nope_mask)

            k_nope_offsets = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * head_dim
                + nope_dim_offsets[None, None, :]
            )
            k_nope_mask = seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & nope_dim_mask[None, None, :]
            k_nope_tile = tl.load(k_ptr + k_nope_offsets, mask=k_nope_mask, other=0.0)
            tl.store(k_out_ptr + k_nope_offsets, k_nope_tile, mask=k_nope_mask)

        if ALIGNED:
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
            tl.store(q_out_ptr + q_offsets, q_tile, mask=q_mask)

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
            tl.store(k_out_ptr + k_offsets, k_tile, mask=k_mask)
        else:
            q_offsets_half1 = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * head_dim
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            q_offsets_half2 = q_offsets_half1 + half_rope_dim
            q_half_mask = (
                seq_mask[:, None, None] & (head_q_offsets[None, :, None] < n_qh) & half_rope_dim_mask[None, None, :]
            )

            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_q_1, new_q_2 = _compute_rope_separated(q_tile_1, q_tile_2, sin_tile, cos_tile, INVERSE)
            tl.store(q_out_ptr + q_offsets_half1, new_q_1, mask=q_half_mask)
            tl.store(q_out_ptr + q_offsets_half2, new_q_2, mask=q_half_mask)

            k_offsets_half1 = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * head_dim
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            k_offsets_half2 = k_offsets_half1 + half_rope_dim
            k_half_mask = (
                seq_mask[:, None, None] & (head_k_offsets[None, :, None] < n_kh) & half_rope_dim_mask[None, None, :]
            )

            k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_k_1, new_k_2 = _compute_rope_separated(k_tile_1, k_tile_2, sin_tile, cos_tile, INVERSE)
            tl.store(k_out_ptr + k_offsets_half1, new_k_1, mask=k_half_mask)
            tl.store(k_out_ptr + k_offsets_half2, new_k_2, mask=k_half_mask)


def _normalize_for_rope(
    x: torch.Tensor,
    head_first: bool,
) -> Tuple[torch.Tensor, int, int, bool, int, int, int, int]:
    """Normalize a tensor for the RoPE kernel.

    The kernel hardcodes head_dim as the head stride, so it requires the physical
    layout to be [B,S,N,D] (head stride == head_dim). This function detects whether
    the input already has this layout (optimized path, no transpose needed) or
    falls back to transpose+contiguous.

    Returns: (x, batch_stride, seq_stride, need_transpose_back, batch_size, seq_len, n_head, head_dim)
    """
    need_transpose_back = False

    if x.dim() == 4:
        batch_size = x.shape[0]
        if head_first:
            # Logical [B, N, S, D]
            n_head, seq_len, head_dim = x.shape[1], x.shape[2], x.shape[3]
            if x.stride(1) == head_dim:
                # Physical [B,S,N,D] (e.g. after transpose): head stride == head_dim
                batch_stride, seq_stride = x.stride(0), x.stride(2)
            else:
                # Physical [B,N,S,D] contiguous: need transpose to [B,S,N,D]
                need_transpose_back = True
                x = x.transpose(1, 2).contiguous()
                batch_stride, seq_stride = x.stride(0), x.stride(1)
        else:
            # [B, S, N, D]: already in the correct layout
            seq_len, n_head, head_dim = x.shape[1], x.shape[2], x.shape[3]
            batch_stride, seq_stride = x.stride(0), x.stride(1)
    else:
        assert x.dim() == 3
        batch_size = 1
        if head_first:
            # Logical [N, T, D]
            n_head, seq_len, head_dim = x.shape[0], x.shape[1], x.shape[2]
            if x.stride(0) == head_dim:
                # Physical [T,N,D] (e.g. after transpose): head stride == head_dim
                batch_stride, seq_stride = 0, x.stride(1)
            else:
                # Physical [N,T,D] contiguous: need transpose to [T,N,D]
                need_transpose_back = True
                x = x.transpose(0, 1).contiguous()
                batch_stride, seq_stride = 0, x.stride(0)
        else:
            # [T, N, D]: already in the correct layout
            seq_len, n_head, head_dim = x.shape[0], x.shape[1], x.shape[2]
            batch_stride, seq_stride = 0, x.stride(0)

    return x, batch_stride, seq_stride, need_transpose_back, batch_size, seq_len, n_head, head_dim


def _run_rope_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool,
    inverse: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared kernel launch logic for forward and backward RoPE.

    Allocates output buffers via empty_like (alloc only, no data copy) and launches
    the non-inplace kernel. If the input required transpose, transposes the output back.
    """
    q, q_bs, q_ss, need_tb, batch_size, seq_len, n_qh, head_dim = _normalize_for_rope(q, head_first)
    k, k_bs, k_ss, _, _, _, n_kh, _ = _normalize_for_rope(k, head_first)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    rope_dim = cos.shape[-1]
    nope_dim = head_dim - rope_dim
    half_rope_dim = rope_dim // 2

    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    cos_batch_stride = cos.stride(0) if (cos.dim() == 3 and cos.shape[0] > 1) else 0
    sin_batch_stride = sin.stride(0) if (sin.dim() == 3 and sin.shape[0] > 1) else 0

    grid = (get_num_cores(),)
    _rope_kernel[grid](
        q, q_bs, q_ss,
        k, k_bs, k_ss,
        q_out, k_out,
        cos, cos_batch_stride, cos.stride(-2),
        sin, sin_batch_stride, sin.stride(-2),
        seq_len, batch_size,
        n_qh, n_kh, head_dim, nope_dim, rope_dim, half_rope_dim,
        ALIGNED=True, INVERSE=inverse,
    )

    if need_tb:
        if q_out.dim() == 4:
            q_out = q_out.transpose(1, 2).contiguous()
            k_out = k_out.transpose(1, 2).contiguous()
        else:
            q_out = q_out.transpose(0, 1).contiguous()
            k_out = k_out.transpose(0, 1).contiguous()

    return q_out, k_out


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

    token_block_size = 32
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


def rope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
    keep_cos_sin_dtype: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k with pre-extracted cos/sin (forward pass).

    Supports 4D [B,S,N,D]/[B,N,S,D] and 3D [T,N,D]/[N,T,D] inputs.
    Uses a non-inplace kernel with empty_like output allocation to avoid clone overhead.

    When keep_cos_sin_dtype is True, cos/sin are NOT cast to q.dtype, preserving
    float32 precision for callers (e.g. indexer) that pass float32 cos/sin.
    """
    if not keep_cos_sin_dtype:
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
    return _run_rope_kernel(q, k, cos, sin, head_first, inverse=False)


def rope_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply inverse RoPE to dq/dk (backward pass).

    Uses a non-inplace kernel with empty_like output allocation to avoid clone overhead.
    """
    return _run_rope_kernel(dq, dk, cos, sin, head_first, inverse=True)
