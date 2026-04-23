from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.mlu.utils import get_mlu_total_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_lens
from mojo_opset.backends.ttx.kernels.utils import tensor_cache

ROPE_TOKEN_BLOCK_SIZE_TABLE = {
    (4, 1): 128,
    (8, 1): 96,
    (8, 2): 64,
    (16, 1): 32,
    (16, 8): 16,
    (32, 8): 16,
    (32, 4): 16,
    (32, 32): 16,
}

SRAM_ALIGNMENT = 32

def _get_token_block_size(n_qh: int, n_kh: int) -> int:
    assert n_qh <= 84 and n_kh <= 84, "don't support head_num > 84, please raise an issue."
    if (n_qh, n_kh) in ROPE_TOKEN_BLOCK_SIZE_TABLE:
        return ROPE_TOKEN_BLOCK_SIZE_TABLE[(n_qh, n_kh)]

    # Fallback: use smaller block size for larger head counts to stay within NRAM limits
    head_count = max(n_qh, n_kh)
    if head_count >= 64:
        return 8
    elif head_count > 32:
        return 16
    else:
        return 32

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
    inverse: tl.constexpr,
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

        # Process Q
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
        new_q_1, new_q_2 = _compute_rope_separated(q_tile_1, q_tile_2, sin_tile, cos_tile, inverse)
        tl.store(q_ptr + q_offsets_half1, new_q_1, mask=q_half_mask)
        tl.store(q_ptr + q_offsets_half2, new_q_2, mask=q_half_mask)

        # Process K
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
        new_k_1, new_k_2 = _compute_rope_separated(k_tile_1, k_tile_2, sin_tile, cos_tile, inverse)
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
    """Extract position-specific cos/sin from the full embedding table (MLU)."""
    if position_ids is not None:
        return cos[position_ids], sin[position_ids]
    if cu_seqlens_q is None:
        seq_len = x.shape[-2]
        return cos[:seq_len], sin[:seq_len]

    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    if seqlens_kv is not None:
        context_lens = seqlens_kv - seqlens_q
    else:
        context_lens = torch.zeros_like(seqlens_q, dtype=seqlens_q.dtype, device=seqlens_q.device)

    total = x.shape[0]
    rope_dim = cos.shape[-1]
    device = x.device
    dtype = cos.dtype
    cos_out = torch.empty((total, rope_dim), device=device, dtype=dtype)
    sin_out = torch.empty((total, rope_dim), device=device, dtype=dtype)
    for i in range(seqlens_q.numel()):
        start = int(cu_seqlens_q[i].item())
        end = int(cu_seqlens_q[i + 1].item())
        q_len = end - start
        ctx = int(context_lens[i].item())
        positions = torch.arange(ctx, ctx + q_len, device=device, dtype=torch.int64)
        cos_out[start:end] = cos[positions.to(cos.device)]
        sin_out[start:end] = sin[positions.to(sin.device)]
    return cos_out, sin_out


def rope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k with pre-extracted cos/sin (MLU).
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

    token_block_size = _get_token_block_size(n_q_head, n_kv_head)
    num_seq_blocks = (seq_len + token_block_size - 1) // token_block_size

    num_programs = get_mlu_total_cores()
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
        inverse=False,  # False for forward
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
    """Backward pass of RoPE with pre-extracted cos/sin (MLU)."""
    orig_dq_shape = dq.shape
    orig_dk_shape = dk.shape
    (
        dq, dk, batch_size, seq_len, n_q_head, n_kv_head, head_dim,
        dq_batch_stride, dq_seq_stride, dk_batch_stride, dk_seq_stride,
    ) = _normalize_to_bsnd(dq, dk, head_first)

    rope_dim = cos.shape[-1]
    nope_dim = head_dim - rope_dim
    half_rope_dim = rope_dim // 2

    token_block_size = _get_token_block_size(n_q_head, n_kv_head)
    num_seq_blocks = (seq_len + token_block_size - 1) // token_block_size

    num_programs = get_mlu_total_cores()
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
        inverse=True,  # True for backward
    )

    if head_first:
        dq = dq.transpose(-2, -3).contiguous()
        dk = dk.transpose(-2, -3).contiguous()
    dq = dq.reshape(*orig_dq_shape)
    dk = dk.reshape(*orig_dk_shape)
    return dq, dk