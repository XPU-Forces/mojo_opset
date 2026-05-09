from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores

SRAM_ALIGNMENT = 32


def _is_half_rope_dim_aligned(half_rope_dim: int, dtype_size: int = 2) -> bool:
    return (half_rope_dim * dtype_size) % SRAM_ALIGNMENT == 0


def vision_rot_pos_embed_impl(
    inv_freq: torch.Tensor,
    grid_hw: torch.Tensor,
    rope_dim: int,
    adapooling_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin for Vision 2D RoPE from grid dimensions.

    Args:
        inv_freq: [rope_dim // 4] inverse frequency buffer.
        grid_hw: [B, 2] per-sample (H, W) in patches.
        rope_dim: full rope head dimension (must be divisible by 4).
        adapooling_factor: window size for adapooling regrouping.

    Returns:
        (cos, sin): each [total_tokens, rope_dim] in float32.
    """
    device = inv_freq.device
    if device.type == "cpu":
        device = grid_hw.device

    grid_hw_cpu = grid_hw.to(device="cpu", dtype=torch.int64)
    max_grid_size = int(grid_hw_cpu.max().item())

    seq = torch.arange(max_grid_size, device=device, dtype=torch.float32)
    rotary_pos_emb_full = torch.outer(seq, inv_freq.to(device=device))

    pos_ids = _build_position_ids(grid_hw, adapooling_factor, device=device)
    freqs = rotary_pos_emb_full[pos_ids].flatten(-2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _build_position_ids(
    grid_hw: torch.Tensor,
    adapooling_factor: int,
    device: torch.device,
) -> torch.Tensor:
    """Build adapooling-regrouped 2D position IDs.

    Mirrors MojoVisionRotaryEmbedding2D._build_position_ids.
    """
    pos_ids = []
    grid_hw_cpu = grid_hw.to(device="cpu", dtype=torch.int64)
    for gh, gw in grid_hw_cpu.tolist():
        hpos_ids = torch.arange(gh, device=device).unsqueeze(1).expand(-1, gw)
        hpos_ids = hpos_ids.reshape(
            gh // adapooling_factor,
            adapooling_factor,
            gw // adapooling_factor,
            adapooling_factor,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

        wpos_ids = torch.arange(gw, device=device).unsqueeze(0).expand(gh, -1)
        wpos_ids = wpos_ids.reshape(
            gh // adapooling_factor,
            adapooling_factor,
            gw // adapooling_factor,
            adapooling_factor,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
        sample_pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
        pos_ids.append(sample_pos_ids)

    return torch.cat(pos_ids, dim=0)


@triton.jit
def _compute_vision_rope(
    x,
    sin_tile,
    cos_tile,
    head_num: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    """Vision 2D RoPE rotation on a 3D tile [TOKEN_BLOCK_SIZE, head_num, 2*half_rope_dim].

    Uses split cos/sin: the first half of cos/sin rotates x1/x2, the second
    half rotates x2/x1.
    """
    # Split cos/sin into first and second halves: [TOKEN_BLOCK_SIZE, 1, half_rope_dim]
    c1 = tl.extract_slice(cos_tile, [0, 0, 0], [TOKEN_BLOCK_SIZE, 1, half_rope_dim], [1, 1, 1])
    c2 = tl.extract_slice(cos_tile, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, 1, half_rope_dim], [1, 1, 1])
    s1 = tl.extract_slice(sin_tile, [0, 0, 0], [TOKEN_BLOCK_SIZE, 1, half_rope_dim], [1, 1, 1])
    s2 = tl.extract_slice(sin_tile, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, 1, half_rope_dim], [1, 1, 1])

    # Split x into halves: [TOKEN_BLOCK_SIZE, head_num, half_rope_dim]
    x1 = tl.extract_slice(x, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x2 = tl.extract_slice(x, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])

    # out_half1 = x1*c1 - x2*s1  (broadcasts c1/s1 across heads)
    # out_half2 = x2*c2 + x1*s2
    roped_x1 = x1 * c1 - x2 * s1
    roped_x2 = x2 * c2 + x1 * s2

    x = tl.insert_slice(x, roped_x1, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x = tl.insert_slice(x, roped_x2, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])

    return x


@triton.jit
def _compute_vision_rope_separated(
    x1,
    x2,
    sin_half1,
    sin_half2,
    cos_half1,
    cos_half2,
):
    """Vision 2D RoPE on pre-split halves."""
    roped_x1 = x1 * cos_half1 - x2 * sin_half1
    roped_x2 = x2 * cos_half2 + x1 * sin_half2
    return roped_x1, roped_x2


@triton.jit
def _vision_rope_apply_kernel(
    q_ptr,
    q_token_stride,
    q_head_stride,
    k_ptr,
    k_token_stride,
    k_head_stride,
    cos_ptr,
    cos_token_stride,
    sin_ptr,
    sin_token_stride,
    T,
    num_token_blocks,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    ALIGNED: tl.constexpr,
    CAST_TO_FP32: tl.constexpr,
):
    """Apply vision 2D RoPE to q and k in a single kernel.

    Layout: q/k [T, N, D] token-first packed, cos/sin [T, D].
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    for block_id in range(pid, num_token_blocks, grid_size):
        token_start = block_id * TOKEN_BLOCK_SIZE
        token_offsets = token_start + tl.arange(0, TOKEN_BLOCK_SIZE)
        token_mask = token_offsets < T

        half_dim_offsets = tl.arange(0, HALF_D)
        half_dim_mask = half_dim_offsets < HALF_D

        cos_token_ptr = cos_ptr + token_offsets[:, None] * cos_token_stride
        sin_token_ptr = sin_ptr + token_offsets[:, None] * sin_token_stride

        head_q_offsets = tl.arange(0, n_qh)
        head_k_offsets = tl.arange(0, n_kh)

        if ALIGNED:
            dim_offsets = tl.arange(0, D)
            dim_mask = dim_offsets < D

            # Load cos/sin as 2D, then reshape to 3D with singleton head dim for broadcasting
            cos_block_2d = tl.load(
                cos_token_ptr + dim_offsets[None, :],
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            sin_block_2d = tl.load(
                sin_token_ptr + dim_offsets[None, :],
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            cos_tile = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, D), can_reorder=True)
            sin_tile = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, D), can_reorder=True)

            # Q: 3D load [TOKEN_BLOCK_SIZE, n_qh, D], rotate, store
            q_offsets = (
                token_offsets[:, None, None] * q_token_stride
                + head_q_offsets[None, :, None] * q_head_stride
                + dim_offsets[None, None, :]
            )
            q_mask = (
                token_mask[:, None, None]
                & (head_q_offsets[None, :, None] < n_qh)
                & dim_mask[None, None, :]
            )
            q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
            if CAST_TO_FP32:
                q_tile = q_tile.to(sin_block_2d.dtype)
            q_tile = _compute_vision_rope(q_tile, sin_tile, cos_tile, n_qh, HALF_D, TOKEN_BLOCK_SIZE)
            tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

            # K: 3D load [TOKEN_BLOCK_SIZE, n_kh, D], rotate, store
            k_offsets = (
                token_offsets[:, None, None] * k_token_stride
                + head_k_offsets[None, :, None] * k_head_stride
                + dim_offsets[None, None, :]
            )
            k_mask = (
                token_mask[:, None, None]
                & (head_k_offsets[None, :, None] < n_kh)
                & dim_mask[None, None, :]
            )
            k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
            if CAST_TO_FP32:
                k_tile = k_tile.to(sin_block_2d.dtype)
            k_tile = _compute_vision_rope(k_tile, sin_tile, cos_tile, n_kh, HALF_D, TOKEN_BLOCK_SIZE)
            tl.store(k_ptr + k_offsets, k_tile, mask=k_mask)
        else:
            # Load cos/sin halves as 2D, reshape for broadcasting
            cos_half1 = tl.load(
                cos_token_ptr + half_dim_offsets[None, :],
                mask=token_mask[:, None] & half_dim_mask[None, :],
                other=0.0,
            )
            cos_half2 = tl.load(
                cos_token_ptr + HALF_D + half_dim_offsets[None, :],
                mask=token_mask[:, None] & half_dim_mask[None, :],
                other=0.0,
            )
            sin_half1 = tl.load(
                sin_token_ptr + half_dim_offsets[None, :],
                mask=token_mask[:, None] & half_dim_mask[None, :],
                other=0.0,
            )
            sin_half2 = tl.load(
                sin_token_ptr + HALF_D + half_dim_offsets[None, :],
                mask=token_mask[:, None] & half_dim_mask[None, :],
                other=0.0,
            )

            c1 = tl.reshape(cos_half1, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)
            c2 = tl.reshape(cos_half2, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)
            s1 = tl.reshape(sin_half1, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)
            s2 = tl.reshape(sin_half2, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)

            # Q halves
            q_offsets_half1 = (
                token_offsets[:, None, None] * q_token_stride
                + head_q_offsets[None, :, None] * q_head_stride
                + half_dim_offsets[None, None, :]
            )
            q_offsets_half2 = (
                token_offsets[:, None, None] * q_token_stride
                + head_q_offsets[None, :, None] * q_head_stride
                + HALF_D
                + half_dim_offsets[None, None, :]
            )
            q_half_mask = (
                token_mask[:, None, None]
                & (head_q_offsets[None, :, None] < n_qh)
                & half_dim_mask[None, None, :]
            )

            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0)
            if CAST_TO_FP32:
                q_tile_1 = q_tile_1.to(tl.float32)
                q_tile_2 = q_tile_2.to(tl.float32)
            new_q_1, new_q_2 = _compute_vision_rope_separated(
                q_tile_1, q_tile_2, s1, s2, c1, c2,
            )
            tl.store(q_ptr + q_offsets_half1, new_q_1, mask=q_half_mask)
            tl.store(q_ptr + q_offsets_half2, new_q_2, mask=q_half_mask)

            # K halves
            k_offsets_half1 = (
                token_offsets[:, None, None] * k_token_stride
                + head_k_offsets[None, :, None] * k_head_stride
                + half_dim_offsets[None, None, :]
            )
            k_offsets_half2 = (
                token_offsets[:, None, None] * k_token_stride
                + head_k_offsets[None, :, None] * k_head_stride
                + HALF_D
                + half_dim_offsets[None, None, :]
            )
            k_half_mask = (
                token_mask[:, None, None]
                & (head_k_offsets[None, :, None] < n_kh)
                & half_dim_mask[None, None, :]
            )

            k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_half_mask, other=0.0)
            k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_half_mask, other=0.0)
            if CAST_TO_FP32:
                k_tile_1 = k_tile_1.to(tl.float32)
                k_tile_2 = k_tile_2.to(tl.float32)
            new_k_1, new_k_2 = _compute_vision_rope_separated(
                k_tile_1, k_tile_2, s1, s2, c1, c2,
            )
            tl.store(k_ptr + k_offsets_half1, new_k_1, mask=k_half_mask)
            tl.store(k_ptr + k_offsets_half2, new_k_2, mask=k_half_mask)


def _get_token_block_size(n_qh: int, n_kh: int) -> int:
    if n_qh <= 8 and n_kh <= 8:
        return 16
    if n_qh <= 32 and n_kh <= 32:
        return 8
    return 4


def vision_rope_apply_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply vision 2D RoPE to q/k tensors in-place via Triton kernel.

    Processes q and k together in a single kernel launch, loading all heads
    as 3D tiles to avoid per-head loop overhead.

    Args:
        q: [T, N_q, D] packed token-first.
        k: [T, N_k, D] packed token-first.
        cos: [T, D] in float32.
        sin: [T, D] in float32.

    Returns:
        (q_rot, k_rot) with same shapes and dtypes as inputs.
    """
    assert q.ndim == 3 and k.ndim == 3
    T, n_qh, D = q.shape
    n_kh = k.shape[1]
    assert cos.ndim == 2 and sin.ndim == 2
    assert cos.shape == (T, D)
    assert sin.shape == (T, D)

    q = q.clone()
    k = k.clone()

    half_D = D // 2
    is_aligned = _is_half_rope_dim_aligned(half_D)
    cast_to_fp32 = q.dtype != torch.float32

    token_block_size = _get_token_block_size(n_qh, n_kh)
    num_token_blocks = (T + token_block_size - 1) // token_block_size

    num_programs = get_num_cores()

    cos = cos.contiguous()
    sin = sin.contiguous()

    grid = (num_programs,)
    _vision_rope_apply_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        k,
        k.stride(0),
        k.stride(1),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        T,
        num_token_blocks,
        n_qh,
        n_kh,
        D,
        half_D,
        token_block_size,
        is_aligned,
        cast_to_fp32,
    )

    return q, k
