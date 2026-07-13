from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores

from .utils import SRAM_ALIGN_BYTES


def _is_half_rope_dim_aligned(half_rope_dim: int, dtype_size: int = 2) -> bool:
    return (half_rope_dim * dtype_size) % SRAM_ALIGN_BYTES == 0


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
    sin_half_tile,
    cos_half_tile,
    head_num: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
):
    """Vision 2D RoPE rotation on a 3D tile [TOKEN_BLOCK_SIZE, head_num, 2*half_rope_dim].

    The current vision RoPE table is built as ``cat([freqs, freqs], dim=-1)``,
    so both cos/sin halves are numerically identical and we only need one half.
    """
    # Split x into halves: [TOKEN_BLOCK_SIZE, head_num, half_rope_dim]
    x1 = tl.extra.cann.extension.extract_slice(x, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x2 = tl.extra.cann.extension.extract_slice(x, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])

    # out_half1 = x1*c - x2*s  (broadcasts c/s across heads)
    # out_half2 = x2*c + x1*s
    roped_x1 = x1 * cos_half_tile - x2 * sin_half_tile
    roped_x2 = x2 * cos_half_tile + x1 * sin_half_tile

    x = tl.extra.cann.extension.insert_slice(x, roped_x1, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x = tl.extra.cann.extension.insert_slice(x, roped_x2, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])

    return x


@triton.jit
def _compute_vision_rope_separated(
    x1,
    x2,
    sin_half,
    cos_half,
):
    """Vision 2D RoPE on pre-split halves."""
    roped_x1 = x1 * cos_half - x2 * sin_half
    roped_x2 = x2 * cos_half + x1 * sin_half
    return roped_x1, roped_x2


@triton.jit
def _vision_rope_apply_kernel(
    q_ptr,
    q_token_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    k_ptr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    cos_ptr,
    cos_token_stride: tl.constexpr,
    sin_ptr,
    sin_token_stride: tl.constexpr,
    T,
    num_token_blocks,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    ALIGNED: tl.constexpr,
    CAST_TO_FP32: tl.constexpr,
    TOKEN_ALIGNED: tl.constexpr,
):
    """Apply vision 2D RoPE to q and k in a single kernel.

    Layout: q/k [T, N, D] token-first packed, cos/sin [T, D].
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    half_dim_offsets = tl.arange(0, HALF_D)
    head_q_offsets = tl.arange(0, n_qh)
    head_k_offsets = tl.arange(0, n_kh)

    for block_id in range(pid, num_token_blocks, grid_size):
        token_start = block_id * TOKEN_BLOCK_SIZE
        token_offsets = token_start + tl.arange(0, TOKEN_BLOCK_SIZE)

        cos_token_ptr = cos_ptr + token_offsets[:, None] * cos_token_stride
        sin_token_ptr = sin_ptr + token_offsets[:, None] * sin_token_stride

        if ALIGNED:
            if TOKEN_ALIGNED:
                # cos/sin: 2D block_ptr [TOKEN_BLOCK_SIZE, HALF_D]
                cos_block_ptr = tl.make_block_ptr(
                    base=cos_ptr,
                    shape=(T, HALF_D),
                    strides=(cos_token_stride, 1),
                    offsets=(token_start, 0),
                    block_shape=(TOKEN_BLOCK_SIZE, HALF_D),
                    order=(1, 0),
                )
                sin_block_ptr = tl.make_block_ptr(
                    base=sin_ptr,
                    shape=(T, HALF_D),
                    strides=(sin_token_stride, 1),
                    offsets=(token_start, 0),
                    block_shape=(TOKEN_BLOCK_SIZE, HALF_D),
                    order=(1, 0),
                )
                cos_half = tl.load(cos_block_ptr)
                sin_half = tl.load(sin_block_ptr)

                cos_half_tile = tl.reshape(cos_half, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)
                sin_half_tile = tl.reshape(sin_half, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)

                # Q: 3D block_ptr [TOKEN_BLOCK_SIZE, n_qh, D]
                q_block_ptr = tl.make_block_ptr(
                    base=q_ptr,
                    shape=(T, n_qh, D),
                    strides=(q_token_stride, q_head_stride, 1),
                    offsets=(token_start, 0, 0),
                    block_shape=(TOKEN_BLOCK_SIZE, n_qh, D),
                    order=(2, 1, 0),
                )
                q_tile = tl.load(q_block_ptr)
                q_load_dtype = q_tile.dtype
                if CAST_TO_FP32:
                    q_tile = q_tile.to(cos_half.dtype)
                q_tile = _compute_vision_rope(q_tile, sin_half_tile, cos_half_tile, n_qh, HALF_D, TOKEN_BLOCK_SIZE)
                tl.store(q_block_ptr, q_tile.to(q_load_dtype))

                # K: 3D block_ptr [TOKEN_BLOCK_SIZE, n_kh, D]
                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr,
                    shape=(T, n_kh, D),
                    strides=(k_token_stride, k_head_stride, 1),
                    offsets=(token_start, 0, 0),
                    block_shape=(TOKEN_BLOCK_SIZE, n_kh, D),
                    order=(2, 1, 0),
                )
                k_tile = tl.load(k_block_ptr)
                k_load_dtype = k_tile.dtype
                if CAST_TO_FP32:
                    k_tile = k_tile.to(cos_half.dtype)
                k_tile = _compute_vision_rope(k_tile, sin_half_tile, cos_half_tile, n_kh, HALF_D, TOKEN_BLOCK_SIZE)
                tl.store(k_block_ptr, k_tile.to(k_load_dtype))
            else:
                token_mask = token_offsets < T

                cos_half = tl.load(
                    cos_token_ptr + half_dim_offsets[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                )
                sin_half = tl.load(
                    sin_token_ptr + half_dim_offsets[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                )

                cos_half_tile = tl.reshape(cos_half, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)
                sin_half_tile = tl.reshape(sin_half, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)

                dim_offsets = tl.arange(0, D)

                q_offsets = (
                    token_offsets[:, None, None] * q_token_stride
                    + head_q_offsets[None, :, None] * q_head_stride
                    + dim_offsets[None, None, :]
                )
                q_mask = token_mask[:, None, None]
                q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
                if CAST_TO_FP32:
                    q_tile = q_tile.to(cos_half.dtype)
                q_tile = _compute_vision_rope(q_tile, sin_half_tile, cos_half_tile, n_qh, HALF_D, TOKEN_BLOCK_SIZE)
                tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

                k_offsets = (
                    token_offsets[:, None, None] * k_token_stride
                    + head_k_offsets[None, :, None] * k_head_stride
                    + dim_offsets[None, None, :]
                )
                k_mask = token_mask[:, None, None]
                k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
                if CAST_TO_FP32:
                    k_tile = k_tile.to(cos_half.dtype)
                k_tile = _compute_vision_rope(k_tile, sin_half_tile, cos_half_tile, n_kh, HALF_D, TOKEN_BLOCK_SIZE)
                tl.store(k_ptr + k_offsets, k_tile, mask=k_mask)
        else:
            if TOKEN_ALIGNED:
                cos_half = tl.load(cos_token_ptr + half_dim_offsets[None, :])
                sin_half = tl.load(sin_token_ptr + half_dim_offsets[None, :])

                cos_half_tile = tl.reshape(cos_half, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)
                sin_half_tile = tl.reshape(sin_half, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)

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

                q_tile_1 = tl.load(q_ptr + q_offsets_half1)
                q_tile_2 = tl.load(q_ptr + q_offsets_half2)
                if CAST_TO_FP32:
                    q_tile_1 = q_tile_1.to(tl.float32)
                    q_tile_2 = q_tile_2.to(tl.float32)
                roped_x1 = q_tile_1 * cos_half_tile - q_tile_2 * sin_half_tile
                roped_x2 = q_tile_2 * cos_half_tile + q_tile_1 * sin_half_tile
                tl.store(q_ptr + q_offsets_half1, roped_x1)
                tl.store(q_ptr + q_offsets_half2, roped_x2)

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

                k_tile_1 = tl.load(k_ptr + k_offsets_half1)
                k_tile_2 = tl.load(k_ptr + k_offsets_half2)
                if CAST_TO_FP32:
                    k_tile_1 = k_tile_1.to(tl.float32)
                    k_tile_2 = k_tile_2.to(tl.float32)
                roped_k1 = k_tile_1 * cos_half_tile - k_tile_2 * sin_half_tile
                roped_k2 = k_tile_2 * cos_half_tile + k_tile_1 * sin_half_tile
                tl.store(k_ptr + k_offsets_half1, roped_k1)
                tl.store(k_ptr + k_offsets_half2, roped_k2)
            else:
                token_mask = token_offsets < T

                cos_half = tl.load(
                    cos_token_ptr + half_dim_offsets[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                )
                sin_half = tl.load(
                    sin_token_ptr + half_dim_offsets[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                )

                cos_half_tile = tl.reshape(cos_half, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)
                sin_half_tile = tl.reshape(sin_half, (TOKEN_BLOCK_SIZE, 1, HALF_D), can_reorder=True)

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
                q_half_mask = token_mask[:, None, None]

                q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0)
                q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0)
                if CAST_TO_FP32:
                    q_tile_1 = q_tile_1.to(tl.float32)
                    q_tile_2 = q_tile_2.to(tl.float32)
                roped_x1 = q_tile_1 * cos_half_tile - q_tile_2 * sin_half_tile
                roped_x2 = q_tile_2 * cos_half_tile + q_tile_1 * sin_half_tile
                tl.store(q_ptr + q_offsets_half1, roped_x1, mask=q_half_mask)
                tl.store(q_ptr + q_offsets_half2, roped_x2, mask=q_half_mask)

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
                k_half_mask = token_mask[:, None, None]

                k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_half_mask, other=0.0)
                k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_half_mask, other=0.0)
                if CAST_TO_FP32:
                    k_tile_1 = k_tile_1.to(tl.float32)
                    k_tile_2 = k_tile_2.to(tl.float32)
                roped_k1 = k_tile_1 * cos_half_tile - k_tile_2 * sin_half_tile
                roped_k2 = k_tile_2 * cos_half_tile + k_tile_1 * sin_half_tile
                tl.store(k_ptr + k_offsets_half1, roped_k1, mask=k_half_mask)
                tl.store(k_ptr + k_offsets_half2, roped_k2, mask=k_half_mask)



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

    half_D = D // 2
    is_aligned = _is_half_rope_dim_aligned(half_D)
    cast_to_fp32 = q.dtype != torch.float32

    token_block_size = 2
    num_token_blocks = (T + token_block_size - 1) // token_block_size
    token_aligned = (T % token_block_size == 0)

    num_programs = get_num_cores()
    num_programs = min(num_programs, num_token_blocks)

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
        token_aligned,
        enable_vf_fusion=True,
    )

    return q, k
