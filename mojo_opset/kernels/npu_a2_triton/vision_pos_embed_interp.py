"""a2 Triton provider for learned position-table interpolation."""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_vision_pos_embed_interp import _batch_body
from mojo_opset.kernels._npu_vision_pos_embed_interp import launch


@triton.jit
def _bilinear_pos_embed_kernel(
    embed_ptr,
    output_ptr,
    GRIDS: tl.constexpr,
    ROW_STARTS: tl.constexpr,
    TILE_STARTS: tl.constexpr,
    SCALES_H: tl.constexpr,
    SCALES_W: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BD: tl.constexpr,
):
    _batch_body(
        embed_ptr,
        output_ptr,
        GRIDS,
        ROW_STARTS,
        TILE_STARTS,
        SCALES_H,
        SCALES_W,
        G,
        M,
        D,
        S0,
        S1,
        BD,
    )


@torch.library.custom_op("mojo_npu_triton_a2::vision_pos_embed_interp_batch_fwd", mutates_args=())
def vision_pos_embed_interp_batch_fwd(
    embed_weight: torch.Tensor,
    grids: list[int],
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(grids) == 3:
        from mojo_opset.kernels._npu_vision_pos_embed_interp_stream import launch_stream, supports_stream

        if supports_stream(embed_weight, *grids, num_grid_per_side, m_size, dtype):
            return launch_stream(embed_weight, grids[1], grids[2], num_grid_per_side)
    if (len(grids) == 3 and 1 <= grids[0] <= 3 and grids[1] * grids[2] >= 512
            and grids[2] == num_grid_per_side and embed_weight.dtype == dtype == torch.bfloat16
            and embed_weight.is_contiguous() and embed_weight.shape[1] % 16 == 0
            and 128 <= embed_weight.shape[1] <= 1152 and 8 % m_size == 0):
        from mojo_opset.kernels._npu_vision_pos_embed_interp_simd import launch_integral

        return launch_integral(embed_weight, *grids, num_grid_per_side, m_size)
    if (len(grids) == 3 and 1 <= grids[0] <= 3 and grids[1] * grids[2] >= 512
            and grids[2] > num_grid_per_side and embed_weight.dtype == dtype == torch.bfloat16
            and embed_weight.is_contiguous() and 128 <= embed_weight.shape[1] <= 1152):
        from mojo_opset.kernels._npu_vision_pos_embed_interp_simd import launch_span

        return launch_span(embed_weight, *grids, num_grid_per_side, m_size)
    return launch(_bilinear_pos_embed_kernel, embed_weight, grids, num_grid_per_side, m_size, dtype)


@vision_pos_embed_interp_batch_fwd.register_fake
def _batch_fake(embed_weight, grids, num_grid_per_side, m_size, dtype):
    total = sum(grids[i] * grids[i + 1] * grids[i + 2] for i in range(0, len(grids), 3))
    return embed_weight.new_empty((total, embed_weight.shape[1]), dtype=dtype)


def vision_pos_embed_interp_fwd(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    return vision_pos_embed_interp_batch_fwd(embed_weight, [t, h, w], num_grid_per_side, m_size, dtype)
