# SPDX-License-Identifier: Apache-2.0
"""Indexed RMSNorm and affine transform using one Ascend Triton kernel."""

import torch
import triton
import triton.language as tl

from triton.language.extra.cann.extension import extract_slice

from mojo_opset.kernels._npu_triton_utils import get_num_cores


@triton.jit
def _rms_norm_indexed_scale_shift_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NT: tl.constexpr,
    NUM_CORES: tl.constexpr,
    SX: tl.constexpr,
    SS: tl.constexpr,
    SC: tl.constexpr,
    SI: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, BLOCK_C).to(tl.int32)
    r2 = tl.arange(0, 2).to(tl.int32)
    # BLOCK_C <= H by construction, so tile 0 covers valid columns only.
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    npairs = (rows + 1) // 2
    for p in range(pid, npairs, NUM_CORES):
        b0 = 2 * p
        rows2 = (b0 + r2)[:, None]
        # Two-row window with a general row stride: no layout dispatch needed.
        x2 = tl.load(x_ptr + rows2 * SX + cols[None, :], mask=rows2 < rows)
        i0 = tl.load(indices_ptr + b0 * SI).to(tl.int32)
        i1 = tl.load(indices_ptr + (b0 + 1) * SI, mask=(b0 + 1) < rows, other=0).to(tl.int32)
        # The table rows do not depend on the RMS result, so they are issued
        # before the reduction: their DMA runs under the vector reduction
        # instead of after it.
        s0 = tl.load(shift_ptr + i0 * SS + cols).to(tl.float32)
        c0 = tl.load(scale_ptr + i0 * SC + cols).to(tl.float32)
        x0 = tl.reshape(extract_slice(x2, (0, 0), (1, BLOCK_C), (1, 1)), (BLOCK_C,)).to(tl.float32)
        sumsq0 = tl.sum(x0 * x0, axis=0)
        for j in tl.static_range(NT - 1):
            c = (j + 1) * BLOCK_C + cols
            m = c < H
            xa = tl.load(x_ptr + b0 * SX + c, mask=m, other=0.0).to(tl.float32)
            sumsq0 += tl.sum(xa * xa, axis=0)
        r0 = tl.rsqrt(sumsq0 / H + eps)
        tl.store(out_ptr + b0 * H + cols, x0 * (r0 * weight) * (1.0 + c0) + s0)
        s1 = tl.load(shift_ptr + i1 * SS + cols).to(tl.float32)
        c1 = tl.load(scale_ptr + i1 * SC + cols).to(tl.float32)
        x1 = tl.reshape(extract_slice(x2, (1, 0), (1, BLOCK_C), (1, 1)), (BLOCK_C,)).to(tl.float32)
        sumsq1 = tl.sum(x1 * x1, axis=0)
        for j in tl.static_range(NT - 1):
            c = (j + 1) * BLOCK_C + cols
            m = c < H
            xb = tl.load(
                x_ptr + (b0 + 1) * SX + c,
                mask=m & ((b0 + 1) < rows),
                other=0.0,
            ).to(tl.float32)
            sumsq1 += tl.sum(xb * xb, axis=0)
        r1 = tl.rsqrt(sumsq1 / H + eps)
        tl.store(
            out_ptr + (b0 + 1) * H + cols,
            x1 * (r1 * weight) * (1.0 + c1) + s1,
            mask=(b0 + 1) < rows,
        )
        for j in tl.static_range(NT - 1):
            c = (j + 1) * BLOCK_C + cols
            m = c < H
            wj = tl.load(weight_ptr + c, mask=m, other=0.0).to(tl.float32)
            xa = tl.load(x_ptr + b0 * SX + c, mask=m, other=0.0).to(tl.float32)
            sh = tl.load(shift_ptr + i0 * SS + c, mask=m, other=0.0).to(tl.float32)
            sc = tl.load(scale_ptr + i0 * SC + c, mask=m, other=0.0).to(tl.float32)
            tl.store(out_ptr + b0 * H + c, xa * (r0 * wj) * (1.0 + sc) + sh, mask=m)
            xb = tl.load(
                x_ptr + (b0 + 1) * SX + c,
                mask=m & ((b0 + 1) < rows),
                other=0.0,
            ).to(tl.float32)
            sh1 = tl.load(shift_ptr + i1 * SS + c, mask=m, other=0.0).to(tl.float32)
            sc1 = tl.load(scale_ptr + i1 * SC + c, mask=m, other=0.0).to(tl.float32)
            tl.store(
                out_ptr + (b0 + 1) * H + c,
                xb * (r1 * wj) * (1.0 + sc1) + sh1,
                mask=m & ((b0 + 1) < rows),
            )


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm followed by shift/scale selected by each row's index.

    Activations and condition tables may have padded rows. The feature
    dimension and weight require unit stride. Indices must be in range.
    """
    rows, hidden = x.shape
    output = torch.empty_like(x, memory_format=torch.contiguous_format)

    block_c = hidden if hidden <= 6144 else 2048
    num_tiles = triton.cdiv(hidden, block_c)
    cores = min(get_num_cores("vector"), triton.cdiv(rows, 2))
    _rms_norm_indexed_scale_shift_kernel[(cores,)](
        output,
        x,
        weight,
        shift,
        scale,
        indices,
        rows,
        H=hidden,
        eps=eps,
        BLOCK_C=block_c,
        NT=num_tiles,
        NUM_CORES=cores,
        SX=x.stride(0),
        SS=shift.stride(0),
        SC=scale.stride(0),
        SI=indices.stride(0),
    )
    return output


def _validate_layout(x, weight, shift, scale):
    if x.stride(1) != 1 or weight.stride(0) != 1 or shift.stride(1) != 1 or scale.stride(1) != 1:
        raise ValueError("A5 requires unit stride in the feature dimension of x, weight, shift, and scale")


@torch.library.custom_op("mojo_npu_triton_a5::rms_norm_indexed_scale_shift_fwd", mutates_args=())
def rms_norm_indexed_scale_shift_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    _validate_layout(x, weight, shift, scale)
    if x.shape[0] == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.npu.device(x.device):
        return _launch(x, weight, shift, scale, indices, eps)


@rms_norm_indexed_scale_shift_fwd.register_fake
def _rms_norm_indexed_scale_shift_fake(x, weight, shift, scale, indices, eps):
    _validate_layout(x, weight, shift, scale)
    return torch.empty_like(x, memory_format=torch.contiguous_format)
