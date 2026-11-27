# SPDX-License-Identifier: Apache-2.0
"""A2 indexed RMSNorm using one tile per row."""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores

# A2 vector tile alignment and the single-tile implementation's column budget.
_COLUMN_ALIGNMENT = 128
_MAX_TILE_COLUMNS = 8192


@triton.jit
def _rms_norm_indexed_scale_shift_kernel(
    Y,
    X,
    W,
    Shift,
    Scale,
    Index,
    R,
    H: tl.constexpr,
    N: tl.constexpr,
    CORES: tl.constexpr,
    EPS: tl.constexpr,
    SX: tl.constexpr,
    XC: tl.constexpr,
    SW: tl.constexpr,
    SS: tl.constexpr,
    CS: tl.constexpr,
    SC: tl.constexpr,
    CC: tl.constexpr,
    SI: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    col = tl.arange(0, N).to(tl.int32)
    valid = col < H
    weight = tl.load(W + col * SW, valid, 0).to(tl.float32)
    cached_idx = tl.full((), -1, tl.int64)
    cached_shift = tl.zeros((N,), tl.float32)
    cached_scale_factor = tl.zeros((N,), tl.float32)

    for row in range(pid, R, CORES):
        x = tl.load(X + row * SX + col * XC, valid, 0).to(tl.float32)
        idx = tl.load(Index + row * SI)
        if idx != cached_idx:
            cached_shift = tl.load(Shift + idx * SS + col * CS, valid, 0).to(tl.float32)
            scale = tl.load(Scale + idx * SC + col * CC, valid, 0).to(tl.float32)
            cached_scale_factor = 1.0 + scale
            cached_idx = idx
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, 0) / H + EPS)
        # Normalize x before multiplying by weight to avoid an overflowing intermediate.
        normalized = (x * rstd) * weight
        out = normalized * cached_scale_factor + cached_shift
        tl.store(Y + row * H + col, out, valid)


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse RMSNorm and indexed affine modulation for a single-tile NPU row."""
    rows, hidden = x.shape
    tile_columns = triton.cdiv(hidden, _COLUMN_ALIGNMENT) * _COLUMN_ALIGNMENT

    out = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)

    cores = min(rows, get_num_cores("vector"))
    _rms_norm_indexed_scale_shift_kernel[(cores,)](
        out,
        x,
        weight,
        shift,
        scale,
        indices,
        rows,
        H=hidden,
        N=tile_columns,
        CORES=cores,
        EPS=eps,
        SX=x.stride(0),
        XC=x.stride(1),
        SW=weight.stride(0),
        SS=shift.stride(0),
        CS=shift.stride(1),
        SC=scale.stride(0),
        CC=scale.stride(1),
        SI=indices.stride(0),
    )
    return out


def _validate_layout(x):
    if x.shape[1] > _MAX_TILE_COLUMNS:
        raise ValueError(f"A2 Triton supports hidden sizes up to {_MAX_TILE_COLUMNS}")


@torch.library.custom_op("mojo_npu_triton_a2::rms_norm_indexed_scale_shift_fwd", mutates_args=())
def rms_norm_indexed_scale_shift_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    _validate_layout(x)
    if x.shape[0] == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.npu.device(x.device):
        return _launch(x, weight, shift, scale, indices, eps)


@rms_norm_indexed_scale_shift_fwd.register_fake
def _rms_norm_indexed_scale_shift_fake(x, weight, shift, scale, indices, eps):
    _validate_layout(x)
    return torch.empty_like(x, memory_format=torch.contiguous_format)
