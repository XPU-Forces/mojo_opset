"""A5 Triton L2 normalization with FP32 statistics; forward only."""
# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry
from mojo_opset.kernels._triton_utils import input_guard

# A5 unified buffer budget shared by the 950PR and 950DT SKUs. Device properties
# expose no buffer size, and this provider serves both SKUs, so keep it as a
# conservative bound for both rather than a 950PR-only measurement.
_UB_BYTES = 253952
_FP32_COPIES = 3            # load, squares and result live at once


def _whole_row_fits(D: int) -> bool:
    """True when a single-row tile still fits UB (BT=1, the smallest tile)."""
    BD = triton.next_power_of_2(D)
    return BD * 4 * _FP32_COPIES <= _UB_BYTES


def _max_bt(D: int) -> int:
    """Largest BT whose tile fits UB for this feature width."""
    BD = triton.next_power_of_2(D)
    return _UB_BYTES // (BD * 4 * _FP32_COPIES)


# Chunked path only gets about 40% of UB. Measured on 512x32768 and 2048x32768
# f16: CH=8192 (39% of UB) beat CH=16384 (77%) by 6-12%, because a tile that
# fills UB leaves nothing for the compiler to overlap DMA with compute. CH=16384
# also fails to compile outright at D=65536.
_CHUNK_UB_FRACTION = 0.4


def _pick_chunk(D: int) -> int:
    """Widest power-of-two chunk that fits the chunked path's UB budget.

    Returns a power of two so that ``next_power_of_2(D) % chunk == 0`` and the
    kernel's static_range loop covers the row exactly.
    """
    BD = triton.next_power_of_2(D)
    budget = int(_UB_BYTES * _CHUNK_UB_FRACTION)
    chunk = 2048
    while chunk * 2 <= BD and chunk * 2 * 4 * _FP32_COPIES <= budget:
        chunk *= 2
    return chunk


def _prune_by_ub(configs, named_args, **kwargs):
    """Keep only BT values that fit UB.

    Without this, autotune compiles every candidate before timing it, so a BT
    that overflows UB (or trips Triton's numel limit once BT*BD exceeds it)
    costs hundreds of milliseconds and is then discarded. At D=16384 only BT=1
    fits, so 7 of the 8 entries were pure waste.

    D is a tl.constexpr and never appears in named_args, so take it from x.
    """
    x = named_args.get("x")
    if x is None:
        return configs
    limit = _max_bt(x.shape[-1])
    kept = [c for c in configs if c.kwargs["BT"] <= limit]
    return kept or [min(configs, key=lambda c: c.kwargs["BT"])]


@triton.autotune(
    configs=[triton.Config({"BT": BT}) for BT in [1, 2, 4, 8, 16, 32, 64, 128]],
    key=["D"],
    prune_configs_by={"early_config_prune": _prune_by_ub},
)
@libentry()
@triton.jit(do_not_specialize=["T"])
def _l2norm_fwd_kernel(
    x,
    y,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_x = (o_t[:, None] < T) & (o_d[None, :] < D)
    p_x = x + o_t[:, None] * D + o_d[None, :]
    p_y = y + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_rstd = tl.rsqrt(b_var + eps)[:, None]
    b_y = b_x * b_rstd
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)


@libentry()
@triton.jit(do_not_specialize=["D"])
def _l2norm_fwd_kernel1(
    x,
    y,
    eps,
    D,
    BD: tl.constexpr,
    CH: tl.constexpr,
    N: tl.constexpr,
):
    """One row per program, with the row walked in CH-wide chunks.

    Used once D grows past what a whole-row tile can hold in UB. The row is
    read twice (sum of squares, then normalise) so traffic is 3x instead of 2x,
    which is the price of keeping the live tile at CH elements.

    N = BD // CH is a constexpr because tl.static_range needs one; deriving it
    from BD and CH inside the kernel yields a tensor and is rejected.
    """
    i_t = tl.program_id(0).to(tl.int64)
    base = i_t * D
    c = tl.arange(0, CH)
    acc = tl.zeros((CH,), tl.float32)

    for k in tl.static_range(N):
        m = (k * CH + c) < D
        blk = tl.load(x + base + k * CH + c, mask=m, other=0.0).to(tl.float32)
        acc += blk * blk

    rstd = tl.rsqrt(tl.sum(acc) + eps)

    for k in tl.static_range(N):
        m = (k * CH + c) < D
        blk = tl.load(x + base + k * CH + c, mask=m, other=0.0).to(tl.float32)
        tl.store(y + base + k * CH + c, (blk * rstd).to(y.dtype.element_ty), mask=m)


@input_guard(make_contiguous=True, auto_to_device=True)
def l2norm_fwd_impl(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Launch the fused or chunked kernel; input is made contiguous by the guard."""
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    max_fused_size = 65536 // x.element_size()
    BD = min(max_fused_size, triton.next_power_of_2(D))
    if D > BD:
        raise ValueError(f"This layer doesn't support feature dim >= 64KB, got {D}.")

    # A whole-row tile is BD fp32 values and the kernel keeps about three of
    # them live, so BT=1 (the smallest possible tile) needs BD*4*3 bytes. That
    # puts the real ceiling at BD <= 253952/12 = 21162, i.e. D <= 16384 for a
    # power-of-two width. Past that the row is walked in chunks instead.
    if _whole_row_fits(D):
        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        _l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            T=T,
            D=D,
            BD=BD,
        )
    else:
        # Chunked path: walk the row in pieces sized by _pick_chunk.
        CH = _pick_chunk(D)
        _l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps,
            D,
            BD=BD,
            CH=CH,
            N=BD // CH,
        )
    return y.view(x_shape_og)


@torch.library.custom_op("mojo_npu_triton_a5::l2_norm_fwd", mutates_args=())
def l2_norm_fwd(x: torch.Tensor, eps: float, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if x.numel() == 0:
        return torch.empty_like(x, dtype=x.dtype if output_dtype is None else output_dtype)
    return l2norm_fwd_impl(x, eps, output_dtype)


@l2_norm_fwd.register_fake
def _l2_norm_fwd_fake(
    x: torch.Tensor, eps: float, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    del eps
    dtype = x.dtype if output_dtype is None else output_dtype
    return torch.empty_like(x, dtype=dtype, memory_format=torch.contiguous_format)
