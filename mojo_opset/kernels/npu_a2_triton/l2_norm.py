"""Row-wise L2 normalization Triton kernels for Ascend A2 (910B)."""
# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry
from mojo_opset.kernels._triton_utils import input_guard

_UB_BYTES = 196608  # Ascend A2 (910B) UB size, 192 KiB
_FP32_COPIES = 3  # load, squares and result live at once


def _whole_row_fits(D: int) -> bool:
    """True when a single-row tile still fits UB (BT=1, the smallest tile)."""
    BD = triton.next_power_of_2(int(D))
    return BD * 4 * _FP32_COPIES <= _UB_BYTES


def _max_bt(D: int) -> int:
    """Largest BT whose tile fits UB for this feature width.

    Callers reach this only after ``_whole_row_fits`` passes, so the result is
    always at least 1.
    """
    BD = triton.next_power_of_2(int(D))
    return _UB_BYTES // (BD * 4 * _FP32_COPIES)


# Chunked path only gets about 40% of UB, so a tile leaves room for the compiler
# to overlap DMA with compute. On A2 that budget stops the chunk at 4096: a
# 4096-wide fp32 tile is 49152 bytes (25% of 192 KiB), while 8192 would be
# 98304 bytes and exceed the 40% budget.
_CHUNK_UB_FRACTION = 0.4


def _pick_chunk(D: int) -> int:
    """Widest power-of-two chunk that fits the chunked path's UB budget.

    Returns a power of two so that ``next_power_of_2(D) % chunk == 0`` and the
    kernel's static_range loop covers the row exactly. Only valid once the row
    itself does not fit, where the result divides ``next_power_of_2(D)``.
    """
    BD = triton.next_power_of_2(int(D))
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


@libentry()
@triton.jit(do_not_specialize=["D"])
def _l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
    CH: tl.constexpr,
):
    # The row does not fit in UB, so walk it in CH-wide fp32 chunks.
    # Squares accumulate in a CH-wide vector; the reduction runs once after
    # the loop so chunks do not depend on a scalar sum. Pass 2 reloads x.
    i_t = tl.program_id(0).to(tl.int64)
    x += i_t * D
    y += i_t * D
    c = tl.arange(0, CH)
    acc = tl.zeros((CH,), tl.float32)
    for start in tl.static_range(0, BD, CH):
        mask = (start + c) < D
        blk = tl.load(x + start + c, mask=mask, other=0.0).to(tl.float32)
        acc += blk * blk
    rstd = tl.rsqrt(tl.sum(acc) + eps)
    for start in tl.static_range(0, BD, CH):
        mask = (start + c) < D
        blk = tl.load(x + start + c, mask=mask, other=0.0).to(tl.float32)
        tl.store(y + start + c, (blk * rstd).to(y.dtype.element_ty), mask=mask)


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


@input_guard(make_contiguous=True, auto_to_device=True)
def l2norm_fwd_impl(x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
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
        CH = _pick_chunk(D)
        _l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            CH=CH,
        )

    return y.view(x_shape_og)


def _output_dtype(x: torch.Tensor, output_dtype: Optional[torch.dtype]) -> torch.dtype:
    return x.dtype if output_dtype is None else output_dtype


@torch.library.custom_op("mojo_npu_triton_a2::l2_norm_fwd", mutates_args=())
def l2_norm_fwd(x: torch.Tensor, eps: float, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if x.numel() == 0:
        return torch.empty_like(x, dtype=_output_dtype(x, output_dtype))
    return l2norm_fwd_impl(x.contiguous(), eps, output_dtype)


@l2_norm_fwd.register_fake
def _l2_norm_fwd_fake(
    x: torch.Tensor,
    eps: float,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    del eps
    return torch.empty_like(
        x,
        dtype=_output_dtype(x, output_dtype),
        memory_format=torch.contiguous_format,
    )
