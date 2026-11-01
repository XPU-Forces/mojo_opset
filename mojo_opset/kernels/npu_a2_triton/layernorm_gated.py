"""Gated LayerNorm / RMSNorm with an optional FP8 per-token quant epilogue (Ascend Triton).

Port of SGLang v0.5.22 ``sglang/kernels/ops/attention/fla/layernorm_gated.py``
(``_layer_norm_fwd_1pass_kernel``, itself from fla-org/flash-linear-attention). One kernel
covers the whole capability matrix -- LayerNorm vs RMSNorm, no gate / 2-D gate / 3-D strided
gate, gate-before vs gate-after norm, swish vs sigmoid, affine, group norm, and an FP8 e4m3
per-token quant epilogue -- by constant-folding the ``constexpr`` switches rather than
branching at run time.

Numerical contract (matches the reference implementation)
--------------------------------------------------------
1. Statistics accumulate in FP32 even for FP16/BF16 input; ``Mean`` and ``Rstd`` are written
   as FP32.
2. Group norm reduces inside each ``group_size`` slice, never across the row: with
   ``ngroups > 1`` the reduction is per group and the ``Mean``/``Rstd`` layout is
   ``[ngroups * M]``, indexed ``group * M + row``.
3. The gate is applied in FP32 and only the final store rounds, so the FP16/BF16 rounding
   happens once.
4. The quant path rounds to the output dtype **before** computing the amax, then
   ``scale = max(amax, 1e-10) / 448`` and a clamp to +-448, so ``(q, s)`` is bit-identical to
   the unfused ``per_token_group_quant_fp8``.
5. ``Z_IS_3D`` addresses the gate through ``(row // Z_HEADS) * stride_z_token +
   (row % Z_HEADS) * stride_z_head``; that form is affine in the row vector, so the gate tile
   stays a bulk transfer instead of degenerating into a gather.

Ascend specific design
---------------------
* The tile is ``(ROWS_PER_BLOCK, BLOCK_N)`` with *affine* addresses. This is what the
  bandwidth probe established on 3.6.0: a 1-D per-row kernel caps at MBU 0.05 because every
  256 B row pays a fixed transfer cost, while the same work as a 2-D tile reaches 0.73 at
  ``ROWS_PER_BLOCK=16`` and 0.93 at 32 on a 134 MB case. The reference implementation's
  ``MAX_ROWS_PER_BLOCK = 4`` is tuned for CUDA's SM count and lands at 0.30 here, so the
  default is re-derived for this hardware instead of copied.
* ``ROWS_PER_BLOCK`` is therefore chosen from the UB budget and the row count, not from a
  warp count: the live tiles are ``x``, the gate, the output and the weight, all FP32, so
  ``ROWS_PER_BLOCK * BLOCK_N * 4 B * 4 tiles`` has to stay under the 192 KB UB. The
  batch-invariant mode pins the value to ``MAX_ROWS_PER_BLOCK`` because the row count must
  not change the reduction order.
* ``num_warps`` is kept in the signature for source compatibility; the Ascend backend maps a
  program onto one vector core and ignores it.
"""

import os
from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry

# Pinned by the batch-invariant contract: the value must not vary with M or the core count,
# otherwise the reduction order (and therefore the numerics) would depend on the batch size.
MAX_ROWS_PER_BLOCK = 4

# UB budget for the tile, in bytes. A2 exposes 192 KB per vector core; keeping the live tiles
# under half of it leaves room for the double buffering the backend inserts on its own.
_UB_TILE_BUDGET = 96 * 1024

# Element size of the FP32 tiles the kernel keeps live: x, gate, output, weight/bias.
_LIVE_FP32_TILES = 4

_MAX_FUSED_SIZE_DIVISOR = 65536


@libentry()
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
    Q=None,
    S=None,
    QUANT: tl.constexpr = False,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    x_base = X + row_offsets + col_offsets
    y_base = Y + rows[:, None] * stride_y_row + col_offsets

    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    x = tl.load(x_base, mask=mask, other=0.0).to(tl.float32)

    # Gate-before-norm: the gate multiplies the input, so it has to be read before the
    # statistics. Gate-after-norm defers the same load until after the affine, which keeps the
    # gate tile out of the reduction's live set.
    if HAS_Z and not NORM_BEFORE_GATE:
        if Z_IS_3D:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        z = tl.load(Z + z_row_offsets + col_offsets, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N
        tl.store(Mean + group * M + rows, mean, mask=rows < M)
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        mean = tl.zeros((ROWS_PER_BLOCK,), dtype=tl.float32)

    rstd = tl.rsqrt(var + eps)
    tl.store(Rstd + group * M + rows, rstd, mask=rows < M)

    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    if HAS_BIAS:
        y = x_hat * w[None, :] + b[None, :]
    else:
        y = x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        if Z_IS_3D:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        z = tl.load(Z + z_row_offsets + col_offsets, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    if QUANT:
        # One program is one token, so the amax below spans exactly one token's heads -- the
        # same window per_token_group_quant_fp8 uses.
        y = tl.where(mask, y.to(Y.dtype.element_ty).to(tl.float32), 0.0)
        y_s = tl.maximum(tl.max(tl.abs(y)), 1e-10) / 448.0
        y_q = tl.clamp(y * (1.0 / y_s), -448.0, 448.0).to(Q.dtype.element_ty)
        tl.store(Q + rows[:, None] * N + col_offsets, y_q, mask=mask)
        tl.store(S + tl.program_id(0), y_s)
    else:
        tl.store(y_base, y, mask=mask)


def _pick_rows_per_block(m: int, block_n: int, batch_invariant: bool) -> int:
    """Rows per tile, from the UB budget rather than from a CUDA warp count.

    ``ROWS_PER_BLOCK * BLOCK_N`` is the tile size; the kernel keeps FP32 ``x``, the gate, the
    output and the weight live at the same time, so the budget is divided by four. The result
    is rounded down to a power of two and clamped to ``[1, MAX]`` where ``MAX`` comes from the
    budget, not from the reference's CUDA constant.

    Measured on A2 (134 MB, N=128, BF16, ``p_t3_probe``): MBU 0.30 at 4 rows, 0.53 at 8,
    0.73 at 16, 0.93 at 32. So the ceiling here is 64 rows and the shape decides the rest.
    """
    if batch_invariant:
        # Numerics must not depend on M or the core count, so the constant wins over the budget.
        return MAX_ROWS_PER_BLOCK
    budget = _UB_TILE_BUDGET // (_LIVE_FP32_TILES * 4 * max(block_n, 1))
    budget = max(1, min(budget, 64))
    rows = triton.next_power_of_2(max(1, min(m, budget)))
    return max(1, min(rows, budget))


def layernorm_gated_infer_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    batch_invariant: bool = False,
) -> Tuple[object, Optional[torch.Tensor], torch.Tensor]:
    """One-pass gated norm. Returns ``((q, s) if quant_heads else out), mean, rstd``."""
    m, n = x.shape
    if group_size is None:
        group_size = n
    if n % group_size:
        raise ValueError(f"group_size={group_size} must divide N={n}")
    ngroups = n // group_size
    if x.stride(-1) != 1:
        raise ValueError("x must be contiguous along the last dimension")

    z_is_3d = z is not None and z.ndim == 3
    if z is not None:
        if z.stride(-1) != 1:
            raise ValueError("z must be contiguous along the last dimension")
        if z_is_3d:
            if z.shape[0] * z.shape[1] != m or z.shape[2] != n:
                raise ValueError(
                    f"3-D z must be [T, H, {n}] with T*H == M == {m}, got {tuple(z.shape)}"
                )
        elif z.shape != (m, n):
            raise ValueError(f"z must be {(m, n)}, got {tuple(z.shape)}")

    if out is None:
        out = torch.empty_like(x)

    mean = (
        torch.empty((ngroups * m,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * m,), dtype=torch.float32, device=x.device)

    max_fused = _MAX_FUSED_SIZE_DIVISOR // x.element_size()
    block_n = min(max_fused, triton.next_power_of_2(group_size))
    if group_size > block_n:
        raise RuntimeError("layernorm_gated does not support a feature dim >= 64 KB")

    num_warps = min(max(block_n // 256, 1), 8)
    rows_per_block = _pick_rows_per_block(m, block_n, batch_invariant)

    quant_kwargs = {}
    result = out
    if quant_heads:
        if ngroups != 1:
            raise ValueError("quant_heads requires a single group")
        if m % quant_heads:
            raise ValueError(f"quant_heads={quant_heads} must divide M={m}")
        num_warps = min(num_warps * quant_heads // rows_per_block, 16)
        # One program per token: the amax must span the whole token, so the block size is
        # forced instead of tuned.
        rows_per_block = quant_heads
        q = torch.empty(
            (m // quant_heads, quant_heads * n), dtype=torch.float8_e4m3fn, device=x.device
        )
        s = torch.empty((m // quant_heads, 1), dtype=torch.float32, device=x.device)
        quant_kwargs = {"Q": q, "S": s, "QUANT": True}
        result = (q, s)

    grid = (triton.cdiv(m, rows_per_block), ngroups)
    _layer_norm_fwd_1pass_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None and not z_is_3d else 0,
        z.stride(0) if z_is_3d else 0,
        z.stride(1) if z_is_3d else 0,
        m,
        group_size,
        eps,
        BLOCK_N=block_n,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        Z_IS_3D=z_is_3d,
        Z_HEADS=z.shape[1] if z_is_3d else 1,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        ACTIVATION=activation,
        num_warps=num_warps,
        **quant_kwargs,
    )
    return result, mean, rstd


def layernorm_gated_infer_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    batch_invariant: bool = False,
) -> Tuple[object, Optional[torch.Tensor], torch.Tensor]:
    """Provider entry point; keeps the argument order of the reference ``_layer_norm_fwd``."""
    return layernorm_gated_infer_impl(
        x,
        weight,
        bias,
        eps,
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=quant_heads,
        batch_invariant=batch_invariant,
    )


def _batch_invariant_flag() -> bool:
    """``LN_GATED_BATCH_INVARIANT=1`` pins the row tile so results do not depend on M."""
    return os.environ.get("LN_GATED_BATCH_INVARIANT", "").strip() not in ("", "0", "false", "False")


def layer_norm_gated_fwd(
    x,
    weight,
    bias,
    z,
    eps,
    out,
    group_size,
    norm_before_gate,
    is_rms_norm,
    activation,
    quant_heads,
):
    """Public op id binding; keeps the upstream positional order and forwards to the kernel entry."""
    return layernorm_gated_infer_fwd(
        x,
        weight,
        bias,
        float(eps),
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=int(quant_heads),
        batch_invariant=_batch_invariant_flag(),
    )
