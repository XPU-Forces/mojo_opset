# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused LayerNorm/RMSNorm + gate kernels adapted for triton-ascend on Huawei NPU.

Grid-stride BT-tiled kernels keep weight/bias resident and vectorize a small row tile under UB.
Ascend910 @ D=1024 (multi-buffer compile-validated): fwd BT=8 / bwd BT=4.
"""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._ascend_ub_manager import ASCEND_MAX_GRID_DIM
from mojo_opset.kernels._ascend_ub_manager import compute_row_tile_block_size
from mojo_opset.kernels._ascend_ub_manager import compute_ub_block_size
from mojo_opset.kernels._ascend_ub_manager import get_ub_manager

# peak live fp32 tiles relative to [BT, BD].
# BD uses a single-row budget so large D is not rejected when BT can still be 1.
# fwd grid-stride keeps w/b resident (needs higher mult than one-tile launch).
# fwd calibrated on Ascend950PR (248 KiB UB, 0.85 margin): mult=4.5 → BT=8 @ D=1024,
# large-BD mult=5.5 → BT=4 @ D>=1200 (3.0/4.0 over-plan BT that overflows UB and fail compiles).
# bwd stores dg before the dx path. bwd BT=4 @ D=1024 (mult=8).
# do not lower bwd mult below ~6 without re-validating (BT=8 bwd overflows).
# recompute-output (norm+linear bwd) keeps b_y live for tl.store(y);
# bwd BT=2 @ D=1024 (mult=11). Do not lower without re-validating.
_BD_MEM_MULT = 6.0
_FWD_MEM_MULT = 4.5
_BWD_MEM_MULT = 8.0
_BWD_RECOMPUTE_MEM_MULT = 11.0
_UB_SAFETY_MARGIN = 0.85
# legacy byte cap when UB capacity cannot be detected (65536 // fp32).
_FALLBACK_MAX_BD = 65536 // 4
_MAX_BT = 128
# PoT-padded BD>=2048 needs extra headroom under multi-buffering.
# Ascend950PR calibration: the RMS bwd variant plans ~14.7x [BT,BD] live at
# BD=2048 (fp32 temps/accumulators do not shrink with input dtype), so its
# mult must reach 16 to drop BT to 1. LN stays at the 12x calibration, which
# keeps BT=2 at BD=2048 (compile-validated on Ascend950PR).
_LARGE_BD = 2048
_LARGE_BD_FWD_MEM_MULT = 5.5
_LARGE_BD_BWD_MEM_MULT = 12.0
_LARGE_BD_BWD_RMS_MEM_MULT = 16.0
_LARGE_BD_BWD_RECOMPUTE_MEM_MULT = 16.5
_LARGE_BD_BWD_RMS_RECOMPUTE_MEM_MULT = 22.0
# Largest D that uses an exact (non-power-of-2) BD. Beyond this the UB budget caps
# BD at 4096 and D > BD raises, so there is nothing to gain from going higher.
_EXACT_BD_MAX = 4096

# ACTIVATION constexpr: 0 = swish/silu, 1 = sigmoid
_ACTIVATION_SWISH = 0
_ACTIVATION_SIGMOID = 1


def _num_vectorcores(device_index=None) -> int:
    """Return the A5 vector-core count used as the grid-stride program count."""
    try:
        props = triton.runtime.driver.active.utils.get_device_properties("npu")
        return max(1, int(props["num_vectorcore"]))
    except Exception:
        return 1


def _activation_id(activation: str) -> int:
    if activation in ("swish", "silu"):
        return _ACTIVATION_SWISH
    if activation == "sigmoid":
        return _ACTIVATION_SIGMOID
    raise ValueError(f"Unsupported activation: {activation}")


def _tile_memory_multiplier(base: float, large_bd_mult: float, BD: int) -> float:
    """Bump multiplier when PoT-padded BD is large enough to stress UB."""
    if BD >= _LARGE_BD:
        return max(base, large_bd_mult)
    return base


def _fwd_memory_multiplier(BD: int) -> float:
    """Return fwd tile multiplier; grid-stride + large BD needs more headroom."""
    return _tile_memory_multiplier(base=_FWD_MEM_MULT, large_bd_mult=_LARGE_BD_FWD_MEM_MULT, BD=BD)


def _bwd_memory_multiplier(is_rms_norm: bool, BD: int, *, recompute_output: bool = False) -> float:
    """Return bwd tile multiplier; larger BD needs a higher mult (smaller BT).

    ``is_rms_norm`` selects the large-BD calibration: the RMS variant plans
    more live UB at BD>=2048 (mult=16 -> BT=1) than LN (mult=12 -> BT=2).
    ``recompute_output`` needs extra headroom because the kernel keeps pre-gate ``b_y`` live for the ``y`` store.
    """
    if recompute_output:
        large_bd_mult = _LARGE_BD_BWD_RMS_RECOMPUTE_MEM_MULT if is_rms_norm else _LARGE_BD_BWD_RECOMPUTE_MEM_MULT
        return _tile_memory_multiplier(base=_BWD_RECOMPUTE_MEM_MULT, large_bd_mult=large_bd_mult, BD=BD)
    large_bd_mult = _LARGE_BD_BWD_RMS_MEM_MULT if is_rms_norm else _LARGE_BD_BWD_MEM_MULT
    return _tile_memory_multiplier(base=_BWD_MEM_MULT, large_bd_mult=large_bd_mult, BD=BD)


# BT candidates for runtime autotuning. BD is fixed by D (single-row UB budget) and
# stays a wrapper-computed launch constant, so only BT varies.
# fwd/bwd additionally carry EVEN_SHAPE in each config: it is a correctness flag
# (mask-free load/store is only legal when every tile is full), so it must travel
# with BT rather than be picked independently.
_AUTOTUNE_BT_VALS = (1, 2, 4, 8, 16, 32, 64, 128)
_AUTOTUNE_FWD_CONFIGS = [
    triton.Config({'BT': bt, 'EVEN_SHAPE': even})
    for bt in _AUTOTUNE_BT_VALS
    for even in (True,)
]
_AUTOTUNE_BWD_CONFIGS = [
    triton.Config({'BT': bt, 'EVEN_SHAPE': even})
    for bt in _AUTOTUNE_BT_VALS
    for even in (True,)
]


def _autotune_max_bt(BD: int, memory_multiplier: float, T: int) -> int:
    """Cap BT by the fp32 UB tile model (kernel temps stay fp32 regardless of input
    dtype) and by T; configs that still overflow UB fail MLIR compilation during
    tuning and are dropped (timed as inf)."""
    ub_bits = get_ub_manager().ub_capacity_bits
    safe_bits = int(ub_bits * _UB_SAFETY_MARGIN)
    max_bt = safe_bits // int(memory_multiplier * BD * 4 * 8)
    return max(1, min(_MAX_BT, max_bt, max(1, T)))


def _prune_fwd_configs(configs, nargs, **kwargs):
    """Prune fwd configs by the UB model, keeping all BT values at or below the
    host-estimated cap. The UB model is a lower bound on actual UB demand (it cannot
    see compiler multi-buffering or temp spills), so configs that overflow at compile
    time are dropped by the autotuner (timed as inf). Returning multiple candidates
    lets the autotuner pick the fastest BT that actually compiles, instead of locking
    a single estimate that may fail."""
    BD = kwargs['BD']
    D = kwargs['D']
    T = kwargs['T']
    max_bt = _autotune_max_bt(BD=BD, memory_multiplier=_fwd_memory_multiplier(BD=BD), T=T)
    even = (BD == D)
    pruned = [c for c in configs
              if c.kwargs['BT'] <= max_bt
              and c.kwargs['EVEN_SHAPE'] == (even and T % c.kwargs['BT'] == 0)]
    return pruned if pruned else [configs[0]]


def _prune_bwd_configs(configs, nargs, **kwargs):
    """Prune bwd configs by the UB model, keeping all BT values at or below the
    host-estimated cap. The UB model is a lower bound (compiler multi-buffering and
    temp spills are invisible to it), so configs that overflow at compile time are
    dropped by the autotuner (timed as inf). Returning multiple candidates lets the
    autotuner pick the fastest BT that actually compiles, instead of locking a single
    estimate that may fail — e.g. D=1536 BWD: model says BT=4 is safe (mult=8.0) but
    actual UB demand is ~13.5x, so BT=4 overflows; BT=2 compiles and is auto-selected."""
    BD = kwargs['BD']
    D = kwargs['D']
    T = kwargs['T']
    mult = _bwd_memory_multiplier(
        is_rms_norm=bool(kwargs['IS_RMS_NORM']),
        BD=BD,
        recompute_output=bool(kwargs['RECOMPUTE_OUTPUT']),
    )
    max_bt = _autotune_max_bt(BD=BD, memory_multiplier=mult, T=T)
    even = (BD == D)
    pruned = [c for c in configs
              if c.kwargs['BT'] <= max_bt
              and c.kwargs['EVEN_SHAPE'] == (even and T % c.kwargs['BT'] == 0)]
    return pruned if pruned else [configs[0]]


def _get_layer_norm_gated_tiles(
    D: int,
    is_forward: bool,
    *,
    is_rms_norm: bool = False,
    recompute_output: bool = False,
) -> tuple[int, int]:
    """Return (BD, BT) for row-tiled kernels under UB constraints."""
    # BD: fit feature dim with BT=1 using a single-row budget.
    # ``desired`` is an upper bound, not a target: compute_ub_block_size returns
    # min(desired, UB budget). Passing next_power_of_2(D) therefore rounds the tile
    # UP past D, and the masked lanes still occupy vector-unit slots -- the mask only
    # suppresses the out-of-range accesses, not the arithmetic. At D=1200 that left
    # only 1200/2048 = 58.6% of each tile doing useful work (VEC-bound at 0.60), and
    # D=50 wasted 14/64 lanes.
    # triton-ascend accepts non-power-of-2 tl.arange (verified BD=1200/1280/1536
    # compile and produce correct results), so use the exact D and let the UB model
    # cap it. Measured on Ascend950PR: D=1200 fwd -64.9% / bwd -64.0%, D=50 fwd
    # -46.6% / bwd -41.3%. A smaller BD also frees UB budget for a larger BT, which
    # halves the tile count and therefore the SET_FLAG/WAIT_FLAG sync pairs.
    # Exact BD is kept up to _EXACT_BD_MAX. Above that the UB budget caps BD at 4096
    # anyway (D > BD raises below), so the PoT branch is only a safety net for the
    # multi-buffered large-tile path; it was calibrated against PoT shapes, so do not
    # widen this further without re-measuring _LARGE_BD_*_MEM_MULT.
    # Measured on Ascend950PR (T=8192, fp32): D=2049 fwd 641.0 -> 210.5 us (-67.2%),
    # D=3000 fwd 534.4 -> 234.3 us (-56.2%). Both also fix a pre-existing bwd UB
    # overflow at BD=4096 (needed 2491392 bits vs 1200000 available).
    BD = compute_ub_block_size(
        dim_size=D,
        memory_multiplier=_BD_MEM_MULT,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=_FALLBACK_MAX_BD,
        desired=(D if D <= _EXACT_BD_MAX else triton.next_power_of_2(D)),
    )
    if D > BD:
        raise RuntimeError(
            f"LayerNormGated feature dim {D} exceeds UB-safe block size {BD}. "
            "Column-tiled kernels are not yet implemented for this size."
        )
    if is_forward:
        memory_multiplier = _fwd_memory_multiplier(BD=BD)
    else:
        memory_multiplier = _bwd_memory_multiplier(is_rms_norm=is_rms_norm, BD=BD, recompute_output=recompute_output)
    # large synthetic row dim so BT is limited by UB, not by a host-side T guess.
    BT = compute_row_tile_block_size(
        row_dim=1 << 20,
        fixed_dim=BD,
        memory_multiplier=memory_multiplier,
        tiling_row=True,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=16,
        min_block=1,
        max_block=_MAX_BT,
    )
    return BD, BT


def _launch_config(
    T: int,
    D: int,
    device_index: int,
    *,
    is_forward: bool,
    is_rms_norm: bool = False,
    recompute_output: bool = False,
) -> tuple[int, int, int]:
    """Return (BD, BT, NS) for a grid-stride launch over T rows."""
    BD, BT = _get_layer_norm_gated_tiles(
        D=D,
        is_forward=is_forward,
        is_rms_norm=is_rms_norm,
        recompute_output=recompute_output,
    )
    # BT here is only the host UB-model bound; the kernels autotune the launched BT,
    # so NS must not depend on it (idle programs just store their zeroed dw/db rows).
    NS = max(1, min(_num_vectorcores(device_index), T, ASCEND_MAX_GRID_DIM))
    return BD, BT, NS


# The autotune key only lists what the prune decision actually depends on
# (D -> BD, T -> max_bt/EVEN_SHAPE). Feature flags such as HAS_WEIGHT or
# ACTIVATION change the compiled binary but not the chosen BT, so keeping them
# here would only re-benchmark the same deterministic answer under a new key.
@triton.autotune(
    configs=_AUTOTUNE_FWD_CONFIGS,
    key=['D', 'IS_RMS_NORM', 'T'],
    prune_configs_by={'early_config_prune': _prune_fwd_configs},
)
@triton.jit(do_not_specialize=['T'])
def layer_norm_gated_fwd_kernel(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    eps,
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN_SHAPE: tl.constexpr,
):
    """Grid-stride forward: each program owns a BT-row tile stream."""
    i_s = tl.program_id(0)
    cols = tl.arange(0, BD)
    col_mask = cols < D

    if HAS_WEIGHT:
        b_w = tl.load(w + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + cols, mask=col_mask).to(tl.float32)

    NT = tl.cdiv(T, BT)
    for i_t in range(i_s, NT, NS):
        rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        row_off = rows[:, None] * D + cols[None, :]
        if not EVEN_SHAPE:
            row_mask = rows < T
            mask = row_mask[:, None] & col_mask[None, :]

        if EVEN_SHAPE:
            b_x = tl.load(x + row_off).to(tl.float32)
        else:
            b_x = tl.load(x + row_off, mask=mask, other=0.0).to(tl.float32)

        if HAS_RESIDUAL:
            if EVEN_SHAPE:
                b_x += tl.load(residual + row_off).to(tl.float32)
            else:
                b_x += tl.load(residual + row_off, mask=mask, other=0.0).to(tl.float32)
        if STORE_RESIDUAL_OUT:
            if EVEN_SHAPE:
                tl.store(residual_out + row_off, b_x.to(residual_out.dtype.element_ty))
            else:
                tl.store(residual_out + row_off, b_x.to(residual_out.dtype.element_ty), mask=mask)

        if not IS_RMS_NORM:
            b_mean = tl.sum(b_x, axis=1) / D
            if EVEN_SHAPE:
                tl.store(mean + rows, b_mean)
            else:
                tl.store(mean + rows, b_mean, mask=row_mask)
            b_xbar = b_x - b_mean[:, None] if EVEN_SHAPE else tl.where(mask, b_x - b_mean[:, None], 0.0)
            b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
        else:
            b_xbar = b_x if EVEN_SHAPE else tl.where(mask, b_x, 0.0)
            b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
        b_rstd = 1 / tl.sqrt(b_var + eps)
        if EVEN_SHAPE:
            tl.store(rstd + rows, b_rstd)
        else:
            tl.store(rstd + rows, b_rstd, mask=row_mask)

        b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
        b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]

        if EVEN_SHAPE:
            b_g = tl.load(g + row_off).to(tl.float32)
        else:
            b_g = tl.load(g + row_off, mask=mask, other=0.0).to(tl.float32)

        if ACTIVATION == 0:
            b_y = b_y * b_g * tl.sigmoid(b_g)
        else:
            b_y = b_y * tl.sigmoid(b_g)

        if EVEN_SHAPE:
            tl.store(y + row_off, b_y.to(y.dtype.element_ty))
        else:
            tl.store(y + row_off, b_y.to(y.dtype.element_ty), mask=mask)


# Key = exactly the prune inputs: D -> BD, IS_RMS_NORM/RECOMPUTE_OUTPUT -> the
# large-BD memory multiplier, T -> max_bt/EVEN_SHAPE. HAS_WEIGHT, STORE_DRESIDUAL
# change the compiled binary but never the chosen BT.
@triton.autotune(
    configs=_AUTOTUNE_BWD_CONFIGS,
    key=['D', 'IS_RMS_NORM', 'RECOMPUTE_OUTPUT', 'T'],
    prune_configs_by={'early_config_prune': _prune_bwd_configs},
)
@triton.jit(do_not_specialize=['T'])
def layer_norm_gated_bwd_kernel(
    x,
    g,
    w,
    b,
    y,
    dy,
    dx,
    dg,
    dw,
    db,
    dresidual,
    dresidual_in,
    mean,
    rstd,
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    EVEN_SHAPE: tl.constexpr,
):
    """Grid-stride backward: each program owns a BT-row tile stream.

    Gate gradients are stored before the dx path, so Ascend can release their temporaries and keep BT within the UB budget.
    """
    i_s = tl.program_id(0)
    cols = tl.arange(0, BD)
    col_mask = cols < D

    if HAS_WEIGHT:
        b_w = tl.load(w + cols, mask=col_mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + cols, mask=col_mask, other=0.0).to(tl.float32)
        b_db = tl.zeros((BD,), dtype=tl.float32)

    NT = tl.cdiv(T, BT)

    for i_t in range(i_s, NT, NS):
        rows = i_t * BT + tl.arange(0, BT)
        row_off = rows[:, None] * D + cols[None, :]
        if EVEN_SHAPE:
            b_x = tl.load(x + row_off).to(tl.float32)
        else:
            mask = (rows < T)[:, None] & col_mask[None, :]
            b_x = tl.load(x + row_off, mask=mask, other=0.0).to(tl.float32)

        if not IS_RMS_NORM:
            if EVEN_SHAPE:
                b_mean = tl.load(mean + rows)
            else:
                b_mean = tl.load(mean + rows, mask=rows < T, other=0.0)
        if EVEN_SHAPE:
            b_rstd = tl.load(rstd + rows)
        else:
            b_rstd = tl.load(rstd + rows, mask=rows < T, other=0.0)
        b_xhat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
        if not EVEN_SHAPE:
            mask = (rows < T)[:, None] & col_mask[None, :]
            b_xhat = tl.where(mask, b_xhat, 0.0)

        b_y = b_xhat * b_w[None, :] if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]

        if EVEN_SHAPE:
            b_g = tl.load(g + row_off).to(tl.float32)
        else:
            b_g = tl.load(g + row_off, mask=mask, other=0.0).to(tl.float32)
        if EVEN_SHAPE:
            b_dy = tl.load(dy + row_off).to(tl.float32)
        else:
            b_dy = tl.load(dy + row_off, mask=mask, other=0.0).to(tl.float32)
        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 0:
            # silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
            b_dsilu = b_sigmoid_g * (1 + b_g * (1 - b_sigmoid_g))
            b_gate = b_g * b_sigmoid_g
            if EVEN_SHAPE:
                tl.store(dg + row_off, (b_dy * b_y * b_dsilu).to(dg.dtype.element_ty))
            else:
                tl.store(dg + row_off, (b_dy * b_y * b_dsilu).to(dg.dtype.element_ty), mask=mask)
        else:
            b_gate = b_sigmoid_g
            if EVEN_SHAPE:
                tl.store(dg + row_off, (b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)).to(dg.dtype.element_ty))
            else:
                tl.store(dg + row_off, (b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)).to(dg.dtype.element_ty), mask=mask)
        # dg needs pre-gate b_y; the recomputed output must include the gate to match the caller's linear input.
        if RECOMPUTE_OUTPUT:
            if EVEN_SHAPE:
                tl.store(y + row_off, (b_y * b_gate).to(y.dtype.element_ty))
            else:
                tl.store(y + row_off, (b_y * b_gate).to(y.dtype.element_ty), mask=mask)
        b_dy = b_dy * b_gate

        if HAS_WEIGHT:
            if EVEN_SHAPE:
                b_dw += tl.sum(b_dy * b_xhat, axis=0)
            else:
                b_dw += tl.sum(tl.where(mask, b_dy * b_xhat, 0.0), axis=0)
            b_wdy = b_dy * b_w[None, :]
        else:
            b_wdy = b_dy
        if HAS_BIAS:
            if EVEN_SHAPE:
                b_db += tl.sum(b_dy, axis=0)
            else:
                b_db += tl.sum(tl.where(mask, b_dy, 0.0), axis=0)

        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_c2 = tl.sum(b_wdy, axis=1) / D
            b_dx = (b_wdy - (b_xhat * b_c1[:, None] + b_c2[:, None])) * b_rstd[:, None]
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_dx = (b_wdy - b_xhat * b_c1[:, None]) * b_rstd[:, None]

        if HAS_DRESIDUAL:
            if EVEN_SHAPE:
                b_dx += tl.load(dresidual + row_off).to(tl.float32)
            else:
                b_dx += tl.load(dresidual + row_off, mask=mask, other=0.0).to(tl.float32)
        if STORE_DRESIDUAL:
            if EVEN_SHAPE:
                tl.store(dresidual_in + row_off, b_dx.to(dresidual_in.dtype.element_ty))
            else:
                tl.store(dresidual_in + row_off, b_dx.to(dresidual_in.dtype.element_ty), mask=mask)
        if EVEN_SHAPE:
            tl.store(dx + row_off, b_dx.to(dx.dtype.element_ty))
        else:
            tl.store(dx + row_off, b_dx.to(dx.dtype.element_ty), mask=mask)

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + cols, b_dw, mask=col_mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + cols, b_db, mask=col_mask)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)

    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)

    BD, _, NS = _launch_config(T=T, D=D, device_index=x.device.index, is_forward=True, is_rms_norm=is_rms_norm)
    act_id = _activation_id(activation=activation)
    layer_norm_gated_fwd_kernel[(NS,)](
        x=x,
        g=g,
        y=y,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_out,
        mean=mean,
        rstd=rstd,
        eps=eps,
        T=T,
        NS=NS,
        D=D,
        BD=BD,
        ACTIVATION=act_id,
        IS_RMS_NORM=is_rms_norm,
        STORE_RESIDUAL_OUT=residual_out is not None,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    return y, mean, rstd, residual_out if residual_out is not None else x


def layer_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    mean: torch.Tensor | None = None,
    rstd: torch.Tensor | None = None,
    dresidual: torch.Tensor | None = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype | None = None,
    recompute_output: bool = False,
):
    T, D = x.shape
    assert dy.shape == (T, D)
    if dresidual is not None:
        assert dresidual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)

    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dg = torch.empty_like(g) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    BD, _, NS = _launch_config(
        T=T,
        D=D,
        device_index=x.device.index,
        is_forward=False,
        is_rms_norm=is_rms_norm,
        recompute_output=recompute_output,
    )

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None

    act_id = _activation_id(activation=activation)
    layer_norm_gated_bwd_kernel[(NS,)](
        x=x,
        g=g,
        w=weight,
        b=bias,
        y=y,
        dy=dy,
        dx=dx,
        dg=dg,
        dw=dw,
        db=db,
        dresidual=dresidual,
        dresidual_in=dresidual_in,
        mean=mean,
        rstd=rstd,
        T=T,
        NS=NS,
        D=D,
        BD=BD,
        ACTIVATION=act_id,
        IS_RMS_NORM=is_rms_norm,
        STORE_DRESIDUAL=dresidual_in is not None,
        HAS_DRESIDUAL=dresidual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        RECOMPUTE_OUTPUT=y is not None,
    )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dg, dw, db, dresidual_in) if not recompute_output else (dx, dg, dw, db, dresidual_in, y)


def rms_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
):
    return layer_norm_gated_fwd(
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        residual=residual,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        is_rms_norm=True,
    )


def rms_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    mean: torch.Tensor | None = None,
    rstd: torch.Tensor | None = None,
    dresidual: torch.Tensor | None = None,
    has_residual: bool = False,
    x_dtype: torch.dtype | None = None,
    recompute_output: bool = False,
):
    return layer_norm_gated_bwd(
        dy=dy,
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        mean=mean,
        rstd=rstd,
        dresidual=dresidual,
        has_residual=has_residual,
        is_rms_norm=True,
        x_dtype=x_dtype,
        recompute_output=recompute_output,
    )
