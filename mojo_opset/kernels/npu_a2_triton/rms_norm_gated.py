# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused RMSNorm + gate kernels adapted for triton-ascend on Huawei NPU.

Grid-stride BT-tiled kernels keep weight/bias resident and vectorize a small row tile under UB.
"""

import functools

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._ascend_ub_manager import ASCEND_MAX_GRID_DIM
from mojo_opset.kernels._ascend_ub_manager import compute_row_tile_block_size
from mojo_opset.kernels._ascend_ub_manager import compute_ub_block_size
from mojo_opset.kernels._ascend_ub_manager import get_ub_manager
from mojo_opset.utils.env import get_bool_env

# Peak live fp32 [BT, BD] tiles per kernel variant; the UB model uses these to cap BT.
# BD is sized from a single-row budget, so a large D can still run with BT=1.
# Keep axis-1 reductions as plain tl.sum: tl.join-based fusion is slower on Ascend
# and the manual log2-halving path overflows UB at D=1024.
_BD_MEM_MULT = 6.0
_HALF_NATIVE = get_bool_env("MOJO_ENABLE_HALF_NATIVE", default=False)
_FWD_MEM_MULT = 4.5
# Masked kernels keep extra fp32 [BT, BD] temporaries live, so cap BT*BD here.
_MASKED_FWD_BT_ELEMS = 8192
_MASKED_BWD_BT_ELEMS = 4096
# BD equals D exactly (non-power-of-2 tl.arange is supported), so every tile is full
# and EVEN_SHAPE can run mask-free when T % BT == 0. Above this width the PoT size is
# used: D > 4096 already forces BT=1, so an exact width buys nothing.
_EXACT_BD_MAX = 4096
# bwd keeps fewer live tiles than fwd: dw/db accumulate per round into [BD] vectors
# and b_y is formed only in the dg path.
_BWD_MEM_MULT = 5.0
_BWD_RECOMPUTE_MEM_MULT = 11.0
_UB_SAFETY_MARGIN = 0.85
# legacy byte cap when UB capacity cannot be detected (65536 // fp32).
_FALLBACK_MAX_BD = 65536 // 4
_MAX_BT = 128
# BD >= 2048 needs extra headroom under multi-buffering.
_LARGE_BD = 2048
_LARGE_BD_FWD_MEM_MULT = 5.5
_LARGE_BD_BWD_MEM_MULT = 16.0
_LARGE_BD_BWD_RECOMPUTE_MEM_MULT = 22.0

# ACTIVATION constexpr: 0 = swish/silu, 1 = sigmoid
_ACTIVATION_SWISH = 0
_ACTIVATION_SIGMOID = 1


def _num_vectorcores(device_index=None) -> int:
    """Return the A2 vector-core count used as the grid-stride program count."""
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
    """Bump multiplier when BD is large enough to stress UB."""
    if BD >= _LARGE_BD:
        return max(base, large_bd_mult)
    return base


def _fwd_memory_multiplier(BD: int) -> float:
    """Return fwd tile multiplier; grid-stride + large BD needs more headroom."""
    return _tile_memory_multiplier(base=_FWD_MEM_MULT, large_bd_mult=_LARGE_BD_FWD_MEM_MULT, BD=BD)


def _bwd_memory_multiplier(BD: int, *, recompute_output: bool = False) -> float:
    """Return bwd tile multiplier; larger BD needs a higher mult (smaller BT).

    ``recompute_output`` needs extra headroom because the kernel keeps pre-gate ``b_y`` live for the ``y`` store.
    """
    if recompute_output:
        return _tile_memory_multiplier(
            base=_BWD_RECOMPUTE_MEM_MULT,
            large_bd_mult=_LARGE_BD_BWD_RECOMPUTE_MEM_MULT,
            BD=BD,
        )
    return _tile_memory_multiplier(base=_BWD_MEM_MULT, large_bd_mult=_LARGE_BD_BWD_MEM_MULT, BD=BD)


# BT candidates for runtime autotuning; BD is a wrapper-computed launch constant.
# EVEN_SHAPE is a correctness flag (mask-free access is only legal when every tile is
# full), so it travels with BT rather than being picked independently.
_AUTOTUNE_BT_VALS = (1, 2, 4, 8, 16, 32, 64, 128)
_AUTOTUNE_FWD_CONFIGS = [
    triton.Config({'BT': bt, 'EVEN_SHAPE': even})
    for bt in _AUTOTUNE_BT_VALS
    for even in (False, True)
]
_AUTOTUNE_BWD_CONFIGS = [
    triton.Config({'BT': bt, 'EVEN_SHAPE': even})
    for bt in _AUTOTUNE_BT_VALS
    for even in (False, True)
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
    """Lock the largest fwd BT allowed by the UB model.

    fwd is sync-bound, so fewer (larger) tiles always win and letting the runtime
    benchmark pick among BTs only adds noise. Oversized masked tiles are halved.
    The wrapper passes keyword arguments only, so ``nargs`` is empty.
    """
    BD = kwargs['BD']
    D = kwargs['D']
    T = kwargs['T']
    max_bt = _autotune_max_bt(BD=BD, memory_multiplier=_fwd_memory_multiplier(BD=BD), T=T)
    best_bt = max(c.kwargs['BT'] for c in configs if c.kwargs['BT'] <= max_bt)
    even = (BD == D) and (T % best_bt == 0)
    if not even and best_bt * BD > _MASKED_FWD_BT_ELEMS:
        best_bt = max(1, best_bt // 2)
        even = (BD == D) and (T % best_bt == 0)
    for c in configs:
        if c.kwargs['BT'] == best_bt and c.kwargs['EVEN_SHAPE'] == even:
            return [c]
    return [configs[0]]


def _prune_bwd_configs(configs, nargs, **kwargs):
    """Lock the largest bwd BT allowed by the UB model (see _prune_fwd_configs)."""
    BD = kwargs['BD']
    D = kwargs['D']
    T = kwargs['T']
    mult = _bwd_memory_multiplier(BD=BD, recompute_output=bool(kwargs['RECOMPUTE_OUTPUT']))
    # dw/db accumulate into [BD] vectors, so the live-tile count does not depend on
    # the weight/bias flags.
    max_bt = _autotune_max_bt(BD=BD, memory_multiplier=mult, T=T)
    best_bt = max(c.kwargs['BT'] for c in configs if c.kwargs['BT'] <= max_bt)
    even = (BD == D) and (T % best_bt == 0)
    if not even and best_bt * BD > _MASKED_BWD_BT_ELEMS:
        best_bt = max(1, best_bt // 2)
        even = (BD == D) and (T % best_bt == 0)
    for c in configs:
        if c.kwargs['BT'] == best_bt and c.kwargs['EVEN_SHAPE'] == even:
            return [c]
    return [configs[0]]


def _get_rms_norm_gated_tiles(
    D: int,
    is_forward: bool,
    *,
    recompute_output: bool = False,
) -> tuple[int, int]:
    """Return (BD, BT) for row-tiled kernels under UB constraints."""
    # BD: fit feature dim with BT=1 using a single-row budget.
    BD = compute_ub_block_size(
        dim_size=D,
        memory_multiplier=_BD_MEM_MULT,
        safety_margin=_UB_SAFETY_MARGIN,
        fallback=_FALLBACK_MAX_BD,
        desired=D if D <= _EXACT_BD_MAX else triton.next_power_of_2(D),
    )
    if D > BD:
        raise RuntimeError(
            f"RMSNormGated feature dim {D} exceeds UB-safe block size {BD}. "
            "Column-tiled kernels are not yet implemented for this size."
        )
    if is_forward:
        memory_multiplier = _fwd_memory_multiplier(BD=BD)
    else:
        memory_multiplier = _bwd_memory_multiplier(BD=BD, recompute_output=recompute_output)
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


@functools.lru_cache(maxsize=None)
def _launch_config(
    T: int,
    D: int,
    device_index: int,
    *,
    is_forward: bool,
    recompute_output: bool = False,
) -> tuple[int, int, int]:
    """Return (BD, BT, NS) for a grid-stride launch over T rows."""
    BD, BT = _get_rms_norm_gated_tiles(
        D=D,
        is_forward=is_forward,
        recompute_output=recompute_output,
    )
    # One grid-stride program per vector core, capped by T and the grid limit.
    NS = max(1, min(_num_vectorcores(device_index), T, ASCEND_MAX_GRID_DIM))
    return BD, BT, NS


@triton.jit
def _store(ptr, off, v, m, HAS_MASK: tl.constexpr):
    """Store with or without a mask; ``m`` is None on the even (mask-free) path."""
    if HAS_MASK:
        tl.store(ptr + off, v, mask=m)
    else:
        tl.store(ptr + off, v)


# Key covers exactly the prune inputs (D -> BD, T -> max_bt/EVEN_SHAPE); flags like
# HAS_WEIGHT change the binary but never the chosen BT.
@triton.autotune(
    configs=_AUTOTUNE_FWD_CONFIGS,
    key=['D', 'T'],
    prune_configs_by={'early_config_prune': _prune_fwd_configs},
)
@triton.jit(do_not_specialize=['T'])
def rms_norm_gated_fwd_kernel(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    rstd,
    eps,
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN_SHAPE: tl.constexpr,
    HALF_NATIVE: tl.constexpr,
):
    """Grid-stride forward: each program owns a BT-row tile stream."""
    i_s = tl.program_id(0)
    # A2 affinity (attnres pattern): contiguous/alignment hints let the compiler
    # emit the widest available vectorized MTE accesses.
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    col_mask = cols < D

    if HAS_WEIGHT:
        b_w = tl.load(w + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + cols, mask=col_mask).to(tl.float32)

    NT = tl.cdiv(T, BT)
    for i_t in range(i_s, NT, NS):
        rows = tl.cast(i_t, tl.int64) * BT + tl.arange(0, BT)
        row_off = rows[:, None] * D + cols[None, :]
        if EVEN_SHAPE and BD % 16 == 0:
            # BD == D and D % 16 == 0: every row starts 16-aligned
            row_off = tl.multiple_of(row_off, (1, 16))
        mask = None
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

        # masked loads already return 0.0 in the padding lanes
        b_var = tl.sum(b_x * b_x, axis=1) / D
        b_rstd = 1 / tl.sqrt(b_var + eps)
        if EVEN_SHAPE:
            tl.store(rstd + rows, b_rstd)
        else:
            tl.store(rstd + rows, b_rstd, mask=row_mask)

        b_x_hat = b_x * b_rstd[:, None]
        b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]

        if EVEN_SHAPE:
            b_g = tl.load(g + row_off)
        else:
            b_g = tl.load(g + row_off, mask=mask, other=0.0)
        if HALF_NATIVE:
            # fp16 keeps the gate chain in the input dtype (2x elements per AIV
            # instruction); bf16 uses fp32 because eps 0.0078 breaks the 1e-3 tol.
            b_gate = tl.sigmoid(b_g)
            b_y = b_y.to(g.dtype.element_ty)
            if ACTIVATION == 0:
                b_y = b_y * b_g * b_gate
            else:
                b_y = b_y * b_gate
        else:
            b_g = b_g.to(tl.float32)
            if ACTIVATION == 0:
                b_y = b_y * b_g * tl.sigmoid(b_g)
            else:
                b_y = b_y * tl.sigmoid(b_g)

        _store(y, row_off, b_y.to(y.dtype.element_ty), mask, HAS_MASK=not EVEN_SHAPE)


# Key = exactly the prune inputs (see the fwd kernel above).
@triton.autotune(
    configs=_AUTOTUNE_BWD_CONFIGS,
    key=['D', 'RECOMPUTE_OUTPUT', 'T'],
    prune_configs_by={'early_config_prune': _prune_bwd_configs},
)
@triton.jit(do_not_specialize=['T'])
def rms_norm_gated_bwd_kernel(
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
    rstd,
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    EVEN_SHAPE: tl.constexpr,
    HALF_NATIVE: tl.constexpr,
):
    """Grid-stride backward: each program owns a BT-row tile stream.

    dg is stored before the dx path so its temporaries are released early and BT stays
    within the UB budget.
    """
    i_s = tl.program_id(0)
    # A2 affinity (attnres pattern): contiguous/alignment hints for the widest
    # vectorized MTE accesses.
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    col_mask = cols < D

    if HAS_WEIGHT:
        b_w = tl.load(w + cols, mask=col_mask).to(tl.float32)
        # Per-round axis=0 reduction into [BD] saves one live [BT, BD] tile.
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + cols, mask=col_mask, other=0.0).to(tl.float32)
        b_db = tl.zeros((BD,), dtype=tl.float32)

    NT = tl.cdiv(T, BT)

    for i_t in range(i_s, NT, NS):
        rows = i_t * BT + tl.arange(0, BT)
        row_off = rows[:, None] * D + cols[None, :]
        if EVEN_SHAPE and BD % 16 == 0:
            # BD == D and D % 16 == 0: every row starts 16-aligned
            row_off = tl.multiple_of(row_off, (1, 16))
        mask = None
        if EVEN_SHAPE:
            b_x = tl.load(x + row_off).to(tl.float32)
        else:
            row_mask = rows < T
            mask = row_mask[:, None] & col_mask[None, :]
            b_x = tl.load(x + row_off, mask=mask, other=0.0).to(tl.float32)

        if EVEN_SHAPE:
            b_rstd = tl.load(rstd + rows)
        else:
            b_rstd = tl.load(rstd + rows, mask=row_mask, other=0.0)
        b_xhat = b_x * b_rstd[:, None]
        if not EVEN_SHAPE:
            b_xhat = tl.where(mask, b_xhat, 0.0)

        # b_y is only needed by the dg path; forming it there keeps one less
        # [BT, BD] tile live.
        if EVEN_SHAPE:
            b_g = tl.load(g + row_off)
            b_dy = tl.load(dy + row_off)
        else:
            b_g = tl.load(g + row_off, mask=mask, other=0.0)
            b_dy = tl.load(dy + row_off, mask=mask, other=0.0)
        if HALF_NATIVE:
            # fp16 keeps the gate/dg chains in the input dtype; bf16 uses fp32
            # because eps 0.0078 breaks the 1e-3 tolerance.
            b_sig = tl.sigmoid(b_g)
            if ACTIVATION == 0:
                # silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
                b_dgate = b_sig * (1 + b_g * (1 - b_sig))
                b_gate = b_g * b_sig
            else:
                b_dgate = b_sig * (1 - b_sig)
                b_gate = b_sig
            if HAS_WEIGHT:
                b_y = b_xhat * b_w[None, :] if not HAS_BIAS else b_xhat * b_w[None, :] + b_b[None, :]
            else:
                b_y = b_xhat
            b_dg = b_dy * b_y.to(b_dy.dtype) * b_dgate
        else:
            b_g = b_g.to(tl.float32)
            b_dy = b_dy.to(tl.float32)
            b_sigmoid_g = tl.sigmoid(b_g)
            if ACTIVATION == 0:
                # silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
                b_dgate = b_sigmoid_g * (1 + b_g * (1 - b_sigmoid_g))
                b_gate = b_g * b_sigmoid_g
            else:
                b_dgate = b_sigmoid_g * (1 - b_sigmoid_g)
                b_gate = b_sigmoid_g
            if HAS_WEIGHT:
                b_y = b_xhat * b_w[None, :] if not HAS_BIAS else b_xhat * b_w[None, :] + b_b[None, :]
            else:
                b_y = b_xhat
            b_dg = b_dy * b_y * b_dgate
        _store(dg, row_off, b_dg.to(dg.dtype.element_ty), mask, HAS_MASK=not EVEN_SHAPE)
        # dg needs pre-gate b_y; the recomputed output must include the gate to match the caller's linear input.
        if RECOMPUTE_OUTPUT:
            _store(y, row_off, (b_y * b_gate).to(y.dtype.element_ty), mask, HAS_MASK=not EVEN_SHAPE)
        b_dy = b_dy * b_gate
        if HALF_NATIVE:
            b_dy = b_dy.to(tl.float32)

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

        b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
        b_dx = (b_wdy - b_xhat * b_c1[:, None]) * b_rstd[:, None]

        if HAS_DRESIDUAL:
            if EVEN_SHAPE:
                b_dx += tl.load(dresidual + row_off).to(tl.float32)
            else:
                b_dx += tl.load(dresidual + row_off, mask=mask, other=0.0).to(tl.float32)
        if STORE_DRESIDUAL:
            _store(dresidual_in, row_off, b_dx.to(dresidual_in.dtype.element_ty), mask, HAS_MASK=not EVEN_SHAPE)
        _store(dx, row_off, b_dx.to(dx.dtype.element_ty), mask, HAS_MASK=not EVEN_SHAPE)

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + cols, b_dw, mask=col_mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + cols, b_db, mask=col_mask)


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
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)

    BD, _, NS = _launch_config(T=T, D=D, device_index=x.device.index, is_forward=True)
    act_id = _activation_id(activation=activation)
    rms_norm_gated_fwd_kernel[(NS,)](
        x=x,
        g=g,
        y=y,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_out,
        rstd=rstd,
        eps=eps,
        T=T,
        NS=NS,
        D=D,
        BD=BD,
        ACTIVATION=act_id,
        STORE_RESIDUAL_OUT=residual_out is not None,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        HALF_NATIVE=_HALF_NATIVE and (x.dtype == torch.float16),
    )
    return y, None, rstd, residual_out if residual_out is not None else x


def rms_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    rstd: torch.Tensor | None = None,
    dresidual: torch.Tensor | None = None,
    has_residual: bool = False,
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
        recompute_output=recompute_output,
    )

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None

    act_id = _activation_id(activation=activation)
    rms_norm_gated_bwd_kernel[(NS,)](
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
        rstd=rstd,
        T=T,
        NS=NS,
        D=D,
        BD=BD,
        ACTIVATION=act_id,
        STORE_DRESIDUAL=dresidual_in is not None,
        HAS_DRESIDUAL=dresidual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        RECOMPUTE_OUTPUT=y is not None,
        HALF_NATIVE=_HALF_NATIVE and (x.dtype == torch.float16),
    )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dg, dw, db, dresidual_in) if not recompute_output else (dx, dg, dw, db, dresidual_in, y)
