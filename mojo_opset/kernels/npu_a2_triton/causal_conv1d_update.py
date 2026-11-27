# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Extracted from flash-linear-attention @749524ca:
#   fla/modules/causal_conv1d/ops.py#L351-L437  (causal_conv1d_update_kernel)
#   fla/modules/causal_conv1d/ops.py#L868-L929  (causal_conv1d_update)
#
# Single-token decode update of a causal depthwise conv1d state (A2 / Ascend
# 910B Triton only), fp32 accumulation.
#
# State layout: rotation planes instead of a shifted window.  The buffer is
# ``W`` planes of ``[N, D]`` plus a cursor ``head`` ("plane p holds the token
# of age (head - p) mod W"):
#
#     plane[head] = x
#     y = x * w[W-1] + sum_{j < W-1} plane[(head+1+j) % W] * w[j]
#     head = (head + 1) % W
#
# One step is a pure elementwise pass over ``[N, D]`` tiles -- traffic
# ``(W + 2) * N * D`` instead of ``2 * (W + 1) * N * D`` for the shifted
# window, and no broadcast / insert / W-wide reduce.
#
# A2 notes (910B3):
#   * Write x into ``cache[head]`` *after* the plane loads; first, the
#     aliasing store serialises every later load (43.2 -> 37.9 us).
#   * One row block per program, not two: ``[BM, BD]`` with ``BM * BD == 4096``
#     at ``BD = 512`` moves each buffer in a single DMA and keeps VEC packed
#     (41.0 -> 36.7 us, bound 0.80 -> 0.92).  ``[8, 1024]`` overflows UB,
#     ``BD = 256`` pays the per-program fill/drain.
#   * ``W = 2`` has a single plane to move and prefers ``[16, 512]``; at
#     ``W = 4`` that block stalls (bound 0.77), hence the width prune.
#   * The conv weight is transposed to ``[W, D]`` once in the wrapper; reading
#     the upstream ``[D, W]`` layout gathers at stride ``W`` (~140x slower).
#   * Rejected: software pipelining, extra tiles, ``tl.fma``, ``num_warps`` --
#     they only add vector instructions; the kernel is VEC-bound.

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._triton_utils import input_guard


def _prune_by_width(configs, named_args, **kwargs):
    """W = 2 wants the wider block, W >= 3 two rows (see the notes above)."""
    W = kwargs.get("W", named_args.get("W", 4))
    wide = W <= 2
    return [c for c in configs if (c.kwargs["BM"] == 16) == wide]


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BD": BD, "BM": BM}) for BD, BM in ((512, 8), (512, 16))],
    key=["D", "W", "N"],
    prune_configs_by={"early_config_prune": _prune_by_width},
    restore_value=["cache"],
)
@triton.jit
def causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    PL,  # N * D; kept a *runtime* scalar so ``p * PL`` folds into the DMA base
    # (as a constexpr it becomes an index-vector add, +4.5 us).
    N: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    HEAD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0)
    ND: tl.constexpr = (D + BD - 1) // BD
    EVEN: tl.constexpr = (D % BD == 0) and (N % BM == 0)
    EVEN_D: tl.constexpr = D % BD == 0
    i_d = pid % ND
    i_n = (pid // ND) * BM

    rows = i_n + tl.arange(0, BM)
    cols = i_d * BD + tl.arange(0, BD)
    o = rows[:, None] * D + cols[None, :]

    if EVEN:
        xv = tl.load(x + o)
    else:
        m = (rows[:, None] < N) & (cols[None, :] < D)
        xv = tl.load(x + o, mask=m, other=0.0)

    # weights arrive transposed as [W, D]: unit-stride rows.  ``w<j>`` belongs
    # to the j-th oldest token, the newest takes ``w[W-1]``.
    if HAS_WEIGHT:
        w_off = (W - 1) * D
        if EVEN_D:
            w_last = tl.load(weight + w_off + cols).to(tl.float32)
        else:
            w_last = tl.load(weight + w_off + cols, mask=cols < D, other=0.0).to(tl.float32)
        a = xv.to(tl.float32) * w_last[None, :]
    else:
        a = xv.to(tl.float32)

    # older tokens: plane (HEAD + 1 + j) % W carries w[j]
    for j in tl.static_range(W - 1):
        p = (HEAD + 1 + j) % W
        if EVEN:
            q = tl.load(cache + p * PL + o)
        else:
            q = tl.load(cache + p * PL + o, mask=m, other=0.0)
        if HAS_WEIGHT:
            if EVEN_D:
                wj = tl.load(weight + j * D + cols).to(tl.float32)
            else:
                wj = tl.load(weight + j * D + cols, mask=cols < D, other=0.0).to(tl.float32)
            a += q.to(tl.float32) * wj[None, :]
        else:
            a += q.to(tl.float32)

    # store x last -- written first it aliases the plane loads above
    if EVEN:
        tl.store(cache + HEAD * PL + o, xv.to(cache.dtype.element_ty))
    else:
        tl.store(cache + HEAD * PL + o, xv.to(cache.dtype.element_ty), mask=m)

    if HAS_BIAS:
        if EVEN_D:
            bv = tl.load(bias + cols).to(tl.float32)
        else:
            bv = tl.load(bias + cols, mask=cols < D, other=0.0).to(tl.float32)
        a += bv[None, :]

    if ACTIVATION == "swish" or ACTIVATION == "silu":
        a = a * tl.sigmoid(a)

    if HAS_RESIDUAL:
        if EVEN:
            r = tl.load(residual + o).to(tl.float32)
        else:
            r = tl.load(residual + o, mask=m, other=0.0).to(tl.float32)
        a += r

    yo = tl.cast(a, y.dtype.element_ty, fp_downcast_rounding="rtne")
    if EVEN:
        tl.store(y + o, yo)
    else:
        tl.store(y + o, yo, mask=m)


# Transposed-weight memo, keyed on (data_ptr, _version, shape, dtype); the value
# holds the source tensor so a recycled pointer never serves a stale transpose.
_WT_CACHE: "dict[tuple, tuple[torch.Tensor, torch.Tensor]]" = {}
_WT_CACHE_MAX = 8


def _transposed_weight(weight: torch.Tensor) -> torch.Tensor:
    """``[D, W]`` -> contiguous ``[W, D]``, memoised on the tensor's identity."""
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    hit = _WT_CACHE.get(key)
    if hit is not None and hit[0] is weight:
        return hit[1]
    if len(_WT_CACHE) >= _WT_CACHE_MAX:
        _WT_CACHE.pop(next(iter(_WT_CACHE)))
    wt = weight.detach().t().contiguous()
    _WT_CACHE[key] = (weight, wt)
    return wt


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_update_fwd(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    head: int = 0,
):
    """Update the rotation state ``cache`` with one token of ``x``; return ``(y, cache)``.

    ``x`` is ``[N, D]``, ``[1, N, D]`` or ``[N, 1, D]``, ``weight`` is
    ``[D, W]``, ``cache`` is ``[W, N, D]``.  ``head`` names the plane that
    receives ``x``; advance it with ``head = (head + 1) % W`` per decoding step.
    """
    shape = x.shape
    if weight is not None and x.shape[-1] != weight.shape[0] and x.dim() >= 3:
        x = x.reshape(x.shape[0], x.shape[1], -1)

    D = x.shape[-1]
    N = x.numel() // D
    # The state layout fixes W even when the weights are omitted.
    W = weight.shape[1] if weight is not None else cache.shape[0]
    head = int(head) % W

    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    x_2d = x.reshape(N, D)
    y_2d = y.reshape(N, D)

    wt = _transposed_weight(weight) if weight is not None else None

    def grid(meta) -> tuple[int, ...]:
        return (triton.cdiv(D, meta["BD"]) * triton.cdiv(N, meta["BM"]),)

    causal_conv1d_update_kernel[grid](
        x=x_2d,
        cache=cache,
        residual=None if residual is None else residual.reshape(N, D),
        y=y_2d,
        weight=wt,
        bias=bias,
        PL=N * D,
        N=N,
        D=D,
        W=W,
        HEAD=head,
        ACTIVATION=activation,
    )
    return y.view(shape), cache
