"""Gated LayerNorm / RMSNorm with an optional FP8 per-token quant epilogue.

Public entry point for the SGLang FLA ``layernorm_gated`` family: one fused kernel does the
normalisation, the gate, the affine and (optionally) the quantisation, replacing the multi
kernel eager chain.

The capability matrix is carried by ``constexpr`` switches, so a given call compiles one
specialisation:

===========================================  ==============================================
``is_rms_norm``                              LayerNorm (mean/rstd out) vs RMSNorm (rstd only)
``z``                                        no gate, 2-D ``[M, N]`` gate, or 3-D ``[T, H, D]``
                                             strided gate with ``T*H == M``
``norm_before_gate``                         ``norm(x) * act(z)`` vs ``norm(x * act(z))``
``activation``                               ``"swish"`` / ``"silu"`` / ``"sigmoid"``
``bias``                                     affine with or without the shift
``group_size``                               whole-row norm vs group norm (``N % group_size``)
``quant_heads``                              FP8 e4m3 per-token quant epilogue (950)
===========================================  ==============================================
"""

from typing import Optional
from typing import Tuple
from typing import Union

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

_ACTIVATIONS = ("swish", "silu", "sigmoid")


def _validate(
    x,
    weight,
    bias,
    z,
    eps,
    group_size,
    activation,
    quant_heads,
    op,
):
    if x.ndim != 2:
        raise ValueError(f"x must be a 2-D [M, N] view, got {tuple(x.shape)}")
    m, n = x.shape
    if not x.is_floating_point():
        raise ValueError("x must be floating point")
    if x.stride(-1) != 1:
        raise ValueError("x must be contiguous along the last dimension")

    if weight.ndim != 1 or weight.shape[0] != n:
        raise ValueError(f"weight must be [{n}], got {tuple(weight.shape)}")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != n):
        raise ValueError(f"bias must be [{n}], got {tuple(bias.shape)}")
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")

    if group_size is not None:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if n % group_size:
            raise ValueError(f"group_size={group_size} must divide N={n}")
        # 64 KB per feature is the fused-kernel limit the reference enforces.
        if group_size > 65536 // x.element_size():
            raise ValueError(
                f"group_size={group_size} exceeds the fused limit "
                f"{65536 // x.element_size()} for {x.dtype}"
            )

    if z is not None:
        if z.stride(-1) != 1:
            raise ValueError("z must be contiguous along the last dimension")
        if z.dtype != x.dtype:
            raise ValueError(f"z dtype {z.dtype} must match x dtype {x.dtype}")
        if z.ndim == 3:
            if z.shape[0] * z.shape[1] != m or z.shape[2] != n:
                raise ValueError(
                    f"3-D z must be [T, H, {n}] with T*H == M == {m}, got {tuple(z.shape)}"
                )
        elif z.ndim == 2:
            if tuple(z.shape) != (m, n):
                raise ValueError(f"2-D z must be {(m, n)}, got {tuple(z.shape)}")
        else:
            raise ValueError(f"z must be 2-D or 3-D, got {z.ndim}-D")

    if quant_heads:
        if group_size is not None and group_size != n:
            raise ValueError("quant_heads requires a single group (group_size must be None or N)")
        if m % quant_heads:
            raise ValueError(f"quant_heads={quant_heads} must divide M={m}")
    if not isinstance(eps, (int, float)):
        raise ValueError(f"eps must be a scalar, got {type(eps).__name__}")
    _require_inference(op, x, weight)


def layernorm_gated_infer(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    *,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    batch_invariant: bool = False,
    implementation: Optional[str] = None,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], torch.Tensor]:
    """Fused gated norm over the last dimension of ``x``.

    Returns ``((q, s), mean, rstd)`` when ``quant_heads`` is set and ``(out, mean, rstd)``
    otherwise. ``mean`` is ``None`` for RMSNorm; both statistics are FP32 with the
    ``[ngroups * M]`` group-major layout, matching the reference implementation.

    ``batch_invariant=True`` pins the tile to a constant row count so the reduction order does
    not depend on ``M`` or the core count; that is required for batch-invariant serving and
    costs bandwidth, so it is off by default.
    """
    _validate(x, weight, bias, z, eps, group_size, activation, quant_heads, "layernorm_gated_infer")
    forward, _ = load_impl("layernorm_gated_infer", implementation)
    return forward(
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
        quant_heads=quant_heads,
        batch_invariant=batch_invariant,
    )
