"""Alias spellings of the gated-normalization entry points.

:func:`mojo_opset.functions.layer_norm_gated` is the canonical front end; the
names here keep callers written against the earlier native API working:
``layernorm_gated`` places the epsilon in fourth position, ``rms_norm_gated``
mirrors the SGLang call that returns only the normalized output, and
``_layer_norm_fwd`` is the SGLang-compatible 2-D entry point.

They live in their own module so that the canonical front end can be replaced
by the one of the community acceptance suite
(``XPU-Forces/mojo_opset#525``) without breaking the package import.
"""

from typing import Optional

import torch

from .layernorm_gated import layer_norm_gated


def layernorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    *,
    implementation: Optional[str] = None,
):
    """Same operator as :func:`layer_norm_gated`, with the epsilon in fourth position."""

    return layer_norm_gated(
        x,
        weight,
        bias,
        z=z,
        eps=eps,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=quant_heads,
        implementation=implementation,
    )


def rms_norm_gated(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    implementation: Optional[str] = None,
):
    """SGLang-compatible entry point that returns only the normalized output."""

    result, _, _ = layer_norm_gated(
        x,
        weight,
        bias,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        implementation=implementation,
    )
    return result


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
    quant_heads: int = 0,
):
    """SGLang-compatible 2-D entry point dispatched through the target config."""

    if x.ndim != 2:
        raise ValueError("_layer_norm_fwd expects a 2-D x")
    return layer_norm_gated(
        x,
        weight,
        bias,
        z=z,
        eps=eps,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=quant_heads,
    )


__all__ = ["_layer_norm_fwd", "layernorm_gated", "rms_norm_gated"]
