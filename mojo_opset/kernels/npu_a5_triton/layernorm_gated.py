"""Triton-Ascend A5 binding for fused gated LayerNorm/RMSNorm."""

from mojo_opset.kernels._layernorm_gated_common import MAX_ROWS_PER_BLOCK
from mojo_opset.kernels._layernorm_gated_common import _layer_norm_fwd_1pass_kernel
from mojo_opset.kernels._layernorm_gated_common import _layer_norm_fwd_impl
from mojo_opset.kernels._layernorm_gated_common import calc_rows_per_block


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
    return _layer_norm_fwd_impl(
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
    )


def layernorm_gated_fwd(
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
    return _layer_norm_fwd(
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
    )


__all__ = [
    "MAX_ROWS_PER_BLOCK",
    "_layer_norm_fwd",
    "_layer_norm_fwd_1pass_kernel",
    "calc_rows_per_block",
    "layernorm_gated_fwd",
]
