"""Ascend A5 wrapper for fused gated normalization.

A5 adds the three-axis gate tile: long strided 3D gates move whole head rows
instead of one scattered row segment per row.

Provider layout follows the public implementation of the same community task,
``brandfucker/mojo_opset#516`` (Apache-2.0).
"""

from mojo_opset.kernels._layernorm_gated import _layer_norm_fwd_impl


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
        use_z3d_tile=True,
    )
