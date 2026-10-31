"""Ascend A2 wrapper for fused gated normalization.

Provider layout follows the public implementation of the same community task,
``brandfucker/mojo_opset#516`` (Apache-2.0).
"""

from mojo_opset.kernels._layernorm_gated import _layer_norm_fwd_impl


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
    if quant_heads:
        # The 910B/910C runtime does not implement DT_FLOAT8_E4M3FN, so the FP8
        # epilogue cannot be built or run here; the toolkit aborts inside the
        # compiler without saying why.  Ascend 950 carries the capability.
        raise RuntimeError(
            "the fused FP8 epilogue needs a runtime that implements DT_FLOAT8_E4M3FN; "
            "the Ascend 910B/910C runtime does not, so the FP8 output cannot be produced "
            "on this target.  Ascend 950 (npu.a5) is the target that supports it."
        )
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


# The same wrapper under the alias spelling of the dispatch id.
layernorm_gated_fwd = layer_norm_gated_fwd
