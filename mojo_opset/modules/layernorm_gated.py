"""Parameterised gated norm block (LayerNorm or RMSNorm, optionally gated)."""

from typing import Optional

import torch
import torch.nn as nn

from mojo_opset.functions.layernorm_gated import layernorm_gated_infer


class LayerNormGated(nn.Module):
    """Gated LayerNorm / RMSNorm mirroring the FLA ``RMSNorm`` / ``LayerNorm`` blocks.

    ``group_size`` selects group norm (``None`` means one group covering the whole row),
    ``is_rms_norm`` drops the mean subtraction, and ``activation`` picks the gate nonlinearity.
    The gate itself is an argument to :meth:`forward`, as upstream, because it is produced by a
    different projection in the model.

    The bias parameter is registered only when ``has_bias`` is set, so a RMSNorm block exposes
    the same ``state_dict`` shape as the reference implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: Optional[int] = None,
        norm_before_gate: bool = True,
        is_rms_norm: bool = False,
        activation: str = "swish",
        has_bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.is_rms_norm = is_rms_norm
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(self.hidden_size, device=device, dtype=dtype))
        if has_bias:
            self.bias = nn.Parameter(torch.zeros(self.hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, z=None, *, out=None, implementation=None):
        result, _mean, _rstd = layernorm_gated_infer(
            x,
            self.weight,
            self.bias,
            self.eps,
            z=z,
            out=out,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=self.is_rms_norm,
            activation=self.activation,
            implementation=implementation,
        )
        return result

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, group_size={self.group_size}, "
            f"is_rms_norm={self.is_rms_norm}, norm_before_gate={self.norm_before_gate}, "
            f"activation={self.activation}, bias={self.bias is not None}"
        )
