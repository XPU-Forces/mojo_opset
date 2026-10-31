"""Gated normalization layers built on :func:`mojo_opset.functions.layer_norm_gated`."""

from typing import Optional

import torch

from mojo_opset import functions


class _GatedNorm(torch.nn.Module):
    """Parameters shared by the gated LayerNorm and RMSNorm layers."""

    is_rms_norm = False
    default_eps = 1e-5
    default_bias = True

    def __init__(
        self,
        hidden_size: int,
        eps: Optional[float] = None,
        group_size: Optional[int] = None,
        norm_before_gate: bool = True,
        bias: Optional[bool] = None,
        activation: str = "swish",
        *,
        implementation: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if group_size is not None and (group_size <= 0 or hidden_size % group_size):
            raise ValueError("group_size must be positive and divide hidden_size")
        self.hidden_size = hidden_size
        self.eps = self.default_eps if eps is None else eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.implementation = implementation
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        has_bias = self.default_bias if bias is None else bias
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype)) if has_bias else None

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None, out=None) -> torch.Tensor:
        result, _, _ = functions.layer_norm_gated(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            out=out,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=self.is_rms_norm,
            activation=self.activation,
            implementation=self.implementation,
        )
        return result

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, group_size={self.group_size}, "
            f"norm_before_gate={self.norm_before_gate}, activation={self.activation}, "
            f"bias={self.bias is not None}"
        )


class LayerNormGated(_GatedNorm):
    """LayerNorm, optionally over ``group_size`` groups and followed/preceded by a gate."""


class RMSNormGated(_GatedNorm):
    """RMSNorm variant used by GDN and Mixer2RMSNormGated-style blocks."""

    is_rms_norm = True
    default_eps = 1e-6
    default_bias = False


__all__ = ["LayerNormGated", "RMSNormGated"]
