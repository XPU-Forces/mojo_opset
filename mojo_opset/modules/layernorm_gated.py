"""Parameterized gated normalization modules."""

from typing import Optional

import torch

from mojo_opset import functions


class _GatedNormBase(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: Optional[int] = None,
        norm_before_gate: bool = True,
        activation: str = "swish",
        *,
        is_rms_norm: bool = False,
        elementwise_affine: bool = True,
        has_bias: bool = True,
        implementation: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.is_rms_norm = is_rms_norm
        self.elementwise_affine = elementwise_affine
        self.implementation = implementation
        self.weight = (
            torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype)) if elementwise_affine else None
        )
        self.bias = (
            torch.nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
            if elementwise_affine and has_bias
            else None
        )

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        return functions.rms_norm_gated(
            x=x,
            weight=self.weight,
            bias=self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=self.is_rms_norm,
            activation=self.activation,
            implementation=self.implementation,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, group_size={self.group_size}, "
            f"norm_before_gate={self.norm_before_gate}, activation={self.activation}"
        )


class RMSNormGated(_GatedNormBase):
    """``norm(x) * act(z)`` with an RMS denominator; the GDN out-norm shape."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, **kwargs):
        kwargs.setdefault("is_rms_norm", True)
        kwargs.setdefault("has_bias", False)
        super().__init__(hidden_size, eps, **kwargs)


class LayerNormGated(_GatedNormBase):
    """LayerNorm (mean + variance) with the same gate semantics."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, **kwargs):
        kwargs.setdefault("is_rms_norm", False)
        super().__init__(hidden_size, eps, **kwargs)
