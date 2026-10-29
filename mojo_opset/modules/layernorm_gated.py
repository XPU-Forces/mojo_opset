"""Stateful wrappers for fused gated normalization."""

from typing import Optional

import torch

from mojo_opset import functions


class LayerNormGated(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
        norm_before_gate: bool = True,
        activation: str = "swish",
        *,
        implementation: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.implementation = implementation
        self.weight = torch.nn.Parameter(
            torch.ones(normalized_shape, device=device, dtype=dtype)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(normalized_shape, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        result, mean, rstd = functions.layernorm_gated(
            x,
            self.weight,
            self.bias,
            self.eps,
            z=z,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
            implementation=self.implementation,
        )
        return result


class RMSNormGated(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
        norm_before_gate: bool = True,
        activation: str = "swish",
        *,
        implementation: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.implementation = implementation
        self.weight = torch.nn.Parameter(
            torch.ones(normalized_shape, device=device, dtype=dtype)
        )
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        result, _, _ = functions.layernorm_gated(
            x,
            self.weight,
            None,
            self.eps,
            z=z,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=True,
            activation=self.activation,
            implementation=self.implementation,
        )
        return result


__all__ = ["LayerNormGated", "RMSNormGated"]
