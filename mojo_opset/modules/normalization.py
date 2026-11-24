from typing import Optional

import torch

from mojo_opset import functions


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        *,
        implementation: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.implementation = implementation
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functions.rms_norm(
            x,
            self.weight,
            self.eps,
            implementation=self.implementation,
        )

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


class LayerNormInfer(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True,
                 *, implementation: Optional[str] = None, device=None, dtype=None):
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.implementation = implementation
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype)) if elementwise_affine else None
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError("input last dimension must match normalized_shape")
        return functions.layer_norm_infer(x, self.weight, self.bias, self.eps, implementation=self.implementation)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class RMSNormInfer(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, *,
                 implementation: Optional[str] = None, device=None, dtype=None):
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.implementation = implementation
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functions.rms_norm_infer(x, self.weight, self.eps, implementation=self.implementation)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


class GroupRMSNormInfer(torch.nn.Module):
    def __init__(self, num_groups: int, normalized_shape: int, eps: float = 1e-5,
                 elementwise_affine: bool = True, *, implementation: Optional[str] = None,
                 device=None, dtype=None):
        super().__init__()
        if num_groups <= 0 or normalized_shape <= 0:
            raise ValueError("num_groups and normalized_shape must be positive")
        self.num_groups = num_groups
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.implementation = implementation
        self.weight = torch.nn.Parameter(torch.ones(num_groups, normalized_shape, device=device, dtype=dtype)) if elementwise_affine else None

    def forward(self, input_groups: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(input_groups) != self.num_groups or any(x.shape[-1] != self.normalized_shape for x in input_groups):
            raise ValueError("input group count and last dimension must match the module")
        return functions.group_rms_norm_infer(input_groups, self.weight, self.eps, implementation=self.implementation)

    def extra_repr(self):
        return (f"num_groups={self.num_groups}, normalized_shape={self.normalized_shape}, "
                f"eps={self.eps}, elementwise_affine={self.elementwise_affine}")


class ResidualAddLayerNormInfer(LayerNormInfer):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, norm_pos: str = "pre", *,
                 implementation: Optional[str] = None, device=None, dtype=None):
        super().__init__(normalized_shape, eps, implementation=implementation, device=device, dtype=dtype)
        if norm_pos not in ("pre", "post"):
            raise ValueError("norm_pos must be 'pre' or 'post'")
        self.norm_pos = norm_pos

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return functions.residual_add_layer_norm_infer(
            x, residual, self.weight, self.bias, self.eps,
            norm_pos=self.norm_pos, implementation=self.implementation
        )

    def extra_repr(self):
        return super().extra_repr() + f", norm_pos={self.norm_pos!r}"


class ResidualAddRMSNormInfer(RMSNormInfer):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, norm_pos: str = "pre", *,
                 implementation: Optional[str] = None, device=None, dtype=None):
        super().__init__(normalized_shape, eps, implementation=implementation, device=device, dtype=dtype)
        if norm_pos not in ("pre", "post"):
            raise ValueError("norm_pos must be 'pre' or 'post'")
        self.norm_pos = norm_pos

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return functions.residual_add_rms_norm_infer(
            x, residual, self.weight, self.eps, norm_pos=self.norm_pos, implementation=self.implementation
        )

    def extra_repr(self):
        return super().extra_repr() + f", norm_pos={self.norm_pos!r}"
