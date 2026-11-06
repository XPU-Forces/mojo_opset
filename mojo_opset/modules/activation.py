from typing import Optional

import torch

from mojo_opset import functions


class SiLU(torch.nn.Module):
    def __init__(self, *, implementation: Optional[str] = None):
        super().__init__()
        self.implementation = implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functions.silu(x, implementation=self.implementation)


class GELU(torch.nn.Module):
    def __init__(self, approximate: str = "tanh", *, implementation: Optional[str] = None):
        super().__init__()
        self.approximate = approximate
        self.implementation = implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functions.gelu(x, approximate=self.approximate, implementation=self.implementation)


class SwiGLU(torch.nn.Module):
    def __init__(self, swiglu_limit: float = 0.0, *, implementation: Optional[str] = None):
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.implementation = implementation

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return functions.swiglu(
            x1,
            x2,
            scales,
            swiglu_limit=self.swiglu_limit,
            implementation=self.implementation,
        )


class SituAndMul(torch.nn.Module):
    """Kimi-K3 SiTU gated activation on concatenated [gate | up] inputs."""

    def __init__(
        self,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
        *,
        implementation: Optional[str] = None,
    ):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta
        self.implementation = implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functions.situ_and_mul(
            x,
            beta=self.beta,
            linear_beta=self.linear_beta,
            implementation=self.implementation,
        )
