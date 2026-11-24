from typing import Optional

import torch

from mojo_opset import functions


class RoPE(torch.nn.Module):
    """Apply rotary position embedding to Q/K using caller-supplied tables."""

    def __init__(
        self,
        *,
        unsqueeze_dim: int = 1,
        implementation: Optional[str] = None,
    ):
        super().__init__()
        self.unsqueeze_dim = unsqueeze_dim
        self.implementation = implementation

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return functions.rope(
            q,
            k,
            cos,
            sin,
            unsqueeze_dim=self.unsqueeze_dim,
            implementation=self.implementation,
        )
