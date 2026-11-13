"""Stateful wrapper for fused Q/K RMSNorm and packed RoPE."""

from typing import Optional

import torch

from mojo_opset import functions


class FusedQKNormRope(torch.nn.Module):
    def __init__(
        self,
        head_dim: int = 128,
        rotary_dim: int = 96,
        eps: float = 1e-5,
        *,
        interleaved: bool = False,
        cuda_rms_norm: bool = False,
        implementation: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and in [2, head_dim]")
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.eps = eps
        self.interleaved = interleaved
        self.cuda_rms_norm = cuda_rms_norm
        self.implementation = implementation
        self.q_weight = torch.nn.Parameter(torch.ones(head_dim, device=device, dtype=dtype))
        self.k_weight = torch.nn.Parameter(torch.ones(head_dim, device=device, dtype=dtype))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return functions.fused_qk_norm_rope(
            q,
            k,
            self.q_weight,
            self.k_weight,
            rope_table,
            self.eps,
            head_dim=self.head_dim,
            rotary_dim=self.rotary_dim,
            interleaved=self.interleaved,
            cuda_rms_norm=self.cuda_rms_norm,
            implementation=self.implementation,
        )


QKNormRope = FusedQKNormRope
