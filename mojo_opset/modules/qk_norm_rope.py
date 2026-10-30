"""Parameterised QK RMSNorm + RoPE block."""

import torch
import torch.nn as nn

from mojo_opset.functions.qk_norm_rope import qk_norm_rope_infer


class QkNormRope(nn.Module):
    """Holds the two per-head RMSNorm weights and applies the fused QK norm + RoPE.

    ``head_dim`` and ``rotary_dim`` are stored so the module can build a packed
    ``[cos | sin]`` table from a ``[tokens, rotary_dim // 2]`` theta, which is how the
    MiniMax-H3 transformer feeds it.

    Forward only, the same convention as ``RMSNorm``: ``q_weight``/``k_weight`` are
    ``nn.Parameter``s, and the shared inference guard rejects them whenever grad mode is on,
    so ``forward`` has to be called inside ``torch.no_grad()`` or an inference mode.
    """

    def __init__(
        self,
        head_dim: int = 128,
        rotary_dim: int = 96,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.rotary_dim = int(rotary_dim)
        self.eps = float(eps)
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        # ``rotary_dim`` has to be even in both directions: ``rope_table`` packs
        # ``rotary_dim // 2`` angles into ``[cos | sin]``, so an odd width would build a table
        # that is one element shorter than the width the module advertises, and the geometry
        # cannot exceed ``head_dim`` either.
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError(f"rotary_dim must be a positive even number, got {rotary_dim}")
        if self.rotary_dim > self.head_dim:
            raise ValueError(
                f"rotary_dim={self.rotary_dim} must not exceed head_dim={self.head_dim}"
            )
        # The weights follow the activation dtype (BF16 on the MiniMax-H3 contract). Leaving
        # them float32 makes F.rms_norm fall back off its fused kernel, which changes the
        # rounding of the yardstick it is compared against.
        self.q_weight = nn.Parameter(torch.ones(self.head_dim, dtype=dtype))
        self.k_weight = nn.Parameter(torch.ones(self.head_dim, dtype=dtype))

    def rope_table(self, theta: torch.Tensor) -> torch.Tensor:
        """Pack ``theta`` of width ``rotary_dim // 2`` into ``[cos | sin]``."""
        half = self.rotary_dim // 2
        if theta.shape[-1] != half:
            raise ValueError(f"theta must be [tokens, {half}], got {tuple(theta.shape)}")
        return torch.cat((theta.cos(), theta.sin()), dim=-1).to(theta.dtype)

    def forward(self, q, k, rope_table, *, implementation=None):
        return qk_norm_rope_infer(
            q,
            k,
            self.q_weight,
            self.k_weight,
            rope_table,
            self.eps,
            implementation=implementation,
        )

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, rotary_dim={self.rotary_dim}, eps={self.eps}, "
            f"dtype={self.q_weight.dtype}"
        )
