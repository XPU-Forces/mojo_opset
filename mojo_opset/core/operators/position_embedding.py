from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


class MojoRoPE(MojoOperator):
    def __init__(
        self,
        interleaved: bool = False,
    ):
        """
        Args:
            interleaved (bool, default=False): If True, use interleaved head layout when applying rotary.

        """
        super().__init__()

        assert interleaved == False, "interleaved impl is not supported yet."
        self.interleaved = interleaved

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings (RoPE) to queries and keys.

        Args:
            q (torch.Tensor): Query tensor; last dimension must be even to allow rotation.
            k (torch.Tensor): Key tensor; same shape as `q`.
            cos (torch.Tensor): Precomputed cosine tensor, broadcastable to `q`/`k`.
            sin (torch.Tensor): Precomputed sine tensor, broadcastable to `q`/`k`.
            cu_seqlens (Optional[torch.Tensor], default=None): Reserved; not used here.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(q_rot, k_rot)` with the same shape/dtype as inputs.
        """

        assert cu_seqlens is None, "cu_seqlens is not supported yet."

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot


class MojoRoPEStoreKV(MojoOperator):
    pass


class MojoNormRoPE(MojoOperator):
    pass


class MojoNormRoPEStoreKV(MojoOperator):
    pass
