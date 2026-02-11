from typing import Optional
from typing import Tuple, List

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


class MojoGridRoPE(MojoOperator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply 3D grid rotary position embeddings (RoPE) over (F, H, W) axes using
        precomputed per-sample frequency tensors.

        Args:
            x (torch.Tensor): [B, L, N, D]; D must be even (paired into complex components).
            grid_sizes (torch.Tensor): [B, 3] per-sample (F, H, W); seq_len = F*H*W.
            freqs_list (List[torch.Tensor]): length-B list; each item is a complex unit-phase tensor
                of shape [seq_len, 1, D/2], broadcastable to [seq_len, N, D/2].

        Returns:
            torch.Tensor: Same shape as `x`. Per sample, the first F*H*W tokens are rotated;
                remaining padding tokens are preserved. Output dtype matches input.
        """
        assert x.dim() == 4, "x must be 4D: [B, L, N, D]"
        assert x.size(-1) % 2 == 0, "D must be even for complex pairing"
        assert grid_sizes.dim() == 2 and grid_sizes.size(1) == 3, "grid_sizes must be [B, 3]"

        n = x.size(2)
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
            freqs_i = freqs_list[i]
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        y = torch.stack(output)
        return y.type_as(x)
