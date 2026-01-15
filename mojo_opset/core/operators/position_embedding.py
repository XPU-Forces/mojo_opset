from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


class MojoRoPE(MojoOperator):
    def __init__(
        self,
        rotary_offset: int = 0,
        interleaved: bool = False,
        dynamic_ntk: bool = False,
        max_seq_len: Optional[int] = None,
        is_varlen: bool = True,
    ):
        """
        Initialize rotary position embedding configuration.

        Args:
            rotary_offset (int, default=0): Non-negative rotation offset applied to positions.
            interleaved (bool, default=False): If True, use interleaved head layout when applying rotary.
            dynamic_ntk (bool, default=False): Enable dynamic NTK scaling to extend effective context length.
            max_seq_len (Optional[int], default=None): Positive max sequence length for precomputation; or None.
            is_varlen (bool, default=True): Prefer token-first TND layout when True; else use BSND.

        Raises:
            ValueError: If `rotary_offset` < 0 or `max_seq_len` <= 0 when provided.
            TypeError: If `interleaved`, `dynamic_ntk`, or `is_varlen` are not bools.

        Notes:
            This initializer performs light validation and stores configuration flags; the
            actual rotary application happens in the forward path.
        """
        super().__init__()

        if not isinstance(rotary_offset, int) or rotary_offset < 0:
            raise ValueError("rotary_offset should be non-negative integer.")
        if not isinstance(interleaved, bool):
            raise TypeError("interleaved should be bool.")
        if not isinstance(dynamic_ntk, bool):
            raise TypeError("dynamic_ntk should be bool.")
        if max_seq_len is not None and (not isinstance(max_seq_len, int) or max_seq_len <= 0):
            raise ValueError("max_seq_len should be positive integer or None.")
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen should be bool.")

        self.rotary_offset = rotary_offset
        self.interleaved = interleaved
        self.dynamic_ntk = dynamic_ntk
        self.max_seq_len = max_seq_len
        self.is_varlen = is_varlen

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cum_sum_query_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings (RoPE) to queries and keys.

        Args:
            q (torch.Tensor): Query tensor; last dimension must be even to allow rotation.
            k (torch.Tensor): Key tensor; same shape as `q`.
            cos (torch.Tensor): Precomputed cosine tensor, broadcastable to `q`/`k`.
            sin (torch.Tensor): Precomputed sine tensor, broadcastable to `q`/`k`.
            position_ids (Optional[torch.Tensor], default=None): Reserved; not used here.
            cum_sum_query_len (Optional[torch.Tensor], default=None): Reserved; not used here.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(q_rot, k_rot)` with the same shape/dtype as inputs.

        Notes:
            Uses standard RoPE: `x_rot = x * cos + rotate_half(x) * sin`, where `rotate_half`
            swaps the two halves of the last dimension with a sign flip.
        """

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
