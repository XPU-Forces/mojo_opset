import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..operator import MojoOperator


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


class MojoEmbedding(MojoOperator):
    """Standard embedding lookup (drop-in replacement for ``torch.nn.Embedding``)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Integer indices of shape ``(*)``.

        Returns:
            torch.Tensor: ``(*, embedding_dim)``.
        """
        return F.embedding(
            input,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}, norm_type={self.norm_type}"
        return s


class MojoParallelEmbedding(MojoOperator):
    """Vocabulary-parallel embedding.

    The embedding table is sharded along the ``num_embeddings`` (vocab)
    dimension.  Each rank stores ``ceil(num_embeddings / world_size)`` rows.
    Indices outside the local shard produce zero vectors; an ``all_reduce``
    (sum) across ranks assembles the final result.

    When ``torch.distributed`` is not initialised the operator behaves
    identically to :class:`MojoEmbedding`.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        process_group: Optional[dist.ProcessGroup] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.process_group = process_group

        if _is_dist_initialized():
            world_size = dist.get_world_size(group=process_group)
            rank = dist.get_rank(group=process_group)
        else:
            world_size = 1
            rank = 0

        # Divide vocab evenly (last rank may own fewer rows)
        local_size = math.ceil(num_embeddings / world_size)
        self.vocab_start_index = rank * local_size
        self.vocab_end_index = min(self.vocab_start_index + local_size, num_embeddings)
        self.local_num_embeddings = self.vocab_end_index - self.vocab_start_index

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(self.local_num_embeddings, embedding_dim, **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            local_pad = self.padding_idx - self.vocab_start_index
            if 0 <= local_pad < self.local_num_embeddings:
                with torch.no_grad():
                    self.weight[local_pad].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Integer indices of shape ``(*)``
                in ``[0, num_embeddings)``.

        Returns:
            torch.Tensor: ``(*, embedding_dim)``.
        """
        # Shift indices into the local range
        local_input = input - self.vocab_start_index

        # Mask out-of-range indices → look up row 0 (will be zeroed later)
        in_range = (local_input >= 0) & (local_input < self.local_num_embeddings)
        masked_input = local_input.clamp(0, self.local_num_embeddings - 1)

        output = F.embedding(
            masked_input,
            self.weight,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )

        # Zero contributions from out-of-range indices
        output = output * in_range.unsqueeze(-1)

        if _is_dist_initialized():
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        return output

    def extra_repr(self) -> str:
        s = (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"local_range=[{self.vocab_start_index}, {self.vocab_end_index})"
        )
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s


class MojoRelativeEmbedding(MojoOperator):
    def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128):
        """
        Initialize T5-style relative position embedding.

        Args:
            num_buckets (int): Number of relative position buckets.
            num_heads (int): Attention heads; also the embedding output channels.
            bidirectional (bool): If True, allocate half buckets for positive direction.
            max_dist (int, default=128): Maximum distance used in logarithmic bucketing.
        """
        super().__init__()
        if not isinstance(num_buckets, int) or num_buckets <= 0:
            raise ValueError("num_buckets must be a positive integer")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(bidirectional, bool):
            raise TypeError("bidirectional must be a bool")
        if not isinstance(max_dist, int) or max_dist <= 0:
            raise ValueError("max_dist must be a positive integer")
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = torch.nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        """
        Compute relative position bias tensor for attention.

        Args:
            lq (int): Length of query sequence (Lq).
            lk (int): Length of key/value sequence (Lk).

        Returns:
            torch.Tensor: Bias tensor of shape [1, num_heads, Lq, Lk], dtype follows embedding weights.
        """
        if not isinstance(lq, int) or not isinstance(lk, int) or lq <= 0 or lk <= 0:
            raise ValueError("lq and lk must be positive integers")
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets

    def extra_repr(self) -> str:
        return f"{self.num_buckets=}, {self.num_heads=}, {self.bidirectional=}, {self.max_dist=}".replace("self.", "")
