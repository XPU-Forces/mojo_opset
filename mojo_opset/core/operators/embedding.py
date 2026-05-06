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
