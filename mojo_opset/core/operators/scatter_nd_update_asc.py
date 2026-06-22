from __future__ import annotations

import torch

from ..operator import MojoOperator


class MojoScatterNdUpdateAsc(MojoOperator):
    def forward(
        self,
        var: torch.Tensor,
        indices: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reference implementation for ``torch.ops.custom.scatter_nd_update_asc``.

        Custom op signature:
            scatter_nd_update_asc(Tensor(a!) var, Tensor indices, Tensor update) -> ()

        Behavior:
            In-place update: for each i, if indices[i, 0] >= 0, then
            var[indices[i, 0], :] = update[i, :].

        Returns:
            The updated ``var`` tensor (same object).
        """
        if var.dim() != 2:
            raise ValueError(f"var must be 2D [b, s], got shape {tuple(var.shape)}")
        if indices.dim() != 2 or indices.size(-1) != 1:
            raise ValueError(f"indices must be 2D [u, 1], got shape {tuple(indices.shape)}")
        if update.dim() != 2:
            raise ValueError(f"update must be 2D [u, s], got shape {tuple(update.shape)}")
        if update.size(0) != indices.size(0):
            raise ValueError(
                f"update and indices must have same first dim, got {update.size(0)} vs {indices.size(0)}"
            )
        if update.size(1) != var.size(1):
            raise ValueError(f"update.size(1) must equal var.size(1), got {update.size(1)} vs {var.size(1)}")

        # Match doc: negative indices are skipped (no update).
        idx = indices.to(dtype=torch.int64).view(-1)
        for i in range(idx.numel()):
            j = int(idx[i].item())
            if j >= 0:
                var[j, :] = update[i, :]
        return var

