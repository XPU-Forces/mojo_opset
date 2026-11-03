"""Stateful wrapper for the ``vision_pos_embed_interp`` function.

The functions layer is stateless: every call takes ``num_grid_per_side``,
``m_size`` and ``dtype``. A vision encoder reuses those values on every forward,
so this module holds them as members and exposes only the data-dependent
arguments.

Placement follows the task description (``mojo_opset/modules/``) and the upstream
module convention -- a thin ``torch.nn.Module`` built directly on ``functions``,
mirroring ``multimodal_rope_infer.py`` and ``rope.py``.
"""

import torch

from mojo_opset import functions


class VisionPosEmbedInterp(torch.nn.Module):
    """Bilinear position-embedding interpolation with the grid config fixed.

    Args:
        num_grid_per_side: side length ``G`` of the learned position table.
        m_size: spatial-merge side length; ``h`` and ``w`` must be divisible by it.
        dtype: output dtype (``torch.bfloat16`` / ``torch.float32``).
        implementation: optional provider override ("torch" / "triton"); when
            ``None`` the configured provider for the current target is used.

    Example:
        >>> m = VisionPosEmbedInterp(48, 2, torch.bfloat16)
        >>> out = m(embed_weight, 1, 16, 16)                    # one grid
        >>> out = m.forward_multi(embed_weight, [(1, 8, 16)])   # several grids
    """

    def __init__(
        self,
        num_grid_per_side: int,
        m_size: int,
        dtype: torch.dtype = torch.bfloat16,
        *,
        implementation: str = None,
    ):
        super().__init__()
        if num_grid_per_side < 2:
            raise ValueError(
                f"num_grid_per_side must be >= 2, got {num_grid_per_side}"
            )
        if m_size < 1:
            raise ValueError(f"m_size must be >= 1, got {m_size}")
        self.num_grid_per_side = num_grid_per_side
        self.m_size = m_size
        self.dtype = dtype
        self.implementation = implementation

    def forward(self, embed_weight: torch.Tensor, t: int, h: int, w: int):
        """Interpolate a single ``(t, h, w)`` grid.

        Returns:
            ``[t*h*w, hidden]`` in spatial-merge order.
        """
        return functions.vision_pos_embed_interp(
            embed_weight,
            int(t),
            int(h),
            int(w),
            self.num_grid_per_side,
            self.m_size,
            self.dtype,
            implementation=self.implementation,
        )

    def forward_multi(self, embed_weight: torch.Tensor, grid_thw):
        """Interpolate several grids and concatenate the results in order.

        Composed from the same single-grid entry the upstream dispatch exposes
        (``functions.vision_pos_embed_interp``), because the merged multi-grid
        entrypoint is not part of the public function surface. The concatenation
        is therefore host-side, matching how the acceptance suite measures the
        multi-image case.

        Args:
            embed_weight: ``[num_grid_per_side**2, hidden]`` position table.
            grid_thw: sequence of ``(t, h, w)`` triples.

        Returns:
            ``[sum(t*h*w), hidden]``, bit-identical to per-grid calls plus ``cat``.
        """
        return torch.cat([self.forward(embed_weight, *g) for g in grid_thw], dim=0)
