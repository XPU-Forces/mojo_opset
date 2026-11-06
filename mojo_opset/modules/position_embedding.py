import torch

from mojo_opset import functions


class PosEmbedInterp(torch.nn.Module):
    """Bilinear interpolation of a learned ``G x G`` position table onto a ``(t, h, w)`` grid.

    Mirrors the vision-encoder usage: the table is a buffer of the module (or passed per call) and
    the output is ``[t*h*w, hidden]`` in spatial-merge order, i.e. the ``m_size x m_size`` neighbour
    block of each merge unit is written adjacently. ``grids`` switches to the merged single-launch
    path for a list of grids, which is bit-identical to per-grid calls concatenated along dim 0.
    """

    def __init__(self, embed_weight=None, num_grid_per_side=48, m_size=2, *, implementation=None):
        super().__init__()
        if num_grid_per_side < 2:
            raise ValueError(f"num_grid_per_side={num_grid_per_side} must be >= 2")
        if m_size <= 0:
            raise ValueError(f"m_size={m_size} must be positive")
        self.num_grid_per_side = num_grid_per_side
        self.m_size = m_size
        self.implementation = implementation
        if embed_weight is None:
            self.register_buffer("embed_weight", None, persistent=False)
        else:
            self.register_buffer("embed_weight", embed_weight, persistent=False)

    def forward(self, t=None, h=None, w=None, *, grids=None, embed_weight=None, dtype=None):
        weight = embed_weight if embed_weight is not None else self.embed_weight
        if weight is None:
            raise ValueError("PosEmbedInterp requires embed_weight either as a buffer or per call")
        if grids is not None:
            if t is not None or h is not None or w is not None:
                raise ValueError("pass either grids or (t, h, w), not both")
            return functions.pos_embed_interp_multi_infer(
                weight,
                grids,
                self.num_grid_per_side,
                self.m_size,
                dtype,
                implementation=self.implementation,
            )
        if t is None or h is None or w is None:
            raise ValueError("PosEmbedInterp requires (t, h, w) or grids")
        return functions.pos_embed_interp_infer(
            weight,
            t,
            h,
            w,
            self.num_grid_per_side,
            self.m_size,
            dtype,
            implementation=self.implementation,
        )
