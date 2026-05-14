from __future__ import annotations

from typing import List
from typing import Optional

import torch

from ..operator import MojoOperator


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    stacked = torch.stack((-x_odd, x_even), dim=-1)
    return stacked.reshape(x.shape)


class MojoInplacePartialRotaryMul(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        *,
        rotary_mode: str = "interleave",
        partial_slice: List[int] = None,
    ) -> torch.Tensor:
        """
        Reference implementation for ``torch.ops.custom.inplace_partial_rotary_mul``.

        Signature (custom op):
            inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, *, str rotary_mode, int[2] partial_slice) -> ()

        Behavior:
            Applies rotary-mul to ``x[..., partial_slice[0]:partial_slice[1]]`` in-place and returns ``x``.
            Only ``rotary_mode='interleave'`` is supported (matches upstream docs/tests).
        """
        if partial_slice is None or len(partial_slice) != 2:
            raise ValueError(f"partial_slice must be a length-2 list, got {partial_slice!r}.")
        if rotary_mode != "interleave":
            raise NotImplementedError(f"Only rotary_mode='interleave' is supported, got {rotary_mode!r}.")

        start, end = int(partial_slice[0]), int(partial_slice[1])
        if start < 0 or end < 0 or start > end or end > x.size(-1):
            raise ValueError(
                f"Invalid partial_slice {partial_slice}: expected 0 <= start <= end <= x.size(-1)={x.size(-1)}."
            )
        if (end - start) % 2 != 0:
            raise ValueError(f"partial slice length must be even, got length {end - start}.")

        # Slice view.
        x_rope = x[..., start:end]
        # interleave rotary: r1 * x + rotate_every_two(x) * r2
        out = r1 * x_rope + _rotate_every_two(x_rope) * r2
        x[..., start:end] = out
        return x

