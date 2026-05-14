from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..operator import MojoOperator


class MojoCompressor(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        wkv: torch.Tensor,
        wgate: torch.Tensor,
        state_cache: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_head_dim: int,
        cmp_ratio: int,
        *,
        state_block_table: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        seqused: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        coff: int = 1,
        norm_eps: float = 1e-6,
        rotary_mode: int = 1,
        cache_mode: int = 1,
    ) -> torch.Tensor:
        """
        Compressor operator wrapper.

        This operator is provided by Ascend custom kernels as ``torch.ops.custom.compressor``.
        A pure PyTorch reference implementation is non-trivial and currently not provided.

        When the custom kernel is available, this method dispatches to it; otherwise it raises
        ``NotImplementedError`` so tests can be skipped via ``@bypass_not_implemented``.
        """
        if hasattr(torch.ops, "custom") and hasattr(torch.ops.custom, "compressor"):
            return torch.ops.custom.compressor(
                x,
                wkv,
                wgate,
                state_cache,
                ape,
                norm_weight,
                rope_sin,
                rope_cos,
                rope_head_dim=rope_head_dim,
                cmp_ratio=cmp_ratio,
                state_block_table=state_block_table,
                cu_seqlens=cu_seqlens,
                seqused=seqused,
                start_pos=start_pos,
                coff=coff,
                norm_eps=norm_eps,
                rotary_mode=rotary_mode,
                cache_mode=cache_mode,
            )

        raise NotImplementedError("MojoCompressor reference implementation is not available (custom kernel missing).")

