from __future__ import annotations

from typing import Optional

import torch

from mojo_opset.core import MojoCompressor
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class AscendcCompressor(MojoCompressor):
    supported_platforms_list = ["npu"]

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
        try:
            try:
                import custom_ops  # noqa: F401
            except Exception:
                pass

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
        except Exception:
            logger.warning("AscendC compressor kernel not available, falling back to reference implementation.")
            return super().forward(
                x,
                wkv,
                wgate,
                state_cache,
                ape,
                norm_weight,
                rope_sin,
                rope_cos,
                rope_head_dim,
                cmp_ratio,
                state_block_table=state_block_table,
                cu_seqlens=cu_seqlens,
                seqused=seqused,
                start_pos=start_pos,
                coff=coff,
                norm_eps=norm_eps,
                rotary_mode=rotary_mode,
                cache_mode=cache_mode,
            )

