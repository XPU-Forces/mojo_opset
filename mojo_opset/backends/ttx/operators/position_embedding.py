from typing import Optional
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import rot_pos_embed
from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.core import MojoRotaryEmbedding
from mojo_opset.core import MojoRoPE


class TTXRotaryEmbedding(MojoRotaryEmbedding):
    supported_platforms_list = ["npu"]

    def __init__(self, rope_theta, rope_dim, init_max_length: Optional[int] = None, **kwargs):
        super().__init__(rope_theta, rope_dim, init_max_length, **kwargs)
        if init_max_length is None:
            raise ValueError("init_max_length must be provided for TTXRotaryEmbedding")

    def forward(
        self,
        *,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        seqlens_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert cu_seqlens_q is None or position_ids is None, (
            "Exactly one of cu_seqlens_q or position_ids should be provided"
        )

        return rot_pos_embed(
            self.cos,
            self.sin,
            cu_seqlens=cu_seqlens_q,
            seqlens_kv=seqlens_kv,
            position_ids=position_ids,
        )


class TTXRoPE(MojoRoPE):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rope_fwd(q, k, cos, sin, head_first)
