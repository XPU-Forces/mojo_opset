from typing import List
from typing import Optional
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import relative_embedding_fwd_impl
from mojo_opset.backends.ttx.kernels import rot_pos_embed
from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.backends.ttx.kernels import mrope_fwd_impl
from mojo_opset.backends.ttx.kernels import vision_rope_apply
from mojo_opset.backends.ttx.kernels import vision_rot_pos_embed
from mojo_opset.core import MojoApplyRoPE
from mojo_opset.core import MojoApplyVisionRoPE2D
from mojo_opset.core import MojoRelativeEmbedding
from mojo_opset.core import MojoRotaryEmbedding
from mojo_opset.core import MojoVisionRotaryEmbedding2D
from mojo_opset.core.operators.position_embedding import MojoMRoPE


class TTXRelativeEmbedding(MojoRelativeEmbedding):
    supported_platforms_list = ["ilu"]

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        if not isinstance(lq, int) or not isinstance(lk, int) or lq <= 0 or lk <= 0:
            raise ValueError("lq and lk must be positive integers")
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        bucket = self._relative_position_bucket(rel_pos)
        return relative_embedding_fwd_impl(bucket, self.embedding.weight)


class TTXRotaryEmbedding(MojoRotaryEmbedding):
    supported_platforms_list = ["npu", "ilu"]

    def __init__(self, rope_theta, rope_dim, attention_scaling: float = 1.0, init_max_length: Optional[int] = None, **kwargs):
        super().__init__(rope_theta, rope_dim, attention_scaling, init_max_length, **kwargs)
        if init_max_length is None:
            raise ValueError("init_max_length must be provided for TTXRotaryEmbedding")

    def forward(
        self,
        x: torch.Tensor,
        cu_q_lens: Optional[torch.Tensor] = None,
        total_seq_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if cu_q_lens is not None:
            assert cu_q_lens.dtype == torch.int32
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        if position_ids is not None:
            assert position_ids.dtype == torch.int32
        return rot_pos_embed(
            x,
            self.cos,
            self.sin,
            cu_q_lens=cu_q_lens,
            seqlens_kv=total_seq_lens,
            position_ids=position_ids,
        )


class TTXApplyRoPE(MojoApplyRoPE):
    supported_platforms_list = ["npu", "ilu", "mlu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rope_fwd(q, k, cos, sin, head_first)


class TTXMRoPE(MojoMRoPE):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool = False,
        head_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return mrope_fwd_impl(q, k, cos, sin, mrope_section, is_interleaved, head_dim)


class TTXVisionRotaryEmbedding2D(MojoVisionRotaryEmbedding2D):
    supported_platforms_list = ["npu"]

    def forward(self, grid_hw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return vision_rot_pos_embed(
            self.inv_freq,
            grid_hw,
            self.rope_dim,
            self.adapooling_factor,
        )


class TTXApplyVisionRoPE2D(MojoApplyVisionRoPE2D):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return vision_rope_apply(q, k, cos, sin)