from typing import Optional
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import rot_pos_embed
from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.core import MojoRotaryEmbedding
from mojo_opset.core import MojoApplyRoPE


class TTXRotaryEmbedding(MojoRotaryEmbedding):
    supported_platforms_list = ["npu", "ilu"]

    def __init__(self, rope_theta, rope_dim, attention_scaling: float = 1.0, init_max_length: Optional[int] = None, **kwargs):
        super().__init__(rope_theta, rope_dim, attention_scaling, init_max_length, **kwargs)
        if init_max_length is None:
            raise ValueError("init_max_length must be provided for TTXRotaryEmbedding")

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        total_seq_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if cu_seqlens_q is not None:
            assert cu_seqlens_q.dtype == torch.int32
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        if position_ids is not None:
            assert position_ids.dtype == torch.int32
        return rot_pos_embed(
            x,
            self.cos,
            self.sin,
            cu_seqlens_q=cu_seqlens_q,
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
