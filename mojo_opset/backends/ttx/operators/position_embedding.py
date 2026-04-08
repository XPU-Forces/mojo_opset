from typing import Optional
from typing import Tuple

import torch

import mojo_opset.core.operators.position_embedding as mojo_position_embedding
from mojo_opset.backends.ttx.kernels import rot_pos_embed
from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.core import MojoRotaryEmbedding
from mojo_opset.core import MojoApplyRoPE


class TTXRotaryEmbedding(MojoRotaryEmbedding):
    supported_platforms_list = ["npu", "ilu"]

    def __init__(self, rope_theta, rope_dim, init_max_length: Optional[int] = None, **kwargs):
        super().__init__(rope_theta, rope_dim, init_max_length, **kwargs)
        if init_max_length is None:
            raise ValueError("init_max_length must be provided for TTXRotaryEmbedding")

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        seqlens_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return rot_pos_embed(
            x,
            self.cos,
            self.sin,
            cu_seqlens_q=cu_seqlens_q,
            seqlens_kv=seqlens_kv,
            position_ids=position_ids,
        )


class TTXApplyRoPE(MojoApplyRoPE):
    supported_platforms_list = ["npu", "ilu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rope_fwd(q, k, cos, sin, head_first)


def _patch_torch_apply_rope_for_batched_cos() -> None:
    """Align torch ref with padded prefill cos/sin [B, S, rope] (pos3d) without editing core."""
    torch_cls = getattr(mojo_position_embedding, "TorchApplyRoPE", None)
    if torch_cls is None:
        return
    _orig_forward = torch_cls.forward

    def forward(  # type: ignore[no-untyped-def]
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ):
        if head_first and q.ndim == 4 and cos.ndim == 3 and sin.ndim == 3:
            return MojoApplyRoPE._apply_rope(self, q, k, cos.unsqueeze(1), sin.unsqueeze(1))
        return _orig_forward(self, q, k, cos, sin, head_first)

    torch_cls.forward = forward  # type: ignore[method-assign]


_patch_torch_apply_rope_for_batched_cos()
