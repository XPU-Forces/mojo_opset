from typing import Tuple
import logging

import torch
import torch_npu

from mojo_opset.core import MojoApplyRoPE
from mojo_opset.core import MojoRotaryEmbedding

logger = logging.getLogger(__name__)


class TorchNpuRotaryEmbedding(MojoRotaryEmbedding, default_priority=0):
    """Register RoPE table generation for torch_npu; npu_rotary_mul is used by ApplyRoPE."""

    supported_platforms_list = ["npu"]


class TorchNpuApplyRoPE(MojoApplyRoPE, default_priority=0):
    supported_platforms_list = ["npu"]

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_q, orig_k, orig_cos, orig_sin = q, k, cos, sin
        rope_dim = cos.shape[-1]
        nope_dim = q.shape[-1] - rope_dim

        if nope_dim > 0:
            q_nope, q_rope = torch.split(q, [nope_dim, rope_dim], dim=-1)
            k_nope, k_rope = torch.split(k, [nope_dim, rope_dim], dim=-1)
        else:
            q_rope, k_rope = q, k

        # npu_rotary_mul requires 4D input
        is_less_than_4d = q_rope.dim() < 4
        if is_less_than_4d:
            q_rope = q_rope.unsqueeze(0)
            k_rope = k_rope.unsqueeze(0)
            
        # npu_rotary_mul requires cos/sin to be 4D
        if cos.dim() < 4:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if q_rope.shape[0] > 1 and cos.shape[1] == 1 and rope_dim // 2 * q_rope.element_size() % 32 != 0:
            logger.warning(
                "TorchNpuApplyRoPE: npu_rotary_mul does not support q/k layout [B, N, S, D] "
                "when rope_dim is not 32-byte aligned (rope_dim=%s), fallback torch rope",
                rope_dim,
            )
            return super()._apply_rope(orig_q, orig_k, orig_cos, orig_sin)

        try:
            q_rot = torch_npu.npu_rotary_mul(q_rope, cos, sin)
            k_rot = torch_npu.npu_rotary_mul(k_rope, cos, sin)
        except (RuntimeError, NotImplementedError) as exc:
            logger.warning("TorchNpuApplyRoPE: npu_rotary_mul failed (%s), fallback torch rope", exc)
            return super()._apply_rope(orig_q, orig_k, orig_cos, orig_sin)

        if is_less_than_4d:
            q_rot = q_rot.squeeze(0)
            k_rot = k_rot.squeeze(0)

        if nope_dim > 0:
            q_rot = torch.cat([q_nope, q_rot], dim=-1)
            k_rot = torch.cat([k_nope, k_rot], dim=-1)

        return q_rot, k_rot
