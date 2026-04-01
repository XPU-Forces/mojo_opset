from typing import Tuple

import torch
import torch_npu

from mojo_opset.core import MojoRoPE


class TorchNpuRoPE(MojoRoPE, default_priority=0):
    supported_platforms_list = ["npu"]

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
            
        print(q_rope.shape, cos.shape)

        q_rot = torch_npu.npu_rotary_mul(q_rope, cos, sin)
        k_rot = torch_npu.npu_rotary_mul(k_rope, cos, sin)

        if is_less_than_4d:
            q_rot = q_rot.squeeze(0)
            k_rot = k_rot.squeeze(0)

        if nope_dim > 0:
            q_rot = torch.cat([q_nope, q_rot], dim=-1)
            k_rot = torch.cat([k_nope, k_rot], dim=-1)

        return q_rot, k_rot
