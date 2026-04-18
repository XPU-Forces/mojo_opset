from typing import List
from typing import Tuple

import torch

from ..operator import MojoOperator


class MojoMRoPE(MojoOperator):
    supported_platforms_list = ["npu"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extra_repr(self) -> str:
        return ""

    @staticmethod
    def apply_interleaved_mrope(x: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
        x_t = x[0].clone()
        x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
        x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
        return x_t

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Multimodal RoPE to q and k tensors.

        Args:
            q: [num_tokens, n_qh * head_dim] tensor
            k: [num_tokens, n_kh * head_dim] tensor
            cos: [3, num_tokens, rotary_dim // 2] cos values for T/H/W dimensions
            sin: [3, num_tokens, rotary_dim // 2] sin values for T/H/W dimensions
            mrope_section: [t_section, h_section, w_section] - how half rope_dim is split
            is_interleaved: if True, T/H/W positions are interleaved

        Returns:
            (q, k) with RoPE applied
        """
        num_tokens, n_qh_hd = q.shape
        head_dim = cos.shape[-1] * 2
        rope_dim = sum(mrope_section) * 2
        half_rope_dim = rope_dim // 2
        n_kh_hd = k.shape[1]

        n_qh = n_qh_hd // head_dim
        n_kh = n_kh_hd // head_dim

        q = q.view(num_tokens, n_qh, head_dim)
        k = k.view(num_tokens, n_kh, head_dim)

        q_rot = q[..., :rope_dim]
        q_pass = q[..., rope_dim:]
        k_rot = k[..., :rope_dim]
        k_pass = k[..., rope_dim:]

        if cos.dim() == 3:
            if is_interleaved:
                cos = self.apply_interleaved_mrope(cos, mrope_section)
                sin = self.apply_interleaved_mrope(sin, mrope_section)
            else:
                cos = torch.cat([m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
                sin = torch.cat([m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)

        cos = cos.view(num_tokens, half_rope_dim)
        sin = sin.view(num_tokens, half_rope_dim)

        q_rot_half1 = q_rot[..., :half_rope_dim]
        q_rot_half2 = q_rot[..., half_rope_dim:]
        k_rot_half1 = k_rot[..., :half_rope_dim]
        k_rot_half2 = k_rot[..., half_rope_dim:]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot_new_half1 = q_rot_half1 * cos - q_rot_half2 * sin
        q_rot_new_half2 = q_rot_half2 * cos + q_rot_half1 * sin
        k_rot_new_half1 = k_rot_half1 * cos - k_rot_half2 * sin
        k_rot_new_half2 = k_rot_half2 * cos + k_rot_half1 * sin

        q_rot = torch.cat([q_rot_new_half1, q_rot_new_half2], dim=-1)
        k_rot = torch.cat([k_rot_new_half1, k_rot_new_half2], dim=-1)

        q = torch.cat([q_rot, q_pass], dim=-1).view(num_tokens, -1)
        k = torch.cat([k_rot, k_pass], dim=-1).view(num_tokens, -1)

        return q, k
