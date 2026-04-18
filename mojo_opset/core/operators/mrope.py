from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


class MojoMRoPE(MojoOperator):
    """Multimodal Rotary Position Embedding (MRoPE) for Qwen2-VL.

    Applies 3D rotary position embedding over temporal (T), height (H), and width (W)
    dimensions to query and key tensors. Supports both interleaved and non-interleaved modes.

    Reference: https://qwenlm.github.io/blog/qwen2-vl/
    """

    supported_platforms_list = ["npu"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extra_repr(self) -> str:
        return ""

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        hidden_size = hidden_states.shape[-1]
        hidden_states_half = hidden_size // 2
        left = hidden_states[..., :hidden_states_half]
        right = hidden_states[..., hidden_states_half:]
        return torch.cat((-right, left), dim=-1)

    @staticmethod
    def _apply_interleaved_mrope(
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply interleaved MRoPE pattern to cos/sin tables."""
        cos_interleaved = cos_table[0].clone()
        cos_interleaved[..., 1::3] = cos_table[1, ..., 1::3]
        cos_interleaved[..., 2::3] = cos_table[2, ..., 2::3]

        sin_interleaved = sin_table[0].clone()
        sin_interleaved[..., 1::3] = sin_table[1, ..., 1::3]
        sin_interleaved[..., 2::3] = sin_table[2, ..., 2::3]

        return cos_interleaved, sin_interleaved

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool = False,
        head_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Multimodal Rotary Position Embedding to query and key tensors.

        Args:
            query: ``(num_tokens, n_qh * head_dim)`` query tensor.
            key: ``(num_tokens, n_kh * head_dim)`` key tensor.
            cos_table: ``(3, num_tokens, rotary_dim // 2)`` cos values for T/H/W dimensions.
            sin_table: ``(3, num_tokens, rotary_dim // 2)`` sin values for T/H/W dimensions.
            mrope_section: ``[t_section, h_section, w_section]`` - how half rope_dim is split.
            is_interleaved: if True, T/H/W positions are interleaved.
            head_dim: head dimension. If None, inferred from cos_table (assumes rope_dim == head_dim).

        Returns:
            ``(query, key)`` with RoPE applied, same shape as input.
        """
        num_tokens, n_qh_head_dim = query.shape
        num_tokens_k, n_kh_head_dim = key.shape

        rope_dim = sum(mrope_section) * 2
        half_rope_dim = rope_dim // 2

        if head_dim is None:
            head_dim = cos_table.shape[-1] * 2
            if head_dim != rope_dim:
                raise ValueError(
                    f"head_dim ({head_dim}) inferred from cos_table does not match "
                    f"rope_dim ({rope_dim}). Please pass head_dim explicitly."
                )

        n_qh = n_qh_head_dim // head_dim
        n_kh = n_kh_head_dim // head_dim

        query = query.view(num_tokens, n_qh, head_dim)
        key = key.view(num_tokens_k, n_kh, head_dim)

        query_rot, query_pass = query.split([rope_dim, head_dim - rope_dim], dim=-1)
        key_rot, key_pass = key.split([rope_dim, head_dim - rope_dim], dim=-1)

        if cos_table.dim() == 3:
            if is_interleaved:
                cos_table, sin_table = self._apply_interleaved_mrope(cos_table, sin_table, mrope_section)
            else:
                cos_table = torch.cat([m[i] for i, m in enumerate(cos_table.split(mrope_section, dim=-1))], dim=-1)
                sin_table = torch.cat([m[i] for i, m in enumerate(sin_table.split(mrope_section, dim=-1))], dim=-1)

        cos_table = cos_table.view(num_tokens, half_rope_dim)
        sin_table = sin_table.view(num_tokens, half_rope_dim)

        query_rot_half1 = query_rot[..., :half_rope_dim]
        query_rot_half2 = query_rot[..., half_rope_dim:]
        key_rot_half1 = key_rot[..., :half_rope_dim]
        key_rot_half2 = key_rot[..., half_rope_dim:]

        cos_expanded = cos_table.unsqueeze(1)
        sin_expanded = sin_table.unsqueeze(1)

        query_rot_new_half1 = query_rot_half1 * cos_expanded - query_rot_half2 * sin_expanded
        query_rot_new_half2 = query_rot_half2 * cos_expanded + query_rot_half1 * sin_expanded
        key_rot_new_half1 = key_rot_half1 * cos_expanded - key_rot_half2 * sin_expanded
        key_rot_new_half2 = key_rot_half2 * cos_expanded + key_rot_half1 * sin_expanded

        query_rot = torch.cat([query_rot_new_half1, query_rot_new_half2], dim=-1)
        key_rot = torch.cat([key_rot_new_half1, key_rot_new_half2], dim=-1)

        query = torch.cat([query_rot, query_pass], dim=-1).view(num_tokens, -1)
        key = torch.cat([key_rot, key_pass], dim=-1).view(num_tokens_k, -1)

        return query, key
