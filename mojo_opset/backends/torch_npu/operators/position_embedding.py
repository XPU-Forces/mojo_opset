from typing import Tuple
import torch
import torch_npu

from mojo_opset.core import MojoRoPE
from mojo_opset.core import MojoMRoPE

class TorchNpuRoPE(MojoRoPE, default_priority=0):
    def __init__(self, rotary_offset=0, interleaved=False, dynamic_ntk=False, max_seq_len=None, is_varlen=True):
        super().__init__(interleaved=interleaved)
        self.rotary_offset = rotary_offset
        self.dynamic_ntk = dynamic_ntk
        self.max_seq_len = max_seq_len
        self.is_varlen = is_varlen

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_percentage: float,
    ):
        return self._apply_rope_npu(q, k, cos, sin, rope_percentage)

    def _apply_rope_npu(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_percentage: float,
    ):
        rope_dim = int(q.shape[-1] * rope_percentage)
        nope_dim = q.shape[-1] - rope_dim

        if nope_dim > 0:
            q_nope, q_rope = torch.split(q, [nope_dim, rope_dim], dim=-1)
            k_nope, k_rope = torch.split(k, [nope_dim, rope_dim], dim=-1)
        else:
            q_rope, k_rope = q, k
            q_nope, k_nope = None, None

        # Handle < 4D input for npu_rotary_mul which requires 4D
        is_less_than_4d = q_rope.dim() < 4
        if is_less_than_4d:
            q_rope = q_rope.unsqueeze(0)
            k_rope = k_rope.unsqueeze(0)
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        q_rot = torch_npu.npu_rotary_mul(q_rope, cos, sin)
        k_rot = torch_npu.npu_rotary_mul(k_rope, cos, sin)

        if is_less_than_4d:
            q_rot = q_rot.squeeze(0)
            k_rot = k_rot.squeeze(0)

        if nope_dim > 0:
            q_rot = torch.cat([q_nope, q_rot], dim=-1)
            k_rot = torch.cat([k_nope, k_rot], dim=-1)

        return q_rot, k_rot

class TorchNpuMRoPE(MojoMRoPE):
    supported_platforms_list = ["npu"]
    def __init__(self):
        super().__init__(head_size, mrope_section) # type: ignore
        self.head_size = head_size  # type: ignore
        self.mrope_section = mrope_section  # type: ignore

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rotary_mode: str = "half",
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert rotary_mode in ["half", "full"], "rotary_mode must be 'half' or 'full'"
        mrope_section = [0, 0, 0] if positions.ndim == 1 else self.mrope_section
        query, key = torch_npu.npu_mrope(
                positions,
                query,
                key,
                cos_sin_cache,
                self.head_size,
                mrope_section=mrope_section,
                rotary_mode=rotary_mode,
            )
        return query, key