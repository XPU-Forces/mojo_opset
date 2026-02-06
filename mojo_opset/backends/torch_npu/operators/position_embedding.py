import torch
import torch_npu
from mojo_opset.core import MojoRoPE


class TorchNpuRoPE(MojoRoPE, default_priority=0):
    def __init__(self, rotary_offset=0, interleaved=False, dynamic_ntk=False, max_seq_len=None, is_varlen=True):
        super().__init__(rotary_offset=rotary_offset, interleaved=interleaved, dynamic_ntk=dynamic_ntk, max_seq_len=max_seq_len, is_varlen=is_varlen)

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids=None, cum_sum_query_len=None):
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
        if sin.dim() == 3:
            sin = sin.unsqueeze(1)
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
        return q_embed, k_embed