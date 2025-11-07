import torch
from mojo_opset.core import MojoRoPE
import torch_npu


class TorchRoPE(MojoRoPE, default_priority=0):
    def forward_std(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
        if sin.dim() == 3:
            sin = sin.unsqueeze(1)
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
        return q_embed, k_embed
