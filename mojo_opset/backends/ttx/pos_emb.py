import torch

from mojo_opset.backends.ttx.kernels import rope_bwd
from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.core import MojoRoPE
from mojo_opset.core import MojoRoPEFunction


class TTXRoPE(MojoRoPE, default_priority=0):
    def forward_std(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return rope_fwd(q, k, cos, sin)


class TTXRoPEFunction(MojoRoPEFunction):
    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_rope, k_rope = rope_fwd(q, k, cos, sin)

        ctx.save_for_backward(cos, sin)
        # NOTE(zhangjihang): why we need save seq length here?
        # ctx.seq_len = seq_len
        return q_rope, k_rope

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        # seq_len = ctx.seq_len

        grad_q, grad_k = rope_bwd(dq, dk, sin, cos)

        return grad_q, grad_k, None, None
