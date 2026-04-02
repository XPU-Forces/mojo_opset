import torch

from mojo_opset.backends.ttx.kernels import rope_bwd
from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.core import MojoApplyRoPEFunction


class TTXApplyRoPEFunction(MojoApplyRoPEFunction):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_rope, k_rope = rope_fwd(q, k, cos, sin, head_first)

        ctx.save_for_backward(cos, sin)
        ctx.head_first = head_first

        return q_rope, k_rope

    @staticmethod
    def backward(
        ctx,
        grad_output_q: torch.Tensor,
        grad_output_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        cos, sin = ctx.saved_tensors

        grad_q, grad_k = rope_bwd(grad_output_q, grad_output_k, cos, sin, ctx.head_first)

        return grad_q, grad_k, None, None, None
