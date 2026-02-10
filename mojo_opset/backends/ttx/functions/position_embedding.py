import torch

from mojo_opset.backends.ttx.kernels import rope_bwd
from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.core import MojoRoPEFunction


class TTXRoPEFunction(MojoRoPEFunction):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_rope, k_rope = rope_fwd(q, k, cos, sin)

        ctx.save_for_backward(cos, sin)

        return q_rope, k_rope

    @staticmethod
    def backward(
        ctx,
        grad_output_q: torch.Tensor,
        grad_output_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors

        grad_q, grad_k = rope_bwd(grad_output_q, grad_output_k, sin, cos)

        return grad_q, grad_k, None, None
