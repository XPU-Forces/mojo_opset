import torch

from ..mojo_function import MojoFuncBase
from ..mojo_function import mojo_func_dispatcher


@mojo_func_dispatcher
class MojoRoPEFunction(MojoFuncBase):
    @staticmethod
    def forward_ref(ctx, q, k, cos, sin):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        ctx.save_for_backward(cos, sin)

        return q_rot, k_rot

    @staticmethod
    def backward_ref(ctx, grad_output_q, grad_output_k):
        cos, sin = ctx.saved_tensors

        def inverse_rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((x2, -x1), dim=-1)

        grad_q_part1 = grad_output_q * cos

        grad_q_part2 = inverse_rotate_half(grad_output_q * sin)
        grad_q = grad_q_part1 + grad_q_part2

        grad_k_part1 = grad_output_k * cos

        grad_k_part2 = inverse_rotate_half(grad_output_k * sin)
        grad_k = grad_k_part1 + grad_k_part2

        return grad_q, grad_k, None, None
