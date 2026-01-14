import torch

from ..function import MojoFunction


class MojoRoPEFunction(MojoFunction):
    """
    RoPE function.
    The RoPE function is defined as: RoPE(q, k) = q * cos + rotate_half(q) * sin + k * cos + rotate_half(k) * sin.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function of RoPE.
        Args:
            ctx: The context object.
            q: The query tensor.
            k: The key tensor.
            cos: The cosine tensor.
            sin: The sine tensor.

        Returns:
            tuple: The rotated query tensor and key tensor.
        """

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        ctx.save_for_backward(cos, sin)

        return q_rot, k_rot

    @staticmethod
    def backward(
        ctx,
        grad_output_q: torch.Tensor,
        grad_output_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        The backward function of RoPE.
        Args:
            ctx: The context object.
            grad_output_q: The gradient of the output with respect to q.
            grad_output_k: The gradient of the output with respect to k.

        Returns:
            tuple: The gradient of the output with respect to q and k.
        """

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


def mojo_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The RoPE function.
    Args:
        q: The query tensor.
        k: The key tensor.
        cos: The cosine tensor.
        sin: The sine tensor.

    Returns:
        tuple: The rotated query tensor and key tensor.
    """
    return MojoRoPEFunction.apply(q, k, cos, sin)
