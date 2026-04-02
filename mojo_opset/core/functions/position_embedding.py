import torch

from ..function import MojoFunction


class MojoApplyRoPEFunction(MojoFunction):
    """
    Apply Rotary Position Embedding to q/k with pre-extracted cos/sin.

    Expects cos/sin to already be position-specific (extracted by MojoRotaryEmbedding).
    Supports partial-rope via nope_dim when rope_dim < head_dim.
    """

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _inverse_rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((x2, -x1), dim=-1)

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not head_first:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)

        rope_dim = cos.shape[-1]
        nope_dim = q.shape[-1] - rope_dim

        if nope_dim > 0:
            q_nope, q_rope = torch.split(q, [nope_dim, rope_dim], dim=-1)
            k_nope, k_rope = torch.split(k, [nope_dim, rope_dim], dim=-1)
        else:
            q_rope, k_rope = q, k

        q_rot = (q_rope * cos + MojoApplyRoPEFunction._rotate_half(q_rope) * sin).to(q.dtype)
        k_rot = (k_rope * cos + MojoApplyRoPEFunction._rotate_half(k_rope) * sin).to(k.dtype)

        if nope_dim > 0:
            q_rot = torch.cat([q_nope, q_rot], dim=-1)
            k_rot = torch.cat([k_nope, k_rot], dim=-1)

        ctx.save_for_backward(cos, sin)

        return q_rot.to(q.dtype), k_rot.to(k.dtype)

    @staticmethod
    def backward(
        ctx,
        grad_output_q: torch.Tensor,
        grad_output_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        cos, sin = ctx.saved_tensors
        rope_dim = cos.shape[-1]
        nope_dim = grad_output_q.shape[-1] - rope_dim

        if nope_dim > 0:
            grad_q_nope, grad_q_rope = torch.split(grad_output_q, [nope_dim, cos.shape[-1]], dim=-1)
            grad_k_nope, grad_k_rope = torch.split(grad_output_k, [nope_dim, cos.shape[-1]], dim=-1)
        else:
            grad_q_rope, grad_k_rope = grad_output_q, grad_output_k

        grad_q = (
            grad_q_rope * cos + MojoApplyRoPEFunction._inverse_rotate_half(grad_q_rope * sin)
        ).to(grad_output_q.dtype)
        grad_k = (
            grad_k_rope * cos + MojoApplyRoPEFunction._inverse_rotate_half(grad_k_rope * sin)
        ).to(grad_output_k.dtype)

        if nope_dim > 0:
            grad_q = torch.cat([grad_q_nope, grad_q], dim=-1)
            grad_k = torch.cat([grad_k_nope, grad_k], dim=-1)

        return grad_q, grad_k, None, None, None