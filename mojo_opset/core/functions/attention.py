import torch
import torch.nn.functional as F

from mojo_opset.core.function import MojoFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoDiffusionAttentionFunction(MojoFunction):
    """
    MojoDiffusionAttentionFunction implements the specific attention for text diffusion.
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        scale: float = 1.0,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for diffusion attention.

        Args:
            ctx: Context object for the backward.
            query (torch.Tensor): Query tensor. shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
            key (torch.Tensor): Key tensor. shape [BSZ, K_HEAD_NUM, SEQ, HEAD_DIM]
            value (torch.Tensor): Value tensor. shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]
            mask (torch.Tensor): Attention mask tensor. shape [SEQ, SEQ]
            scale (float, optional): Scale factor for attention. Defaults to 1.0.
            enable_gqa (bool, optional): Whether to enable grouped query attention. Defaults to False.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        ctx.save_for_backward(query, key, value, mask)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        Backward pass for diffusion attention.

        Args:
            ctx: Context object for the backward.
            grad_output (torch.Tensor): Gradient of the output tensor. shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]

        Returns:
            tuple: Gradients of query, key, value, None, None, None.
                grad_query: shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
                grad_key: shape [BSZ, K_HEAD_NUM, SEQ, HEAD_DIM]
                grad_value: shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]
        """
        query, key, value, attn_mask = ctx.saved_tensors

        with torch.enable_grad():
            query = query.detach().requires_grad_(True)
            key = key.detach().requires_grad_(True)
            value = value.detach().requires_grad_(True)

            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                scale=ctx.scale,
                enable_gqa=ctx.enable_gqa,
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                output, (query, key, value), grad_output, retain_graph=False, allow_unused=False
            )

        return grad_query, grad_key, grad_value, None, None, None


def mojo_diffusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Applies the diffusion-specific attention mechanism to the input tensors.

    This is a functional wrapper for the `MojoDiffusionAttentionFunction`.

    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        mask (torch.Tensor): The attention mask.
        scale (float, optional): The attention scaling factor. Defaults to 1.0.
        enable_gqa (bool, optional): Whether to enable Grouped-Query
                                     Attention. Defaults to False.

    Returns:
        torch.Tensor: The output of the attention function.
    """
    return MojoDiffusionAttentionFunction.apply(query, key, value, mask, scale, enable_gqa)
