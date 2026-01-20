from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

from mojo_opset.training.function import MojoFunction
from mojo_opset.training.kernel import MojoKernel
from mojo_opset.training.module import MojoModule
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoDiffusionAttentionKernel(MojoKernel):
    """
    MojoDiffusionAttentionFunction implements the specific attention for text diffusion.
    """

    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,  # TODO: shouldn't be used in non-sdpa kernel. maybe remove.
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        enable_gqa = query.shape[1] != key.shape[1]
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale, enable_gqa=enable_gqa)
        return output, None, None

    @staticmethod
    def backward(
        grad_output: torch.Tensor,
        output_f32: torch.Tensor,
        lse: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,  # TODO: not used in non-sdpa kernel. maybe remove.
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass for diffusion attention.

        Args:
            grad_output (torch.Tensor): Gradient of the output tensor. shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]
            query (torch.Tensor): Query tensor. shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
            key (torch.Tensor): Key tensor. shape [BSZ, K_HEAD_NUM, SEQ, HEAD_DIM]
            value (torch.Tensor): Value tensor. shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]
            mask (torch.Tensor): Attention mask tensor. shape [SEQ, SEQ]
            scale (float, optional): Scale factor for attention. Defaults to 1.0.

        Returns:
            tuple: Gradients of query, key, value.
                grad_query (torch.Tensor): shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
                grad_key (torch.Tensor): shape [BSZ, K_HEAD_NUM, SEQ, HEAD_DIM]
                grad_value (torch.Tensor): shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]
        """
        enable_gqa = query.shape[1] != key.shape[1]
        with torch.enable_grad():
            query = query.detach().requires_grad_(True)
            key = key.detach().requires_grad_(True)
            value = value.detach().requires_grad_(True)

            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=scale, enable_gqa=enable_gqa
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                output, (query, key, value), grad_output, retain_graph=False, allow_unused=False
            )

        return grad_query, grad_key, grad_value


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
        mask: torch.Tensor,  # TODO: not used in non-sdpa kernel. maybe remove.
        scale: float = 1.0,
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
        output, output_f32, lse = MojoDiffusionAttentionKernel.forward(query, key, value, mask, scale)
        ctx.save_for_backward(output_f32, lse, query, key, value, mask, scale)

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
            tuple: grad_query, grad_key, grad_value, None, None.
                grad_query: shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]. Gradient of the query tensor.
                grad_key: shape [BSZ, K_HEAD_NUM, SEQ, HEAD_DIM]. Gradient of the key tensor.
                grad_value: shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]. Gradient of the value tensor.
        """
        output_f32, lse, query, key, value, mask, scale = ctx.saved_tensors
        grad_query, grad_key, grad_value = MojoDiffusionAttentionKernel.backward(
            grad_output, output_f32, lse, query, key, value, mask, scale
        )
        return grad_query, grad_key, grad_value, None, None


def mojo_diffusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
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
    return MojoDiffusionAttentionFunction.apply(query, key, value, mask, scale)


class MojoDiffusionAttentionModule(MojoModule):
    """
    MojoDiffusionAttentionModule implements the specific attention for text diffusion.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        scale: float = 1.0,
    ):
        """
        Args:
            mask (torch.Tensor[SEQ, SEQ]): Attention mask tensor.
            scale (float, optional): Scale factor for attention. Defaults to 1.0.
        """
        super().__init__()
        self.scale = scale
        self.mask = mask
        self.attention = MojoDiffusionAttentionFunction()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for diffusion attention module.

        Args:
            query (torch.Tensor): Query tensor. shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
            key (torch.Tensor): Key tensor. shape [BSZ, K_HEAD_NUM, SEQ, HEAD_DIM]
            value (torch.Tensor): Value tensor. shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        return self.attention.apply(query, key, value, self.mask, self.scale)
