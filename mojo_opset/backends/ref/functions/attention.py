import torch
import torch.nn.functional as F

from mojo_opset.core import MojoDiffusionAttentionFunction


class RefDiffusionAttentionFunction(MojoDiffusionAttentionFunction):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale=1.0, enable_gqa=False):
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
    def backward(ctx, do):
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
                output, (query, key, value), do, retain_graph=False, allow_unused=False
            )

        return grad_query, grad_key, grad_value, None, None, None
