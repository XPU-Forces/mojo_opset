import torch
import torch.nn.functional as F

from mojo_opset.core.function import MojoFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoSdpaFunction(MojoFunction):
    @staticmethod
    def forward_ref(ctx, query, key, value, mask, scale=1.0, enable_gqa=False):
        # ctx.scale = scale
        # ctx.enable_gqa = enable_gqa
        # ctx.scale = scale
        # if enable_gqa:
        #     assert query.shape[1] % key.shape[1] == 0
        #     group_size = query.shape[1] // key.shape[1]
        #     key = key.repeat_interleave(group_size, dim=1)
        #     value = value.repeat_interleave(group_size, dim=1)
        # score = torch.matmul(query, key.transpose(-1, -2)).to(torch.float32) * scale
        # # score = torch.matmul(query, key.transpose(-1, -2)) * scale
        # score.masked_fill_(~mask, float("-inf"))
        # p = F.softmax(score - torch.max(score, dim=-1, keepdim=True).values, dim=-1)
        # # p = F.softmax(score, dim=-1)
        # output = torch.matmul(p.to(query.dtype), value)
        # ctx.save_for_backward(query, key, value, output, mask)
        # print("forward_ref output:", output)
        # return output

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
        print("forward output:", output)
        return output

    @staticmethod
    def backward_ref(ctx, do):
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
