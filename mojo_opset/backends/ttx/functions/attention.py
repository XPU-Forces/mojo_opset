from mojo_opset.backends.ttx.kernels import sdpa_bwd
from mojo_opset.backends.ttx.kernels import sdpa_fwd
from mojo_opset.core import MojoSdpaFunction


class TTXSdpaFunction(MojoSdpaFunction):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale=1.0, enable_gqa=False):
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa
        output, lse = sdpa_fwd(
            query,
            key,
            value,
            mask,
            scale,
            enable_gqa,
        )
        ctx.save_for_backward(query, key, value, mask, output, lse)
        return output

    @staticmethod
    def backward(ctx, do):
        query, key, value, mask, output, lse = ctx.saved_tensors
        dq, dk, dv = sdpa_bwd(
            output,
            do,
            query,
            key,
            value,
            lse,
            mask,
            ctx.scale,
            ctx.enable_gqa,
        )
        return dq, dk, dv, None, None, None


# class TTXSdpaFunction(MojoSdpaFunction):
#     @staticmethod
#     def forward(ctx, query, key, value, attn_mask=None, scale=1.0, enable_gqa=False):
#         ctx.save_for_backward(query, key, value, attn_mask)
#         ctx.scale = scale
#         ctx.enable_gqa = enable_gqa

#         import pdb

#         pdb.set_trace()

#         output = F.scaled_dot_product_attention(
#             query,
#             key,
#             value,
#             attn_mask=attn_mask,
#             scale=scale,
#             enable_gqa=enable_gqa,
#         )
#         print("forward output:", output)
#         return output

#     @staticmethod
#     def backward(ctx, do):
#         query, key, value, attn_mask = ctx.saved_tensors

#         with torch.enable_grad():
#             query = query.detach().requires_grad_(True)
#             key = key.detach().requires_grad_(True)
#             value = value.detach().requires_grad_(True)

#             output = F.scaled_dot_product_attention(
#                 query,
#                 key,
#                 value,
#                 attn_mask=attn_mask,
#                 scale=ctx.scale,
#                 enable_gqa=ctx.enable_gqa,
#             )

#             grad_query, grad_key, grad_value = torch.autograd.grad(
#                 output, (query, key, value), do, retain_graph=False, allow_unused=False
#             )

#         return grad_query, grad_key, grad_value, None, None, None
