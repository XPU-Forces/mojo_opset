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
        return dq, dk, dv
