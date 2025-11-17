import torch

from mojo_opset.backends.ttx.kernels.ascend.layer_norm import ttx_layer_norm
from mojo_opset.core import MojoNorm
from mojo_opset.core import MojoRMSNormFunction


class TTXNorm(MojoNorm, default_priority=0):
    def forward_std(self, hidden_state: torch.Tensor):
        if self.norm_type == "rmsnorm":
            return torch.ops.ttx.rmsnorm_infer(hidden_state, self.gamma, self.epsilon)
        elif self.norm_type == "layernorm":
            return ttx_layer_norm(hidden_state, self.gamma, self.beta, self.epsilon)
        else:
            raise NotImplementedError(f"[TTXNorm] Only support rmsnorm/layernorm, but got {self.norm_type}")


class TTXRMSNormFunction(MojoRMSNormFunction):
    """
    TODO(zhangjihang, zhaowenshuo):
    Liger's rms_norm bwd support inplace modifying dy in order to reduce memory usage.
    By now, we just implement out-place function.
    We can support this soon when we start to think about functionalize.
    Ref: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py#L527
    """

    @staticmethod
    def forward(ctx, X, W, eps):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        # FIXME: Currently, MojoNormFunction base class does not define fields like 'offset', so they are hardcoded here temporarily.
        offset = 0.0
        casting_mode = "llama"
        str_to_casting_mode = {"llama": 0, "gemma": 1, "none": -1}
        casting_mode_int = str_to_casting_mode[casting_mode]

        Y, RSTD = torch.ops.ttx.rmsnorm_fwd(X, W, eps, offset, casting_mode_int)

        ctx.save_for_backward(X, W, RSTD)

        ctx.offset = offset

        ctx.casting_mode_int = casting_mode_int
        ctx.X_dtype = X.dtype

        return Y

    @staticmethod
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        X, W, RSTD = ctx.saved_tensors

        dX, dW = torch.ops.ttx.rmsnorm_bwd(
            dY=dY,
            X=X,
            W=W,
            RSTD=RSTD,
            offset=ctx.offset,
            casting_mode_int=ctx.casting_mode_int,
            X_dtype=ctx.X_dtype,
        )

        return dX, dW, None, None, None, None
