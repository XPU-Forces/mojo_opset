import torch
from mojo_opset.backends.ttx_kernels.src.ascend.rms_norm import ttx_rms_norm, rms_norm_fwd, rms_norm_bwd
from mojo_opset.backends.ttx_kernels.src.ascend.layer_norm import ttx_layer_norm

from mojo_opset.core import MojoNorm, MojoRMSNormFunction
from mojo_opset.backends.ttx_kernels.src.ascend.utils import torch_to_triton_dtype


class TTXNorm(MojoNorm, default_priority=0):
    def forward_std(self, hidden_state: torch.Tensor):
        if self.norm_type == "rmsnorm":
            return rms_norm_fwd(hidden_state, self.gamma, self.epsilon, offset=0.0, casting_mode="llama")[0]
        elif self.norm_type == "layernorm":
            return ttx_layer_norm(hidden_state, self.gamma, self.beta, self.epsilon)
        else:
            raise NotImplementedError(f"[TTXNorm] Only support rmsnorm/layernorm, but got {self.norm_type}")


class TTXRMSNormFunction(MojoRMSNormFunction):
    @staticmethod
    def forward(ctx, X, W, eps):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        # FIXME: Currently, MojoNormFunction base class does not define fields like 'offset', so they are hardcoded here temporarily.
        offset = 0.0
        casting_mode = "llama"
        in_place = True

        Y, X_2d, RSTD = rms_norm_fwd(X, W, eps, offset, casting_mode)

        ctx.save_for_backward(X_2d, W, RSTD)

        ctx.offset = offset
        ctx.in_place = in_place

        str_to_casting_mode = {"llama": 0, "gemma": 1, "none": -1}
        ctx.casting_mode_int = str_to_casting_mode[casting_mode]
        ctx.X_dtype_triton = torch_to_triton_dtype.get(X.dtype)

        return Y

    @staticmethod
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        X_2d, W, RSTD = ctx.saved_tensors

        dX, dW = rms_norm_bwd(
            dY=dY,
            X_2d=X_2d,
            W=W,
            RSTD=RSTD,
            offset=ctx.offset,
            casting_mode_int=ctx.casting_mode_int,
            in_place=ctx.in_place,
            X_dtype_triton=ctx.X_dtype_triton,
        )

        return dX, dW, None, None, None, None
