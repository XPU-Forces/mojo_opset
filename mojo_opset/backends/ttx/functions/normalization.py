import torch

from mojo_opset.backends.ttx.kernels import rmsnorm_bwd
from mojo_opset.backends.ttx.kernels import rmsnorm_fwd
from mojo_opset.core import MojoRMSNormFunction


class TTXRMSNormFunction(MojoRMSNormFunction):
    """
    TODO(zhangjihang, zhaowenshuo):
    Liger's rms_norm bwd support inplace modifying dy in order to reduce memory usage.
    By now, we just implement out-place function.
    We can support this soon when we start to think about functionalize.
    Ref: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py#L527
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # FIXME: Currently, MojoNormFunction base class does not define fields like 'offset', so they are hardcoded here temporarily.
        offset = 0.0
        casting_mode = "llama"
        str_to_casting_mode = {"llama": 0, "gemma": 1, "none": -1}
        casting_mode_int = str_to_casting_mode[casting_mode]

        Y, RSTD = rmsnorm_fwd(
            input,
            weight,
            eps,
            offset,
            casting_mode_int,
        )

        ctx.save_for_backward(input, weight, RSTD)

        ctx.offset = offset

        ctx.casting_mode_int = casting_mode_int
        ctx.X_dtype = input.dtype

        return Y

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        X, W, RSTD = ctx.saved_tensors

        dX, dW = rmsnorm_bwd(
            dY=grad_output,
            X=X,
            W=W,
            RSTD=RSTD,
            offset=ctx.offset,
            casting_mode_int=ctx.casting_mode_int,
            X_dtype=ctx.X_dtype,
        )

        return dX, dW, None
