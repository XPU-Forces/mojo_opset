from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import fused_linear_cross_entropy_1d_bwd
from mojo_opset.backends.ttx.kernels import fused_linear_cross_entropy_1d_fwd
from mojo_opset.backends.ttx.kernels import fused_linear_cross_entropy_bwd
from mojo_opset.backends.ttx.kernels import fused_linear_cross_entropy_fwd
from mojo_opset.core import MojoFusedLinearCrossEntropyFunction
from mojo_opset.utils.platform import get_torch_device

device = get_torch_device()


def _amp_custom(kind: str):
    def decorator(func):
        amp_modules = []
        amp = getattr(torch, "amp", None)
        if amp is not None:
            amp_modules.append(amp)
        cuda = getattr(torch, "cuda", None)
        cuda_amp = getattr(cuda, "amp", None) if cuda is not None else None
        if cuda_amp is not None:
            amp_modules.append(cuda_amp)

        for amp_mod in amp_modules:
            fn = getattr(amp_mod, f"custom_{kind}", None)
            if fn is None:
                continue
            attempts = (
                lambda: fn(device_type=device)(func),  # torch.amp custom_fwd/custom_bwd
                lambda: fn(func),                      # older torch.cuda.amp custom_bwd
                lambda: fn()(func),                    # older torch.cuda.amp custom_fwd factory
            )
            for attempt in attempts:
                try:
                    return attempt()
                except TypeError:
                    continue
        return func

    return decorator


class TTXFusedLinearCrossEntropyFunction(MojoFusedLinearCrossEntropyFunction):
    @staticmethod
    @_amp_custom("fwd")
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ce_weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # downcast to dtype and store for backward
        if reduction == "none":
            loss, z_loss = fused_linear_cross_entropy_1d_fwd(
                _input=input_tensor,
                weight=weight,
                target=target,
                bias=bias,
                ce_weight=ce_weight,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                softcap=softcap,
                return_z_loss=return_z_loss,
            )
            ctx.save_for_backward(
                input_tensor.detach(),
                weight.detach() if weight is not None else None,
                bias.detach() if bias is not None else None,
            )
            ctx.target = target
            ctx.ce_weight = ce_weight
            ctx.ignore_index = ignore_index
            ctx.lse_square_scale = lse_square_scale
            ctx.label_smoothing = label_smoothing
            ctx.reduction = reduction
            ctx.softcap = softcap
            ctx.return_z_loss = return_z_loss
            ctx.accum_dtype = accum_dtype

        else:
            loss, z_loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_fwd(
                _input=input_tensor,
                weight=weight,
                target=target,
                bias=bias,
                ce_weight=ce_weight,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                return_z_loss=return_z_loss,
                accum_dtype=accum_dtype,
            )
            ctx.save_for_backward(
                grad_input.detach(),
                grad_weight.detach() if grad_weight is not None else None,
                grad_bias.detach() if bias is not None else None,
            )
            ctx.return_z_loss = return_z_loss
            ctx.reduction = reduction

        if return_z_loss:
            return loss, z_loss
        else:
            return loss, None

    @staticmethod
    @_amp_custom("bwd")
    def backward(
        ctx,
        grad_loss: torch.Tensor,
        grad_z_loss: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, None, None, None, None, None, None]:
        if ctx.return_z_loss:
            del grad_z_loss  # z_loss is only for logging

        if ctx.reduction == "none":
            _input, weight, bias = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_1d_bwd(
                grad_output=grad_loss,
                _input=_input,
                weight=weight,
                target=ctx.target,
                bias=bias,
                ce_weight=ctx.ce_weight,
                ignore_index=ctx.ignore_index,
                lse_square_scale=ctx.lse_square_scale,
                label_smoothing=ctx.label_smoothing,
                softcap=ctx.softcap,
                accum_dtype=ctx.accum_dtype,
            )
        else:
            (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_bwd(
                grad_loss, grad_input, grad_weight, grad_bias
            )

        return grad_input, grad_weight, None, grad_bias, None, None, None, None, None, None, None, None
