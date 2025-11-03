import os
import torch
from ..mojo_function import MojoFuncBase
from ...mojo_utils import get_mojo_exec_mode
import torch.nn.functional as F


class MojoRMSNormFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(ctx, input, weight, eps):
        pass

    @staticmethod
    def forward_ref(ctx, input, weight, eps):
        normalized_shape = (input.shape[-1],)
        y = F.rms_norm(input, normalized_shape, weight=weight, eps=eps)

        ctx.save_for_backward(input, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        return y

    @staticmethod
    def forward(ctx, input, weight, eps=1e-6):
        print("MojoRMSNormFunction forward dispatch")

        if MojoRMSNormFunction._registry:
            impl_func = MojoRMSNormFunction._registry[0][1].forward
        else:
            print("MojoRMSNormFunction has NO any registered implementation")

            impl_func = MojoRMSNormFunction.forward_ref

        layer_idx = ctx.layer_idx if hasattr(ctx, "layer_idx") else -1
        mode_str = get_mojo_exec_mode(MojoRMSNormFunction.__name__, "FWD", layer_idx)

        if mode_str == "STD":
            return impl_func(ctx, input, weight, eps)
        elif mode_str == "DUMP":
            MojoRMSNormFunction.forward_dump(ctx, input, weight, eps)

            return torch.zeros_like(input)
        elif mode_str == "REF":
            return MojoRMSNormFunction.forward_ref(ctx, input, weight, eps)
        elif mode_str == "DIFF":
            ref_result = MojoRMSNormFunction.forward_ref(ctx, input, weight, eps)
            impl_result = impl_func(ctx, input, weight, eps)

            torch.testing.assert_close(ref_result, impl_result)
            return impl_result
        else:
            raise ValueError(f"Invalid forward mode {mode_str} for RMSNorm, please check.")

    @staticmethod
    def backward_dump(ctx, grad_output):
        pass

    @staticmethod
    def backward_ref(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        input_with_grad = input.detach().clone().requires_grad_(True)
        weight_with_grad = weight.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            y_ref = F.rms_norm(input_with_grad, normalized_shape, weight=weight_with_grad, eps=eps)

        y_ref.backward(gradient=grad_output)

        grad_input = input_with_grad.grad
        grad_weight = weight_with_grad.grad

        return grad_input, grad_weight, None

    @staticmethod
    def backward(ctx, grad_output):
        print("MojoRMSNormFunction backward dispatch")

        if MojoRMSNormFunction._registry:
            impl_func = MojoRMSNormFunction._registry[0][1].backward
        else:
            print("MojoRMSNormFunction has NO any registered implementation")
            impl_func = MojoRMSNormFunction.backward_ref

        mode_str = os.environ.get(f"{MojoRMSNormFunction.__name__.upper()}_BWD_MODE", "STD")

        if mode_str == "STD":
            return impl_func(ctx, grad_output)
        elif mode_str == "DUMP":
            MojoRMSNormFunction.backward_dump(ctx, grad_output)

            input_grad = torch.zeros_like(ctx.saved_tensors[0])
            weight_grad = torch.zeros_like(ctx.saved_tensors[1])
            return input_grad, weight_grad, None
        elif mode_str == "REF":
            return MojoRMSNormFunction.backward_ref(ctx, grad_output)
        elif mode_str == "DIFF":
            print("MojoRMSNormFunction: comparing REF and STD backward...")
            ref_result = MojoRMSNormFunction.backward_ref(ctx, grad_output)
            impl_result = impl_func(ctx, grad_output)

            torch.testing.assert_close(ref_result[0], impl_result[0])
            torch.testing.assert_close(ref_result[1], impl_result[1])

            return impl_result
        else:
            raise ValueError(f"Invalid backward mode {mode_str} for RMSNorm, please check.")
