import os

import torch
from ..mojo_function import MojoFuncBase
from ...mojo_utils import get_mojo_exec_mode
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoSiluFunction(MojoFuncBase):
    """
    MojoSiluFunction is the base class for all MojoSiluFunction.
    """

    @staticmethod
    def forward_dump(ctx, input):
        pass

    @staticmethod
    def forward_ref(ctx, input):
        sigmoid_x = torch.sigmoid(input)
        ctx.save_for_backward(input)
        return input * sigmoid_x

    @staticmethod
    def forward(ctx, input):
        if MojoSiluFunction._registry:
            impl_func = MojoSiluFunction._registry[0][1].forward
        else:
            logger.warning("MojoSiluFunction has NO any registered implementation")

        layer_idx = ctx.layer_idx if hasattr(ctx, "layer_idx") else -1
        mode_str = get_mojo_exec_mode(MojoSiluFunction.__name__, "FWD", layer_idx)

        if mode_str == "STD":
            return impl_func(ctx, input)
        elif mode_str == "DUMP":
            return MojoSiluFunction.forward_dump(ctx, input)
        elif mode_str == "REF":
            return MojoSiluFunction.forward_ref(ctx, input)
        elif mode_str == "DIFF":
            ref_result = MojoSiluFunction.forward_ref(ctx, input)
            rel_result = impl_func(ctx, input)
            torch.testing.assert_close(ref_result, rel_result)
            return rel_result
        else:
            raise ValueError(f"Invalid forward mode {mode_str}, please check the operator implementation.")

    @staticmethod
    def backward_dump(ctx, grad_output):
        pass

    @staticmethod
    def backward_ref(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output * torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))
        return grad_input

    @staticmethod
    def backward_diff(ctx, grad_output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        if MojoSiluFunction._registry:
            impl_func = MojoSiluFunction._registry[0][1].backward
        else:
            logger.warning("MojoSiluFunction has NO any registered implementation")
        mode_str = os.environ.get(f"{MojoSiluFunction.__name__.upper()}_BWD_MODE", "STD")

        if mode_str == "STD":
            return impl_func(ctx, grad_output)
        elif mode_str == "DUMP":
            return MojoSiluFunction.backward_dump(ctx, grad_output)
        elif mode_str == "REF":
            return MojoSiluFunction.backward_ref(ctx, grad_output)
        elif mode_str == "DIFF":
            ref_result = MojoSiluFunction.backward_ref(ctx, grad_output)
            rel_result = impl_func(ctx, grad_output)
            torch.testing.assert_close(ref_result, rel_result)
            return rel_result
        else:
            raise ValueError(f"Invalid backward mode {mode_str}, please check the operator implementation.")
