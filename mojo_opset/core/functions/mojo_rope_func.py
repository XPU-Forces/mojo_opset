import os
import torch
from typing import Tuple


from ..mojo_function import MojoFuncBase
from ...mojo_utils import get_mojo_exec_mode


class MojoRoPEFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(ctx, q, k, cos, sin):
        pass

    @staticmethod
    def forward_ref(ctx, q, k, cos, sin):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        ctx.save_for_backward(q, k, cos, sin)

        return q_rot, k_rot

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        if MojoRoPEFunction._registry:
            impl_func = MojoRoPEFunction._registry[0][1].forward
        else:
            print("MojoRoPEFunction has NO any registered implementation")
            impl_func = MojoRoPEFunction.forward_ref

        layer_idx = ctx.layer_idx if hasattr(ctx, "layer_idx") else -1
        mode_str = get_mojo_exec_mode(MojoRoPEFunction.__name__, "FWD", layer_idx)

        if mode_str == "STD":
            return impl_func(ctx, q, k, cos, sin)
        elif mode_str == "DUMP":
            MojoRoPEFunction.forward_dump(ctx, q, k, cos, sin)
            return torch.zeros_like(q), torch.zeros_like(k)
        elif mode_str == "REF":
            return MojoRoPEFunction.forward_ref(ctx, q, k, cos, sin)
        elif mode_str == "DIFF":
            ref_q, ref_k = MojoRoPEFunction.forward_ref(ctx, q, k, cos, sin)
            impl_q, impl_k = impl_func(ctx, q, k, cos, sin)

            torch.testing.assert_close(ref_q, impl_q)
            torch.testing.assert_close(ref_k, impl_k)
            return impl_q, impl_k
        else:
            raise ValueError(f"Invalid forward mode {mode_str} for RoPE, please check.")

    @staticmethod
    def backward_dump(ctx, grad_output_q, grad_output_k):
        pass

    @staticmethod
    def backward_ref(ctx, grad_output_q, grad_output_k):
        q, k, cos, sin = ctx.saved_tensors

        def inverse_rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((x2, -x1), dim=-1)

        grad_q_part1 = grad_output_q * cos

        grad_q_part2 = inverse_rotate_half(grad_output_q * sin)
        grad_q = grad_q_part1 + grad_q_part2

        grad_k_part1 = grad_output_k * cos

        grad_k_part2 = inverse_rotate_half(grad_output_k * sin)
        grad_k = grad_k_part1 + grad_k_part2

        return grad_q, grad_k, None, None

    @staticmethod
    def backward(ctx, grad_output_q, grad_output_k):
        if MojoRoPEFunction._registry:
            impl_func = MojoRoPEFunction._registry[0][1].backward
        else:
            print("MojoRoPEFunction has NO any registered implementation")
            impl_func = MojoRoPEFunction.backward_ref

        mode_str = os.environ.get(f"{MojoRoPEFunction.__name__.upper()}_BWD_MODE", "STD")

        if mode_str == "STD":
            return impl_func(ctx, grad_output_q, grad_output_k)
        elif mode_str == "DUMP":
            MojoRoPEFunction.backward_dump(ctx, grad_output_q, grad_output_k)
            grad_q = torch.zeros_like(ctx.saved_tensors[0])
            grad_k = torch.zeros_like(ctx.saved_tensors[1])
            return grad_q, grad_k, None, None
        elif mode_str == "REF":
            return MojoRoPEFunction.backward_ref(ctx, grad_output_q, grad_output_k)
        elif mode_str == "DIFF":
            print("MojoRoPEFunction: comparing REF and STD backward...")
            ref_grad_q, ref_grad_k, _, _ = MojoRoPEFunction.backward_ref(ctx, grad_output_q, grad_output_k)
            impl_grad_q, impl_grad_k, _, _ = impl_func(ctx, grad_output_q, grad_output_k)

            torch.testing.assert_close(ref_grad_q, impl_grad_q)
            torch.testing.assert_close(ref_grad_k, impl_grad_k)

            return impl_grad_q, impl_grad_k, None, None
        else:
            raise ValueError(f"Invalid backward mode {mode_str} for RoPE, please check.")
