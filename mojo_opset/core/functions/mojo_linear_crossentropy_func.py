import os
import torch
from ..mojo_function import MojoFuncBase
from ...mojo_utils import get_mojo_exec_mode
import torch.nn.functional as F


class MojoFusedLinearCrossEntropyFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(
        ctx,
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        label_smoothing,
        reduction,
        return_z_loss,
        lse_square_scale,
    ):
        pass

    @staticmethod
    def forward_ref(
        ctx,
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        label_smoothing,
        reduction,
        return_z_loss,
        lse_square_scale,
    ):
        logits_ref = F.linear(input_tensor, weight, bias).float()

        loss_ref = F.cross_entropy(
            logits_ref,
            target,
            weight=ce_weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

        z_loss_ref = None
        if return_z_loss:
            with torch.no_grad():
                valid_logits = logits_ref[target != ignore_index]
                if valid_logits.numel() > 0:
                    lse_ref = torch.logsumexp(valid_logits, dim=-1)

                    z_loss_calc = lse_square_scale * torch.sum(lse_ref * lse_ref) / (target != ignore_index).sum()
                    loss_ref += z_loss_calc
                    z_loss_ref = z_loss_calc

        ctx.save_for_backward(input_tensor, weight, target, bias, ce_weight)
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.reduction = reduction
        ctx.return_z_loss = return_z_loss
        ctx.lse_square_scale = lse_square_scale

        if return_z_loss:
            if z_loss_ref is None:
                z_loss_ref = torch.tensor(0.0, device=loss_ref.device, dtype=loss_ref.dtype)
            return loss_ref, z_loss_ref
        else:
            return loss_ref

    @staticmethod
    def forward(
        ctx,
        input_tensor,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        label_smoothing=0.0,
        reduction="mean",
        return_z_loss=False,
        lse_square_scale=0.0,
    ):
        print("MojoFusedLinearCrossEntropyFunction forward dispatch")

        if MojoFusedLinearCrossEntropyFunction._registry:
            impl_func = MojoFusedLinearCrossEntropyFunction._registry[0][1].forward
        else:
            print("MojoFusedLinearCrossEntropyFunction has NO any registered implementation")
            impl_func = MojoFusedLinearCrossEntropyFunction.forward_ref

        layer_idx = ctx.layer_idx if hasattr(ctx, "layer_idx") else -1
        mode_str = get_mojo_exec_mode(MojoFusedLinearCrossEntropyFunction.__name__, "FWD", layer_idx)

        args = (
            ctx,
            input_tensor,
            weight,
            target,
            bias,
            ce_weight,
            ignore_index,
            label_smoothing,
            reduction,
            return_z_loss,
            lse_square_scale,
        )

        if mode_str == "STD":
            return impl_func(*args)
        elif mode_str == "DUMP":
            MojoFusedLinearCrossEntropyFunction.forward_dump(*args)

            dummy_loss = torch.tensor(0.0, device=input_tensor.device, dtype=torch.float32)
            if return_z_loss:
                dummy_z_loss = torch.tensor(0.0, device=input_tensor.device, dtype=torch.float32)
                return dummy_loss, dummy_z_loss
            return dummy_loss
        elif mode_str == "REF":
            return MojoFusedLinearCrossEntropyFunction.forward_ref(*args)
        elif mode_str == "DIFF":
            ref_result = MojoFusedLinearCrossEntropyFunction.forward_ref(*args)
            impl_result = impl_func(*args)

            if return_z_loss:
                torch.testing.assert_close(ref_result[0], impl_result[0])
                torch.testing.assert_close(ref_result[1], impl_result[1])
            else:
                torch.testing.assert_close(ref_result, impl_result)
            return impl_result
        else:
            raise ValueError(f"Invalid forward mode {mode_str} for FusedLinearCrossEntropy, please check.")

    @staticmethod
    def backward_dump(ctx, grad_loss, grad_z_loss):
        pass

    @staticmethod
    def backward_ref(ctx, grad_loss, grad_z_loss):
        input_tensor, weight, target, bias, ce_weight = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        reduction = ctx.reduction
        return_z_loss = ctx.return_z_loss
        lse_square_scale = ctx.lse_square_scale

        input_with_grad = input_tensor.detach().clone().requires_grad_(True)
        weight_with_grad = weight.detach().clone().requires_grad_(True)
        bias_with_grad = bias.detach().clone().requires_grad_(True) if bias is not None else None

        with torch.enable_grad():
            logits_ref = F.linear(input_with_grad, weight_with_grad, bias_with_grad).float()

            loss_ref = F.cross_entropy(
                logits_ref,
                target,
                weight=ce_weight,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )

            if return_z_loss:
                valid_logits = logits_ref[target != ignore_index]
                if valid_logits.numel() > 0:
                    lse_ref = torch.logsumexp(valid_logits, dim=-1)
                    z_loss_calc = lse_square_scale * torch.sum(lse_ref * lse_ref) / (target != ignore_index).sum()
                    loss_ref += z_loss_calc

        total_grad = grad_loss
        if return_z_loss and grad_z_loss is not None:
            total_grad += grad_z_loss

        loss_ref.backward(gradient=total_grad)

        grad_input = input_with_grad.grad
        grad_weight = weight_with_grad.grad
        grad_bias = bias_with_grad.grad if bias_with_grad is not None else None

        return grad_input, grad_weight, None, grad_bias, None, None, None, None, None, None

    @staticmethod
    def backward(ctx, grad_loss, grad_z_loss=None):
        print("MojoFusedLinearCrossEntropyFunction backward dispatch")

        if MojoFusedLinearCrossEntropyFunction._registry:
            impl_func = MojoFusedLinearCrossEntropyFunction._registry[0][1].backward
        else:
            print("MojoFusedLinearCrossEntropyFunction has NO any registered implementation")
            impl_func = MojoFusedLinearCrossEntropyFunction.backward_ref

        mode_str = os.environ.get(f"{MojoFusedLinearCrossEntropyFunction.__name__.upper()}_BWD_MODE", "STD")

        args = (ctx, grad_loss, grad_z_loss)

        if mode_str == "STD":
            return impl_func(*args)
        elif mode_str == "DUMP":
            MojoFusedLinearCrossEntropyFunction.backward_dump(*args)

            grad_input = torch.zeros_like(ctx.saved_tensors[0])
            grad_weight = torch.zeros_like(ctx.saved_tensors[1])
            grad_bias = torch.zeros_like(ctx.saved_tensors[3]) if ctx.saved_tensors[3] is not None else None
            return grad_input, grad_weight, None, grad_bias, None, None, None, None, None, None
        elif mode_str == "REF":
            return MojoFusedLinearCrossEntropyFunction.backward_ref(*args)
        elif mode_str == "DIFF":
            print("MojoFusedLinearCrossEntropyFunction: comparing REF and STD backward...")
            ref_results = MojoFusedLinearCrossEntropyFunction.backward_ref(*args)
            impl_results = impl_func(*args)

            torch.testing.assert_close(ref_results[0], impl_results[0])
            torch.testing.assert_close(ref_results[1], impl_results[1])
            if ref_results[3] is not None:
                torch.testing.assert_close(ref_results[3], impl_results[3])

            return impl_results
        else:
            raise ValueError(f"Invalid backward mode {mode_str} for FusedLinearCrossEntropy, please check.")
