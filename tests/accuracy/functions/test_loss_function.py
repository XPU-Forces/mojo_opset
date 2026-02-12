import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import bypass_not_implemented

from mojo_opset import MojoFusedLinearCrossEntropyFunction


@pytest.mark.parametrize(
    "batch, hidden_size, target_size",
    [(2048, 1024, 4096)],
)
@pytest.mark.parametrize(
    "has_bias, has_ce_weight, ignore_index, label_smoothing, lse_square_scale, return_z_loss",
    [
        (False, False, -100, 0.0, 0.0, False),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@bypass_not_implemented
def test_fused_ce_forward_backward_diff(
    batch,
    hidden_size,
    target_size,
    has_bias,
    has_ce_weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    return_z_loss,
):
    input_tensor = torch.randn(batch, hidden_size, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(target_size, hidden_size, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, target_size, (batch,), dtype=torch.long)
    bias = None

    ce_weight = None
    if has_ce_weight:
        ce_weight = torch.rand(weight.shape[0], device=weight.device, dtype=torch.float32) + 0.1

    ctx = MockFunctionCtx()
    output = MojoFusedLinearCrossEntropyFunction.forward(
        ctx,
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        None,
        return_z_loss,
        None,
    )

    ctx_ref = MockFunctionCtx()
    output_ref = MojoFusedLinearCrossEntropyFunction._registry.get("torch").forward(
        ctx_ref,
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        None,
        return_z_loss,
        None,
    )

    assert_close(output, output_ref)

    loss, z_loss = output
    if return_z_loss:
        grad_z_loss = torch.rand_like(z_loss)
    else:
        grad_z_loss = None

    if reduction == "mean":
        grad_output = torch.rand_like(loss)
    else:
        grad_output = torch.rand_like(loss) / input_tensor.shape[0]

    grad = MojoFusedLinearCrossEntropyFunction.backward(
        ctx,
        grad_output,
        grad_z_loss,
    )

    grad_ref = MojoFusedLinearCrossEntropyFunction._registry.get("torch").backward(
        ctx_ref,
        grad_output,
        grad_z_loss,
    )

    assert_close(grad, grad_ref)
