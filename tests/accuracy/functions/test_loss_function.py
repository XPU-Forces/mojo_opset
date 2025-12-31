import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented
from tests.utils import assert_close
from tests.utils import MockFunctionCtx

from mojo_opset import MojoFusedLinearCrossEntropyFunction


@pytest.mark.parametrize(
    "input_tensor, weight, target, bias",
    [
        (
            torch.randn(2048, 1024, dtype=torch.bfloat16, requires_grad=True),
            torch.randn(4096, 1024, dtype=torch.bfloat16, requires_grad=True),
            torch.randint(0, 4096, (2048,), dtype=torch.long),
            None,
        )
    ],
)
@pytest.mark.parametrize(
    "has_bias, has_ce_weight, ignore_index, label_smoothing, lse_square_scale, reduction, return_z_loss",
    [
        (False, False, -100, 0.0, 0.0, "mean", False),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_fused_ce_forward_backward_diff(
    # monkeypatch,
    input_tensor,
    weight,
    target,
    bias,
    has_bias,
    has_ce_weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    return_z_loss,
):
    # monkeypatch.setenv("MOJOFUSEDLINEARCROSSENTROPYFUNCTION_FWD_MODE", "DIFF")
    # monkeypatch.setenv("MOJOFUSEDLINEARCROSSENTROPYFUNCTION_BWD_MODE", "DIFF")

    ce_weight = None
    if has_ce_weight:
        ce_weight = torch.rand(weight.shape[0], device=weight.device, dtype=torch.float32) + 0.1

    ctx = MockFunctionCtx()
    loss = MojoFusedLinearCrossEntropyFunction.forward(
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
    loss_ref = MojoFusedLinearCrossEntropyFunction._registry.get("ref").forward(
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
    
    assert_close(loss, loss_ref)

    if return_z_loss:
        grad_output = torch.rand_like(loss[0])
        grad_z_loss = torch.rand_like(loss[1])
    else:
        grad_output = torch.rand_like(loss)
        grad_z_loss = None

    grad = MojoFusedLinearCrossEntropyFunction.backward(
        ctx,
        grad_output,
        grad_z_loss,
    )

    grad_ref = MojoFusedLinearCrossEntropyFunction._registry.get("ref").backward(
        ctx_ref,
        grad_output,
        grad_z_loss,
    )

    assert_close(grad, grad_ref)