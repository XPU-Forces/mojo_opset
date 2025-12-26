import pytest
import torch

from tests.utils import auto_switch_platform


@pytest.mark.parametrize(
    "input_tensor, weight, target, bias, grad_input",
    [
        (
            torch.randn(2048, 1024, dtype=torch.bfloat16),
            torch.randn(4096, 1024, dtype=torch.bfloat16),
            torch.randint(0, 4096, (2048,), dtype=torch.long),
            None,
            torch.randn(2048, 1024, dtype=torch.bfloat16),
        )
    ],
)
@pytest.mark.parametrize(
    "has_bias, has_ce_weight, ignore_index, label_smoothing, lse_square_scale, reduction, return_z_loss",
    [
        (False, False, -100, 0.0, 0.0, "mean", False),
        (False, False, -100, 0.0, 0.0, "sum", False),
        (False, False, -100, 0.0, 0.0, "none", False),
    ],
)
@auto_switch_platform()
def test_fused_ce(
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
    grad_input,
):
    ce_weight = None
    if has_ce_weight:
        ce_weight = torch.rand(weight.shape[0], device=weight.device, dtype=torch.float32) + 0.1

    if reduction == "none":
        torch.library.opcheck(
            torch.ops.ttx.fused_linear_cross_entropy_1d,
            (
                input_tensor,
                weight,
                target,
                ce_weight,
                bias,
                ignore_index,
                lse_square_scale,
                label_smoothing,
                None,
                return_z_loss,
            ),
        )
        torch.library.opcheck(
            torch.ops.ttx.fused_linear_cross_entropy_1d_bwd,
            (
                torch.full((input_tensor.shape[0],), 1.0, dtype=grad_input.dtype, device=grad_input.device),
                input_tensor,
                weight,
                target,
                ce_weight,
                bias,
                ignore_index,
                lse_square_scale,
                label_smoothing,
                None,
                None,
            ),
            # NOTE: Since we directly register the backward operator itself, the
            # test_aot_dispatch_dynamic check is unnecessary.
            test_utils=("test_schema", "test_autograd_registration", "test_faketensor"),
        )
        return

    torch.library.opcheck(
        torch.ops.ttx.fused_linear_cross_entropy,
        (
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
        ),
    )

    torch.library.opcheck(
        torch.ops.ttx.fused_linear_cross_entropy_bwd,
        (
            torch.tensor(1.0, device=grad_input.device),
            grad_input,
            None,
            None,
        ),
        # NOTE: Since we directly register the backward operator itself, the
        # test_aot_dispatch_dynamic check is unnecessary.
        test_utils=("test_schema", "test_autograd_registration", "test_faketensor"),
    )
