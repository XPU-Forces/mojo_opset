import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

# (B, H, T, D, activation, dtype); all cases are kept above the 20us perf floor.
RMS_CASES = [
    (2, 32, 2048, 128, "silu", torch.float16),
    (2, 32, 1024, 256, "sigmoid", torch.bfloat16),
    (2, 16, 1024, 512, "sigmoid", torch.float16),
    (1, 16, 1024, 1024, "silu", torch.float16),
    (2, 8, 1024, 1024, "sigmoid", torch.bfloat16),
    (2, 2, 2048, 1024, "sigmoid", torch.float32),
    (4, 8, 1024, 600, "sigmoid", torch.bfloat16),
    (2, 4, 2048, 1200, "silu", torch.bfloat16),
]

# 2D (T, D) cases; all cases are kept above the 20us perf floor.
# (T, D, activation, affine, dtype)
TWO_D_CASES = [
    (65536, 128, "silu", True, torch.float32),
    (8192, 1024, "sigmoid", True, torch.float32),
]


def _tolerances(dtype):
    if dtype == torch.float32:
        return 1e-3, 1e-3
    return 2e-2, 2e-2


def _run_gated_case(accuracy_backend, cls, options, shape, dtype):
    implementation, _, device = accuracy_backend
    actual = cls(**options, device=device, dtype=dtype, implementation=implementation)
    reference = cls(**options, device=device, dtype=dtype, implementation="torch")
    with torch.no_grad():
        for parameter in actual.parameters():
            parameter.normal_()
        reference.load_state_dict(actual.state_dict())

    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    reference_g = g.detach().clone().requires_grad_(True)

    y = actual(x, g)
    reference_y = reference(reference_x, reference_g)
    upstream = torch.randn_like(y)
    actual_grads = torch.autograd.grad(y, (x, g, *actual.parameters()), upstream)
    reference_grads = torch.autograd.grad(
        reference_y, (reference_x, reference_g, *reference.parameters()), upstream
    )

    rtol, atol = _tolerances(dtype)
    assert_close(y, reference_y, dtype=dtype, rtol=rtol, atol=atol, name="y")
    for index, (a, b) in enumerate(zip(actual_grads, reference_grads)):
        assert_close(a, b, dtype=dtype, rtol=rtol, atol=atol, name=f"grad[{index}]")


@pytest.mark.api("modules.FusedRMSNormGated", ops=["rms_norm_gated"])
@pytest.mark.parametrize(
    "B,H,T,D,activation,dtype",
    RMS_CASES,
    ids=[f"B{t[0]}_H{t[1]}_T{t[2]}_D{t[3]}_{t[4]}_{t[5]}" for t in RMS_CASES],
)
@pytest.mark.accuracy
def test_fused_rms_norm_gated_accuracy(accuracy_backend, B, H, T, D, activation, dtype):
    options = dict(
        hidden_size=D,
        elementwise_affine=True,
        activation=activation,
    )
    _run_gated_case(accuracy_backend, modules.FusedRMSNormGated, options, (B, H, T, D), dtype)


@pytest.mark.api("modules.FusedRMSNormGated", ops=["rms_norm_gated"])
@pytest.mark.parametrize(
    "T,D,activation,affine,dtype",
    TWO_D_CASES,
    ids=[f"T{t[0]}_D{t[1]}_{t[2]}_affine{t[3]}_{t[4]}" for t in TWO_D_CASES],
)
@pytest.mark.accuracy
def test_fused_rms_norm_gated_2d_accuracy(accuracy_backend, T, D, activation, affine, dtype):
    options = dict(
        hidden_size=D,
        elementwise_affine=affine,
        activation=activation,
    )
    _run_gated_case(accuracy_backend, modules.FusedRMSNormGated, options, (T, D), dtype)
