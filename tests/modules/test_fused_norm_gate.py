import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

# (B, H, T, D, affine, bias, activation, dtype)
LN_CASES = [
    (1, 1, 1, 64, False, False, "silu", torch.float32),
    (2, 2, 50, 50, False, False, "sigmoid", torch.float32),
    (2, 2, 512, 128, True, True, "silu", torch.bfloat16),
    (2, 2, 1024, 600, True, False, "sigmoid", torch.float16),
    (2, 2, 2048, 1200, True, False, "sigmoid", torch.bfloat16),
    (4, 4, 256, 256, True, True, "silu", torch.float16),
    (2, 8, 1024, 512, True, False, "sigmoid", torch.bfloat16),
    (2, 4, 512, 768, True, True, "silu", torch.float32),
    (4, 4, 2048, 1024, True, False, "sigmoid", torch.bfloat16),
]

# (B, H, T, D, activation, dtype)
RMS_CASES = [
    (1, 1, 1, 64, "silu", torch.float32),
    (2, 2, 512, 128, "silu", torch.bfloat16),
    (2, 2, 1024, 600, "sigmoid", torch.float16),
    (2, 2, 2048, 1200, "sigmoid", torch.bfloat16),
    (2, 8, 1024, 512, "sigmoid", torch.bfloat16),
    (4, 4, 2048, 1024, "sigmoid", torch.float32),
]

# 2D (T, D) cases mirroring the K3 KDA o_norm shape family.
# (T, D, norm_type, activation, affine, bias, dtype)
TWO_D_CASES = [
    (4096, 128, "layer", "silu", True, True, torch.float32),
    (8192, 256, "layer", "sigmoid", True, False, torch.float32),
    (4096, 512, "layer", "silu", True, True, torch.float32),
    (2048, 1024, "layer", "sigmoid", True, False, torch.float32),
    (4096, 128, "rms", "silu", True, False, torch.float32),
    (2048, 1024, "rms", "sigmoid", True, False, torch.float32),
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


@pytest.mark.api("modules.FusedLayerNormGated", ops=["layer_norm_gated"])
@pytest.mark.parametrize(
    "B,H,T,D,affine,bias,activation,dtype",
    LN_CASES,
    ids=[
        f"B{t[0]}_H{t[1]}_T{t[2]}_D{t[3]}_affine{t[4]}_bias{t[5]}_{t[6]}_{t[7]}"
        for t in LN_CASES
    ],
)
@pytest.mark.accuracy
def test_fused_layer_norm_gated_accuracy(accuracy_backend, B, H, T, D, affine, bias, activation, dtype):
    options = dict(
        hidden_size=D,
        elementwise_affine=affine,
        bias=bias,
        activation=activation,
    )
    _run_gated_case(accuracy_backend, modules.FusedLayerNormGated, options, (B, H, T, D), dtype)


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


@pytest.mark.api("modules.FusedLayerNormGated", ops=["layer_norm_gated"])
@pytest.mark.api("modules.FusedRMSNormGated", ops=["rms_norm_gated"])
@pytest.mark.parametrize(
    "T,D,norm_type,activation,affine,bias,dtype",
    TWO_D_CASES,
    ids=[
        f"T{t[0]}_D{t[1]}_{t[2]}_{t[3]}_affine{t[4]}_bias{t[5]}_{t[6]}"
        for t in TWO_D_CASES
    ],
)
@pytest.mark.accuracy
def test_fused_norm_gate_2d_accuracy(accuracy_backend, T, D, norm_type, activation, affine, bias, dtype):
    if norm_type == "layer":
        cls = modules.FusedLayerNormGated
        options = dict(
            hidden_size=D,
            elementwise_affine=affine,
            bias=bias,
            activation=activation,
        )
    else:
        cls = modules.FusedRMSNormGated
        options = dict(
            hidden_size=D,
            elementwise_affine=affine,
            activation=activation,
        )
    _run_gated_case(accuracy_backend, cls, options, (T, D), dtype)
