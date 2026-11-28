"""Tests for modules.GmmSwigluQuant (forward only, MXFP8 in/out)."""

import pytest
import torch

from mojo_opset import modules
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize
from tests.functions.test_gmm_swiglu_quant import _assert_quantized_close
from tests.functions.test_gmm_swiglu_quant import _reference_silu_half


def _require_fp8(target):
    """float8_e4m3fn (MX path) is A5-only on NPU; 910b (A2) has no FP8 support."""
    if target.startswith("npu.") and not target.startswith("npu.a5"):
        pytest.skip(f"GmmSwigluQuant requires FP8 support (A5 MX path); got {target}")
# glm-5.3 / glm-5.3-flash MLP + shared-expert shapes; first case is tail rows.
GMM_SWIGLU_QUANT_MODULE_CASES = [
    (7, 512, 320),  # tail shapes
    (64, 6144, 2048),  # glm-5.3 shared experts
    (64, 4096, 2048),  # glm-5.3-flash shared experts
    (5, 6144, 12288),  # glm-5.3 dense MLP
    (5, 4096, 12288),  # glm-5.3-flash dense MLP
]


def _reference_forward(x, weight, x_scale, weight_scale, swiglu_limit):
    """Independent oracle: FP32 dequant-GEMM + SwiGLU, pre-quantization S."""
    return _reference_silu_half(x, weight, x_scale, weight_scale, swiglu_limit)


@pytest.mark.api("modules.GmmSwigluQuant", ops=["gmm_swiglu_quant"])
@pytest.mark.parametrize("hidden,intermediate", [case[1:] for case in GMM_SWIGLU_QUANT_MODULE_CASES])
@pytest.mark.parametrize("limit", [0.0, 10.0])
@pytest.mark.accuracy
def test_gmm_swiglu_quant_module_accuracy(accuracy_backend, hidden, intermediate, limit):
    implementation, target, device = accuracy_backend
    _require_fp8(target)
    module = modules.GmmSwigluQuant(hidden, intermediate, limit, implementation=implementation, device=device)

    tokens = 7 if hidden == 512 else 5
    x, x_scale = mxfp8_quantize(torch.randn(tokens, hidden, device=device))

    actual = module(x, x_scale)
    expected = _reference_forward(x, module.weight, x_scale, module.weight_scale, limit)
    _assert_quantized_close(actual[0], actual[1], expected)


@pytest.mark.api("modules.GmmSwigluQuant", ops=["gmm_swiglu_quant"])
@pytest.mark.accuracy
def test_gmm_swiglu_quant_module_state_dict_roundtrip(accuracy_backend):
    """Buffers round-trip through state_dict and the module matches the function."""
    implementation, target, device = accuracy_backend
    _require_fp8(target)
    hidden, intermediate = 64, 48
    module = modules.GmmSwigluQuant(hidden, intermediate, swiglu_limit=10.0,
                                    implementation=implementation, device=device)
    clone = modules.GmmSwigluQuant(hidden, intermediate, device=device)
    clone.load_state_dict(module.state_dict(), strict=True, assign=True)
    assert torch.equal(clone.weight.view(torch.uint8), module.weight.view(torch.uint8))
    assert torch.equal(clone.weight_scale, module.weight_scale)

    x, x_scale = mxfp8_quantize(torch.randn(3, hidden, device=device))
    # Module and direct function call agree bitwise; reference within quant steps.
    module_out = module(x, x_scale)
    clone_out = clone(x, x_scale)
    assert torch.equal(clone_out[0].view(torch.uint8), module_out[0].view(torch.uint8))
    assert torch.equal(clone_out[1], module_out[1])
    expected = _reference_forward(x, module.weight, x_scale, module.weight_scale, 10.0)
    _assert_quantized_close(clone_out[0], clone_out[1], expected)
