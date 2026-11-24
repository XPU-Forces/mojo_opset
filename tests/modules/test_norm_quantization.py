from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules as M
from tests._checks import assert_close

NORM_MODES = [("rms_norm_quant", None), ("layer_norm_quant", None)] + [
    (name, pos) for name in ("residual_add_rms_norm_quant", "residual_add_layer_norm_quant") for pos in ("pre", "post")
]


def check_norm(actual, expected, mode, dtype):
    name, pos = mode
    residual_rms = name == "residual_add_rms_norm_quant"
    q_atol = 2 if residual_rms else 1
    scale_tol = 1e-2 if residual_rms else 1e-3
    assert_close(actual[0], expected[0], dtype, name="quantized", rtol=0, atol=q_atol)
    if pos is not None:
        assert_close(actual[1], expected[1], dtype, name="residual", rtol=scale_tol, atol=scale_tol)
    assert_close(actual[-1], expected[-1], dtype, name="scale", rtol=scale_tol, atol=scale_tol)


MODULE_NAMES = {
    "rms_norm_quant": "RMSNormQuant",
    "layer_norm_quant": "LayerNormQuant",
    "residual_add_rms_norm_quant": "ResidualAddRMSNormQuant",
    "residual_add_layer_norm_quant": "ResidualAddLayerNormQuant",
}


def norm_case(shape, dtype, mode, device, impl, *, fp8=False, smooth=False, affine=True, symmetric=True):
    name, pos = mode
    x = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(shape[-1], device=device) if affine or "rms" in name else None
    bias = torch.randn(shape[-1], device=device) if "layer" in name and affine else None
    smooth_scale = torch.rand(shape[-1], device=device) + 0.1 if smooth else None
    residual = torch.randn_like(x) if pos is not None else None
    quant_dtype = torch.float8_e4m3fn if fp8 else torch.int8
    options = dict(quant_dtype=quant_dtype, symmetric=symmetric)
    if pos is not None:
        options["norm_pos"] = pos
    inputs = (x, residual) if pos is not None else (x,)

    def function(implementation):

        def call(*args):
            x, *rest = args
            params = [x]
            if pos is not None:
                params.append(rest.pop(0))
            params.append(weight)
            if "layer" in name:
                params.append(bias)
            params.append(smooth_scale)
            return getattr(F, name)(*params, **options, implementation=implementation)

        return call

    init = dict(options)
    if "layer" in name:
        init["elementwise_affine"] = affine
    call = getattr(M, MODULE_NAMES[name])(shape[-1], **init, implementation=impl, device=device)
    state = {key: value for key, value in (("weight", weight), ("bias", bias)) if value is not None}
    call.load_state_dict(state)
    call = partial(call, smooth_scale=smooth_scale)
    public_reference = function("torch")

    return call, public_reference, inputs


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(mode, marks=pytest.mark.api("modules." + MODULE_NAMES[mode[0]], ops=[mode[0]]))
        for mode in NORM_MODES
    ],
)
@pytest.mark.parametrize(
    "shape,dtype,smooth,affine,symmetric",
    [(shape, dtype, False, True, True) for shape in [(2, 256)] for dtype in (torch.float16, torch.bfloat16)]
    + [
        ((2, 256), torch.float16, True, affine, symmetric)
        for affine, symmetric in ((True, True), (False, True), (True, False))
    ],
)
def test_norm_quantization_accuracy(accuracy_backend, mode, shape, dtype, smooth, affine, symmetric):
    impl, _, device = accuracy_backend
    call, reference, inputs = norm_case(
        shape, dtype, mode, device, impl, smooth=smooth, affine=affine, symmetric=symmetric
    )
    check_norm(call(*inputs), reference(*inputs), mode, dtype)


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", NORM_MODES)
def test_norm_quantization_fp8_reference_output_metadata(dtype, mode):
    shape = (2, 256)
    # The original FP8 norm-quant tests run on CPU, not unsupported NPU kernels.
    call, _, inputs = norm_case(shape, dtype, mode, "cpu", "torch", fp8=True)
    output = call(*inputs)
    assert output[0].dtype == torch.float8_e4m3fn
    assert output[0].shape == shape and output[-1].shape == (*shape[:-1], 1)
