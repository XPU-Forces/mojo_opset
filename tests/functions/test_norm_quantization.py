import pytest
import torch
import torch.nn.functional as torch_f

from mojo_opset import functions as F
from tests._checks import assert_close

NORM_MODES = [("rms_norm_quant", None), ("layer_norm_quant", None)] + [
    (name, pos) for name in ("residual_add_rms_norm_quant", "residual_add_layer_norm_quant") for pos in ("pre", "post")
]


NORM_SHAPES = [(32, 1024), (64, 8192), (2, 256)]


def check_norm(actual, expected, mode, dtype):
    name, pos = mode
    residual_rms = name == "residual_add_rms_norm_quant"
    q_atol = 2 if residual_rms else 1
    scale_tol = 1e-2 if residual_rms else 1e-3
    assert_close(actual[0], expected[0], dtype, name="quantized", rtol=0, atol=q_atol)
    if pos is not None:
        assert_close(actual[1], expected[1], dtype, name="residual", rtol=scale_tol, atol=scale_tol)
    assert_close(actual[-1], expected[-1], dtype, name="scale", rtol=scale_tol, atol=scale_tol)


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

    call = function(impl)
    public_reference = function("torch")

    def reference(*args):
        actual = public_reference(*args)
        summed = args[0] if pos is None else args[0] + args[1]
        if "rms" in name:
            normed = torch_f.rms_norm(summed.float(), [shape[-1]], weight, eps=1e-05)
        else:
            normed = torch_f.layer_norm(summed.float(), [shape[-1]], weight, bias, eps=1e-05)
        smoothed = normed if smooth_scale is None else normed * smooth_scale.float()
        qmax = torch.finfo(quant_dtype).max if fp8 else 127
        qmin = -qmax if fp8 else -128 if symmetric else 0
        scale = smoothed.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
        quantized = (smoothed / scale).round().clamp(qmin, qmax).to(quant_dtype)
        if not fp8:
            assert_close(actual[0], quantized, atol=0, rtol=0)
        assert_close(actual[-1], scale, atol=0, rtol=0)
        if pos is not None:
            residual_out = normed if pos == "post" and "rms" in name else summed
            assert_close(actual[1], residual_out, atol=0, rtol=0)
        return actual

    return (call, reference, inputs)


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "mode", [pytest.param(mode, marks=pytest.mark.api("functions." + mode[0])) for mode in NORM_MODES]
)
@pytest.mark.parametrize(
    "shape,dtype,smooth,affine,symmetric",
    [(shape, dtype, False, True, True) for shape in NORM_SHAPES for dtype in (torch.float16, torch.bfloat16)]
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
