from functools import partial

import pytest
import torch
import torch.nn.functional as torch_f

from mojo_opset import functions as F
from mojo_opset import modules as M
from tests._checks import assert_close

MOE_CASES = [
    (8, 128, [8]),
    (12, 256, [4, 3, 5]),
    (18, 512, [6, 6, 4, 2]),
    (21, 1024, [2, 5, 1, 7, 6]),
    (32, 2048, [8, 7, 5, 6, 4, 2]),
]


NORM_MODES = [("rms_norm_quant", None), ("layer_norm_quant", None)] + [
    (name, pos) for name in ("residual_add_rms_norm_quant", "residual_add_layer_norm_quant") for pos in ("pre", "post")
]


NORM_SHAPES = [(32, 1024), (64, 8192), (2, 256)]


SWIGLU_CASES = [(12, 64, [4, 3, 5]), (20, 128, [6, 4, 7, 3]), (24, 256, [5, 8, 4, 7]), (30, 512, [6, 3, 8, 5, 8])]


def check_grouped(actual, expected, *, swiglu=False):
    q_atol, scale_tol = (0, 1e-4) if swiglu else (1, 2e-3)
    assert_close(actual[0], expected[0], torch.float32, name="quantized", rtol=0, atol=q_atol)
    assert_close(actual[1], expected[1], torch.float32, name="scale", rtol=scale_tol, atol=scale_tol)


def check_norm(actual, expected, mode, dtype):
    name, pos = mode
    residual_rms = name == "residual_add_rms_norm_quant"
    q_atol = 2 if residual_rms else 1
    scale_tol = 1e-2 if residual_rms else 1e-3
    assert_close(actual[0], expected[0], dtype, name="quantized", rtol=0, atol=q_atol)
    if pos is not None:
        assert_close(actual[1], expected[1], dtype, name="residual", rtol=scale_tol, atol=scale_tol)
    assert_close(actual[-1], expected[-1], dtype, name="scale", rtol=scale_tol, atol=scale_tol)


def moe_case(case, dtype, device, impl, module=False):
    tokens, width, counts = case
    x = torch.randn(tokens, width, dtype=dtype, device=device)
    token_count = torch.tensor(counts, dtype=torch.int32, device=device)
    inv_smooth_scale = (torch.rand(len(counts), width, device=device) + 0.1).reciprocal()
    options = dict(inv_smooth_scale=inv_smooth_scale)
    if module:
        call = M.MoEDynamicQuant(len(counts), width, implementation=impl, device=device)
        call.load_state_dict(options)
    else:
        call = partial(F.moe_dynamic_quant, **options, implementation=impl)
    reference = partial(F.moe_dynamic_quant, **options, implementation="torch_reference")
    return call, reference, (x, token_count)


MODULE_NAMES = {
    "rms_norm_quant": "RMSNormQuant",
    "layer_norm_quant": "LayerNormQuant",
    "residual_add_rms_norm_quant": "ResidualAddRMSNormQuant",
    "residual_add_layer_norm_quant": "ResidualAddLayerNormQuant",
}


def norm_case(shape, dtype, mode, device, impl, module=False, *, fp8=False, smooth=False, affine=True, symmetric=True):
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

    if module:
        init = dict(options)
        if "layer" in name:
            init["elementwise_affine"] = affine
        call = getattr(M, MODULE_NAMES[name])(shape[-1], **init, implementation=impl, device=device)
        state = {key: value for key, value in (("weight", weight), ("bias", bias)) if value is not None}
        call.load_state_dict(state)
        call = partial(call, smooth_scale=smooth_scale)
    else:
        call = function(impl)
    public_reference = function("torch_reference")

    def reference(*args):
        actual = public_reference(*args)
        summed = args[0] if pos is None else args[0] + args[1]
        if "rms" in name:
            normed = torch_f.rms_norm(summed.float(), [shape[-1]], weight, eps=1e-5)
        else:
            normed = torch_f.layer_norm(summed.float(), [shape[-1]], weight, bias, eps=1e-5)
        smoothed = normed if smooth_scale is None else normed * smooth_scale.float()
        qmax = torch.finfo(quant_dtype).max if fp8 else 127
        qmin = -qmax if fp8 else (-128 if symmetric else 0)
        scale = smoothed.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
        quantized = (smoothed / scale).round().clamp(qmin, qmax).to(quant_dtype)
        if not fp8:
            torch.testing.assert_close(actual[0], quantized, atol=0, rtol=0)
        torch.testing.assert_close(actual[-1], scale, atol=0, rtol=0)
        if pos is not None:
            residual_out = normed if pos == "post" and "rms" in name else summed
            torch.testing.assert_close(actual[1], residual_out, atol=0, rtol=0)
        return actual

    return call, reference, inputs


def supported(impl, target, *, op=None):
    providers = (
        (None, "torch_reference", "torch_npu", "triton")
        if op == "moe_dynamic_quant"
        else (None, "torch_reference", "torch_npu")
    )
    if impl not in providers or (impl != "torch_reference" and not target.startswith("npu.")):
        pytest.skip("no migrated quantization provider for this implementation/target")


def swiglu_case(case, device, impl, module=False, *, activate_left=False, with_bias=False):
    tokens, width, counts = case
    x = torch.randint(-1024, 1024, (tokens, width * 2), dtype=torch.int32, device=device)
    token_count = torch.tensor(counts, dtype=torch.int64, device=device)
    activation_scale = torch.rand(tokens, device=device)
    weight_scale = torch.rand(len(counts), width * 2, device=device)
    quant_scale = torch.rand(len(counts), width, device=device)
    bias = torch.randn(len(counts), width * 2, device=device) if with_bias else None
    params = dict(weight_scale=weight_scale, quant_scale=quant_scale)
    options = dict(activate_left=activate_left)
    if module:
        call = M.DequantSwiGLUQuant(len(counts), width, **options, implementation=impl, device=device)
        call.load_state_dict(params)
    else:

        def call(x, activation_scale, bias, quant_offset, token_count):
            return F.dequant_swiglu_quant(
                x,
                **params,
                activation_scale=activation_scale,
                bias=bias,
                quant_offset=quant_offset,
                token_count=token_count,
                **options,
                implementation=impl,
            )

    def reference(x, activation_scale, bias, quant_offset, token_count):
        return F.dequant_swiglu_quant(
            x,
            **params,
            activation_scale=activation_scale,
            bias=bias,
            quant_offset=quant_offset,
            token_count=token_count,
            **options,
            implementation="torch_reference",
        )

    return (
        partial(call, bias=bias, quant_offset=None, token_count=token_count),
        partial(reference, bias=bias, quant_offset=None, token_count=token_count),
        (x, activation_scale),
    )


@pytest.mark.api("functions.rms_norm_quant")
@pytest.mark.api("functions.layer_norm_quant")
@pytest.mark.api("functions.residual_add_rms_norm_quant")
@pytest.mark.api("functions.residual_add_layer_norm_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", NORM_MODES)
def test_norm(accuracy_backend, shape, dtype, mode):
    impl, target, device = accuracy_backend
    supported(impl, target)
    call, reference, inputs = norm_case(shape, dtype, mode, device, impl, False)
    check_norm(call(*inputs), reference(*inputs), mode, dtype)


@pytest.mark.api("functions.moe_dynamic_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", MOE_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cpu_counts", [False, True], ids=["device_counts", "cpu_counts"])
def test_moe(accuracy_backend, case, dtype, cpu_counts):
    impl, target, device = accuracy_backend
    supported(impl, target, op="moe_dynamic_quant")
    call, reference, inputs = moe_case(case, dtype, device, impl, False)
    if cpu_counts:
        inputs = (inputs[0], inputs[1].cpu())
    check_grouped(call(*inputs), reference(*inputs))


@pytest.mark.api("functions.dequant_swiglu_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", SWIGLU_CASES)
def test_swiglu(accuracy_backend, case):
    impl, target, device = accuracy_backend
    supported(impl, target)
    call, reference, inputs = swiglu_case(case, device, impl, False)
    check_grouped(call(*inputs), reference(*inputs), swiglu=True)


@pytest.mark.accuracy
@pytest.mark.parametrize("shape", NORM_SHAPES[:2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", NORM_MODES)
def test_fp8_reference(shape, dtype, mode):
    # The original FP8 norm-quant tests run on CPU, not unsupported NPU kernels.
    call, _, inputs = norm_case(shape, dtype, mode, "cpu", "torch_reference", False, fp8=True)
    output = call(*inputs)
    assert output[0].dtype == torch.float8_e4m3fn
    assert output[0].shape == shape and output[-1].shape == (*shape[:-1], 1)


@pytest.mark.accuracy
@pytest.mark.parametrize("mode", NORM_MODES)
@pytest.mark.parametrize("affine,symmetric", [(True, True), (False, True), (True, False)])
def test_norm_options(accuracy_backend, mode, affine, symmetric):
    # Cover the public smoothing/affine flags in addition to the inherited matrix.
    impl, target, device = accuracy_backend
    supported(impl, target)
    call, reference, inputs = norm_case(
        (2, 256), torch.float16, mode, device, impl, False, smooth=True, affine=affine, symmetric=symmetric
    )
    check_norm(call(*inputs), reference(*inputs), mode, torch.float16)
