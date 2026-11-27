# SPDX-License-Identifier: Apache-2.0
"""Accuracy fixtures for the MiniMax H3 indexed RMSNorm and affine operator."""

import hashlib
import struct

import pytest
import torch

from mojo_opset import functions
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._checks import assert_close
from tests._compile import compile_fullgraph

OP = "rms_norm_indexed_scale_shift"
pytestmark = [pytest.mark.api("functions." + OP), pytest.mark.accuracy]


def case_shape(case):
    return case["rows"], case["hidden_size"], case["conditions"], case.get("active_conditions")


def shape_label(case):
    return "x".join(str(dimension) for dimension in case_shape(case) if dimension is not None)


def _model_case(rows, conditions, active_conditions, seed, indices_segments, indices_sha256):
    return {
        "rows": rows,
        "hidden_size": 5376,
        "conditions": conditions,
        "eps": 1e-5,
        "seed": seed,
        "active_conditions": active_conditions,
        "indices_segments": indices_segments,
        "indices_sha256": indices_sha256,
    }


# MiniMax H3 inputs retain the original condition-index runs and table layout.
MODEL_CASES = (
    _model_case(
        4096,
        3,
        3,
        2026111801,
        ((1, 128), (2, 356), (0, 3612)),
        "653e19ed299e5a60716cd8c020a53f5f7993788f8bc9acbf4dc21a4881a7b8b7",
    ),
    _model_case(
        4096,
        6,
        3,
        2026111802,
        ((1, 128), (5, 356), (0, 3612)),
        "937a4daa8d36bc0853a9b040275fe508f4ddf5574b6a255fe526c5087959b1ec",
    ),
    _model_case(
        512,
        6,
        3,
        2026111803,
        ((1, 128), (5, 356), (0, 28)),
        "4e09502d472a4a0a574061ce98371f26b6c017f2cfb1fd8768e2b75aeaa164be",
    ),
    _model_case(
        16448,
        6,
        3,
        2026111804,
        ((1, 128), (5, 696), (0, 15624)),
        "3705a58cd9d2c79890179d7a95c67bfc57e4bd71cfeb2e7171a81e6458ef21b1",
    ),
    _model_case(
        4112,
        6,
        3,
        2026111805,
        ((1, 128), (5, 696), (0, 3288)),
        "1c2f1a30381bf0c8edcf5db3171d6d41582caa28520922ce97d63cb8f015c1c9",
    ),
    _model_case(
        4112,
        6,
        1,
        2026111806,
        ((0, 4112),),
        "efba72cd933a9a2ccb8b35094e8499f22c2dac508b6b3b2ad5b3d2802b0bf4c7",
    ),
    _model_case(
        63360,
        6,
        3,
        2026111807,
        ((1, 128), (5, 696), (0, 62536)),
        "3b74ae859f078650d7241b35864c931b741353e77e4deb6b519a1310e48c3a15",
    ),
    _model_case(
        15840,
        6,
        3,
        2026111808,
        ((1, 128), (5, 696), (0, 15016)),
        "600ea6392dc1a2f2ed2afeaad10e8141a736c5419f95311fc411f6360c29d13a",
    ),
    _model_case(
        15056,
        6,
        4,
        2026111809,
        ((1, 256), (0, 256), (3, 936), (2, 696), (0, 12912)),
        "612779bcb0d430280beb278a87ac7a6938684cb616c1aae690e3cf0dfb3d67cc",
    ),
    _model_case(
        15056,
        9,
        4,
        2026111810,
        ((1, 256), (0, 256), (6, 936), (5, 696), (0, 12912)),
        "d0d7f9f8aa246b5a916913bd81c08471ac3f7fea6d77de727429010ee84c1e4b",
    ),
    _model_case(
        10192,
        9,
        4,
        2026111811,
        ((1, 512), (0, 512), (6, 2016), (5, 414), (0, 6738)),
        "2f4e4aad6d5e75d2cecef0d078c0705b7827b1678ff0485aaaf5b1a793fc4b4e",
    ),
    _model_case(
        27312,
        6,
        1,
        2026111812,
        ((0, 27312),),
        "d77c340a5939f0d75bfb748f2b0d533c5b924939d3db854bfb3bb5e7161e2b1a",
    ),
    _model_case(
        10688,
        9,
        5,
        2026111813,
        ((1, 384), (0, 128), (3, 252), (8, 160), (2, 414), (0, 9350)),
        "68b2e6f30690a1f1df27134f8ac990e8a4cfa6ecebd72cb904156212504da973",
    ),
    _model_case(
        10688,
        12,
        5,
        2026111814,
        ((1, 384), (0, 128), (6, 252), (11, 160), (5, 414), (0, 9350)),
        "9b220c29add9766233c644dde6782575133d8eac79f06f3fda49cbb11112396c",
    ),
    _model_case(
        5136,
        6,
        3,
        2026111815,
        ((1, 128), (5, 356), (0, 3612), (1, 128), (5, 696), (0, 216)),
        "139ec14957310713227ef8a1296b78e9f1d69cc9f80507fedd32d46277b7363e",
    ),
    _model_case(
        5136,
        9,
        1,
        2026111816,
        ((3, 5136),),
        "43d71e4bd6ebc03a0e489142e6896e5ec308c12a38c4494da5423e079f87aa00",
    ),
)


def _regression_case(rows, hidden_size, conditions, seed, *, input_pattern="random_normal", indices_pattern="cyclic"):
    return {
        "rows": rows,
        "hidden_size": hidden_size,
        "conditions": conditions,
        "eps": 1e-6,
        "seed": seed,
        "input_pattern": input_pattern,
        "indices_pattern": indices_pattern,
    }


REGRESSION_CASES = (
    _regression_case(2, 5376, 1, 0, input_pattern="fp32_affine_rounding_boundary"),
    _regression_case(11, 3072, 4, 17, input_pattern="bf16_residual_boundary"),
    _regression_case(69888, 16, 7, 0),
    _regression_case(1, 5376, 1, 101),
    _regression_case(128, 3072, 4, 102),
    _regression_case(1024, 5376, 7, 103),
    _regression_case(4096, 5376, 7, 104),
    _regression_case(16384, 3072, 4, 105),
    _regression_case(69888, 5376, 7, 106),
    _regression_case(8192, 512, 2, 107),
    _regression_case(8192, 1024, 4, 108),
    _regression_case(4096, 2048, 2, 109),
    _regression_case(2048, 4096, 4, 110),
    _regression_case(2048, 8192, 4, 111),
    _regression_case(1025, 16384, 2, 112),
    _regression_case(4096, 3072, 16, 113),
    _regression_case(4096, 3072, 64, 114),
    _regression_case(65536, 16, 7, 115),
    _regression_case(4096, 5376, 16, 116, indices_pattern="random"),
)


def make_case(case, device):
    rows, hidden, conditions = case["rows"], case["hidden_size"], case["conditions"]
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, generator=generator)

    if "indices_segments" in case:
        x = rand(rows, hidden).to(device)
        weight = rand(hidden).to(device)
        # Move the backing storage before taking views to preserve 6H strides.
        parent = rand(conditions, 6 * hidden).to(device)
        shift, scale, *_ = parent.chunk(6, dim=-1)
        values, lengths = zip(*case["indices_segments"])
        indices = torch.repeat_interleave(torch.tensor(values, dtype=torch.int64), torch.tensor(lengths))
        return x, weight, shift, scale, indices.to(device), case["eps"]

    indices = torch.arange(rows, dtype=torch.int64) % conditions
    pattern = case["input_pattern"]
    if pattern == "fp32_affine_rounding_boundary":
        x = torch.ones((rows, hidden), dtype=torch.bfloat16)
        weight = torch.full((hidden,), 1.125, dtype=torch.bfloat16)
        shift = torch.zeros((conditions, hidden), dtype=torch.bfloat16)
        scale = torch.full_like(shift, 0.09375)
    elif pattern == "bf16_residual_boundary":
        residual, gate, branch = rand(rows, hidden), rand(conditions, hidden), rand(rows, hidden)
        weight, shift, scale = rand(hidden), rand(conditions, hidden), rand(conditions, hidden)
        x = (residual.float() + gate[indices].float() * branch.float()).to(torch.bfloat16)
    else:
        x, weight, shift, scale = rand(rows, hidden), rand(hidden), rand(conditions, hidden), rand(conditions, hidden)
    if case["indices_pattern"] == "random":
        indices = torch.randint(conditions, (rows,), generator=generator, dtype=torch.int64)
    return (*(tensor.to(device) for tensor in (x, weight, shift, scale, indices)), case["eps"])


def _call(args, implementation):
    return functions.rms_norm_indexed_scale_shift(*args, implementation=implementation)


def _check_output(actual, args):
    x, weight, shift, scale, indices, eps = args
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert actual.device == x.device and actual.is_contiguous()
    if x.numel():
        assert all(actual.data_ptr() != tensor.data_ptr() for tensor in args[:-1])
    # Bound the golden's FP32 temporaries even on the largest model case.
    for start in range(0, x.shape[0], 4096):
        rows = slice(start, start + 4096)
        expected = _call((x[rows], weight, shift, scale, indices[rows], eps), "torch")
        assert_close(actual[rows], expected, torch.bfloat16)


@pytest.mark.parametrize("case", MODEL_CASES + REGRESSION_CASES, ids=shape_label)
def test_accuracy(accuracy_backend, case):
    implementation, target, device = accuracy_backend
    args = make_case(case, device)
    if target.startswith("npu.a2") and implementation in (None, "triton") and case["hidden_size"] > 8192:
        with pytest.raises(ValueError, match="hidden sizes up to 8192"):
            _call(args, implementation)
        return
    out = _call(args, implementation)
    _check_output(out, args)
    if case.get("input_pattern") == "fp32_affine_rounding_boundary":
        assert torch.equal(out, torch.full_like(out, 1.2265625))


@pytest.mark.parametrize(("rows", "hidden"), ((1, 16), (3, 8192)), ids=("1x16", "3x8192"))
def test_large_finite_weight_with_zero_input(accuracy_backend, rows, hidden):
    implementation, _, device = accuracy_backend
    x = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=device)
    weight = torch.full((hidden,), 2e38, dtype=x.dtype, device=device)
    shift = torch.ones((1, hidden), dtype=x.dtype, device=device)
    scale = torch.ones_like(shift)
    indices = torch.zeros((rows,), dtype=torch.int64, device=device)
    args = (x, weight, shift, scale, indices)
    reference = functions.rms_norm_indexed_scale_shift(*args, implementation="torch")
    actual = functions.rms_norm_indexed_scale_shift(*args, implementation=implementation)
    assert torch.equal(reference, torch.ones_like(x))
    assert torch.equal(actual, reference)


def _strided_inputs(device):
    rows, hidden = 7, 512
    x = torch.randn((rows, hidden + 16), dtype=torch.bfloat16, device=device)[:, 16:]
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    parent = torch.randn((3, 6 * hidden), dtype=torch.bfloat16, device=device)
    shift, scale, *_ = parent.chunk(6, dim=-1)
    index = torch.arange(rows * 2, dtype=torch.int64, device=device)[1::2] % 3
    backing = torch.empty(rows * 2 + 1, dtype=torch.int64, device=device)
    indices = backing[1::2]
    indices.copy_(index)
    return x, weight, shift, scale, indices, 1e-5


@pytest.mark.parametrize("mode", ("empty", "strided", "stream"))
def test_boundaries(accuracy_backend, mode):
    implementation, _, device = accuracy_backend
    args = _strided_inputs(device)
    if mode == "empty":
        x, weight, shift, scale, indices, eps = args
        args = (x[:0], weight, shift, scale, indices[:0], eps)
    elif mode == "stream" and device == "npu":
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            actual = _call(args, implementation)
        stream.synchronize()
        _check_output(actual, args)
        return
    _check_output(_call(args, implementation), args)


def test_feature_stride(accuracy_backend):
    implementation, target, device = accuracy_backend
    args = _strided_inputs(device)
    tensors = []
    for tensor in args[:4]:
        view = torch.empty((*tensor.shape, 2), dtype=tensor.dtype, device=device)[..., 0]
        view.copy_(tensor)
        tensors.append(view)
    args = (*tensors, *args[4:])
    if target.startswith("npu.a5") and implementation != "torch":
        with pytest.raises(ValueError, match="unit stride"):
            _call(args, implementation)
    else:
        _check_output(_call(args, implementation), args)


@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_fullgraph(accuracy_backend, backend):
    implementation, _, device = accuracy_backend
    preload(OP, implementation=implementation)
    args = _strided_inputs(device)

    def run(*inputs):
        return _call(inputs, implementation)

    try:
        compiled = compile_fullgraph(run, backend=backend)
        _check_output(compiled(*args), args)
        args[0].add_(0.25)
        args[4].copy_((args[4] + 1) % args[2].shape[0])
        _check_output(compiled(*args), args)
    finally:
        torch._dynamo.reset()


def test_schema_and_fake(accuracy_backend):
    implementation, _, device = accuracy_backend
    args = _strided_inputs(device)
    forward, _ = load_impl(OP, implementation)
    if hasattr(forward, "_opoverload"):
        torch.library.opcheck(forward, args, test_utils=("test_schema", "test_faketensor"))


@pytest.mark.reference
def test_reference_rounding():
    args = make_case(REGRESSION_CASES[0], "cpu")
    output = _call(args, "torch")
    assert torch.equal(output, torch.full_like(output, 1.2265625))
    assert not torch.equal(output, torch.full_like(output, 1.125) * (1 + args[3]))


@pytest.mark.reference
@pytest.mark.parametrize("case", MODEL_CASES, ids=shape_label)
def test_model_fixture(case):
    small = dict(case, rows=1)
    _, _, shift, scale, indices, _ = make_case(small, "cpu")
    hidden = case["hidden_size"]
    assert shift.stride() == scale.stride() == (6 * hidden, 1)
    assert shift.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()
    assert (shift.storage_offset(), scale.storage_offset()) == (0, hidden)
    assert indices.numel() == case["rows"] and indices.unique().numel() == case["active_conditions"]
    # The fixture digest is over little-endian signed 64-bit indices.
    index_bytes = struct.pack(f"<{indices.numel()}q", *indices.tolist())
    assert hashlib.sha256(index_bytes).hexdigest() == case["indices_sha256"]


@pytest.mark.reference
@pytest.mark.parametrize(
    "fault",
    (
        "dtype",
        "indices_dtype",
        "indices_shape",
        "weight",
        "table",
        "conditions",
        "eps",
        "grad",
        "hidden",
        "device",
        "addressing",
    ),
)
def test_invalid_metadata(fault):
    args = list(_strided_inputs("cpu"))
    if fault == "dtype":
        args[0] = args[0].float()
    elif fault == "indices_dtype":
        args[4] = args[4].int()
    elif fault == "indices_shape":
        args[4] = args[4][:1]
    elif fault == "weight":
        args[1] = args[1][:-1]
    elif fault == "table":
        args[2] = args[2][:1]
    elif fault == "conditions":
        args[2], args[3] = args[2][:0], args[3][:0]
    elif fault == "eps":
        args[5] = float("nan")
    elif fault == "grad":
        args[0].requires_grad_()
    elif fault == "hidden":
        args[0] = args[0][:, :17]
    elif fault == "device":
        args[4] = args[4].to("meta")
    elif fault == "addressing":
        args[0] = torch.empty_strided((7, 512), (2**30, 1), device="meta", dtype=torch.bfloat16)
        args[1:5] = [tensor.to("meta") for tensor in args[1:5]]
    with pytest.raises((RuntimeError, ValueError)):
        _call(args, "torch")
