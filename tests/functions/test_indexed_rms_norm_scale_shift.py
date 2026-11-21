# SPDX-License-Identifier: Apache-2.0
"""Accuracy fixtures for the MiniMax H3 indexed RMSNorm and affine operator."""

import hashlib
import json

from pathlib import Path

import pytest
import torch

from mojo_opset import functions
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._checks import assert_close
from tests._compile import compile_fullgraph

OP = "indexed_rms_norm_scale_shift_infer"
CASES = json.loads(Path(__file__).with_name("indexed_rms_norm_scale_shift_cases.json").read_text(encoding="utf-8"))
MODEL_CASES = [pytest.param(case, id=case["id"]) for case in CASES["model16"]]
REGRESSION_CASES = [pytest.param(case, id=case["id"]) for case in CASES["regression"]]
pytestmark = [pytest.mark.api("functions." + OP), pytest.mark.accuracy]


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
    return functions.indexed_rms_norm_scale_shift_infer(*args, implementation=implementation)


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


@pytest.mark.parametrize("case", MODEL_CASES + REGRESSION_CASES)
def test_accuracy(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    args = make_case(case, device)
    out = _call(args, implementation)
    _check_output(out, args)
    if case.get("input_pattern") == "fp32_affine_rounding_boundary":
        assert torch.equal(out, torch.full_like(out, 1.2265625))


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
    args = make_case(CASES["regression"][0], "cpu")
    output = _call(args, "torch")
    assert torch.equal(output, torch.full_like(output, 1.2265625))
    assert not torch.equal(output, torch.full_like(output, 1.125) * (1 + args[3]))


@pytest.mark.reference
@pytest.mark.parametrize("case", MODEL_CASES)
def test_model_fixture(case):
    small = dict(case, rows=1)
    _, _, shift, scale, indices, _ = make_case(small, "cpu")
    hidden = case["hidden_size"]
    assert shift.stride() == scale.stride() == (6 * hidden, 1)
    assert shift.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()
    assert (shift.storage_offset(), scale.storage_offset()) == (0, hidden)
    assert indices.numel() == case["rows"] and indices.unique().numel() == case["active_conditions"]
    assert hashlib.sha256(indices.numpy().tobytes()).hexdigest() == case["indices_sha256"]


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
