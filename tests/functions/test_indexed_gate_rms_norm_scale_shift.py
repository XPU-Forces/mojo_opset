# SPDX-License-Identifier: Apache-2.0
"""Legacy regression cases and 16 pinned MiniMax H3 model-rule fixtures."""

import json

from pathlib import Path

import pytest
import torch

from torch._subclasses.fake_tensor import FakeTensorMode

from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._checks import assert_close
from tests._compile import compile_fullgraph

OP = "indexed_gate_rms_norm_scale_shift_infer"
CASES = json.loads(Path(__file__).with_name("indexed_gate_cases.json").read_text())
MODEL_CASES = json.loads(Path(__file__).with_name("indexed_gate_model16_cases.json").read_text())["cases"]
MODEL_CASE_PARAMS = [pytest.param(case, id=case["id"]) for case in MODEL_CASES]
CASE_PARAMS = [pytest.param(case, id=case["id"]) for case in CASES + MODEL_CASES]
pytestmark = [pytest.mark.api("functions." + OP), pytest.mark.accuracy]


def _a2_rejects_width(accuracy_backend, case):
    implementation, target, _ = accuracy_backend
    return (
        (target == "npu.a2" or target.startswith("npu.a2."))
        and implementation
        in (
            None,
            "triton",
        )
        and case["hidden_size"] != 5376
    )


def make_model_case(case, device):
    """Synthetic values with pinned indices and shared AdaLN storage layout."""
    rows, hidden, conditions = case["rows"], case["hidden_size"], case["conditions"]
    generator = torch.Generator(device=device).manual_seed(case["seed"])

    def rand(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)

    residual, branch, weight = rand(rows, hidden), rand(rows, hidden), rand(hidden)
    parent = rand(conditions, 6 * hidden)
    _, _, gate, shift, scale, _ = parent.chunk(6, dim=-1)
    indices = torch.empty(rows, dtype=torch.int64, device=device)
    for segment in case["indices_segments"]:
        start = segment["start"]
        indices[start : start + segment["rows"]] = segment["condition"]
    return (residual, gate, branch, weight, shift, scale, indices, case["eps"])


def make_case(case, device):
    if case["id"].startswith("model_"):
        return make_model_case(case, device)
    rows, hidden, conditions = case["rows"], case["hidden_size"], case["conditions"]
    generator = torch.Generator(device=device).manual_seed(case["seed"])

    def rand(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)

    if case["input_pattern"] == "fp32_affine_rounding_boundary":
        residual = torch.ones((rows, hidden), dtype=torch.bfloat16, device=device)
        branch = residual.clone()
        weight = torch.full((hidden,), 1.125, dtype=torch.bfloat16, device=device)
        shift = torch.zeros((conditions, hidden), dtype=torch.bfloat16, device=device)
        scale = torch.full_like(shift, 0.09375)
        gate = torch.zeros_like(shift)
    elif case["input_pattern"] == "bf16_residual_boundary":
        residual, gate, branch, weight, shift, scale = (
            rand(rows, hidden),
            rand(conditions, hidden),
            rand(rows, hidden),
            rand(hidden),
            rand(conditions, hidden),
            rand(conditions, hidden),
        )
    else:
        rand(rows, hidden)  # Preserve the original input generator's RNG order.
        residual, branch = rand(rows, hidden), rand(rows, hidden)
        weight, shift, scale, gate = (
            rand(hidden),
            rand(conditions, hidden),
            rand(conditions, hidden),
            rand(conditions, hidden),
        )
    if case["indices_pattern"] == "random":
        indices = torch.randint(conditions, (rows,), generator=generator, device=device, dtype=torch.int64)
    else:
        indices = torch.arange(rows, device=device, dtype=torch.int64) % conditions
    return (residual, gate, branch, weight, shift, scale, indices, case["eps"])


def _call(args, implementation):
    return F.indexed_gate_rms_norm_scale_shift_infer(*args, implementation=implementation)


def _check(actual, expected, args):
    assert len(actual) == 2
    for output, reference in zip(actual, expected):
        assert output.shape == args[0].shape and output.dtype == torch.bfloat16
        assert output.device == args[0].device and output.is_contiguous()
        assert_close(output, reference, torch.bfloat16)
        if output.numel():
            assert all(output.data_ptr() != x.data_ptr() for x in args[:-1])
    if actual[0].numel():
        assert actual[0].data_ptr() != actual[1].data_ptr()


@pytest.mark.parametrize("case", CASE_PARAMS)
def test_accuracy(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    if _a2_rejects_width(accuracy_backend, case):
        args = make_case(dict(case, rows=3), device)
        with pytest.raises(ValueError, match="hidden == 5376"):
            _call(args, implementation)
        return
    args = make_case(case, device)
    before = tuple(x.clone() for x in args[:-1])
    expected = _call(args, "torch")
    actual = _call(args, implementation)
    _check(actual, expected, args)
    for source, snapshot in zip(args[:-1], before):
        assert torch.equal(source, snapshot)
    if case["id"] == "cuda_affine_boundary":
        assert torch.equal(actual[1], torch.full_like(actual[1], 1.2265625))


@pytest.mark.parametrize("case", CASE_PARAMS)
@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_fullgraph(accuracy_backend, case, backend):
    implementation, _, device = accuracy_backend
    if _a2_rejects_width(accuracy_backend, case):
        pytest.skip("A2 requires hidden == 5376")
    preload(OP, implementation=implementation)
    args = make_case(case, device)

    def run(*inputs):
        return _call(inputs, implementation)

    try:
        compiled = compile_fullgraph(run, backend=backend)
        for step in range(2):
            if step:
                args[0].add_(0.25)
                args[6].copy_((args[6] + 1) % case["conditions"])
            _check(compiled(*args), _call(args, "torch"), args)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("case", CASE_PARAMS)
def test_npu_graph(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    if _a2_rejects_width(accuracy_backend, case):
        pytest.skip("A2 requires hidden == 5376")
    if device != "npu" or implementation == "torch":
        pytest.skip("NPU Graph checks target the NPU implementation")
    args = make_case(case, device)
    # Resolve imports and compile every launch before capture, on a side stream.
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            _call(args, implementation)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = _call(args, implementation)
    for step in range(2):
        if step:
            args[0].add_(0.25)
            args[6].copy_((args[6] + 1) % case["conditions"])
        graph.replay()
        torch.npu.synchronize()
        _check(actual, _call(args, "torch"), args)


def test_default_dispatch(accuracy_backend):
    _, _, device = accuracy_backend
    args = make_case(CASES[0], device)
    _check(_call(args, None), _call(args, "torch"), args)


@pytest.mark.parametrize("mode", ("empty", "strided", "storage_offset", "stream"))
def test_boundaries(accuracy_backend, mode):
    implementation, _, device = accuracy_backend
    case = dict(CASES[4], rows=7)
    case["hidden_size"] = 5376 if _a2_rejects_width(accuracy_backend, case) else 65
    args = list(make_case(case, device))
    if mode == "empty":
        args[0], args[2], args[6] = args[0][:0], args[2][:0], args[6][:0]
    elif mode in ("strided", "storage_offset"):
        for i, tensor in enumerate(args[:-1]):
            if mode == "strided":
                backing = torch.empty((*tensor.shape, 2), device=device, dtype=tensor.dtype)
                view = backing[..., 0]
            else:
                backing = torch.empty(tensor.numel() + 1, device=device, dtype=tensor.dtype)
                view = backing[1:].view_as(tensor)
            view.copy_(tensor)
            args[i] = view
    if mode == "stream" and device == "npu":
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            args[0].add_(0.25)
            actual = _call(args, implementation)
        stream.synchronize()
    else:
        actual = _call(args, implementation)
    _check(actual, _call(args, "torch"), args)


def test_schema_and_fake(accuracy_backend):
    implementation, _, device = accuracy_backend
    preload(OP, implementation=implementation)
    args = make_case(dict(CASES[4], rows=7, hidden_size=5376), device)
    forward, _ = load_impl(OP, implementation)
    with FakeTensorMode() as mode:
        fake_args = [mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in args]
        outputs = _call(fake_args, implementation)
        assert all(x.shape == args[0].shape and x.is_contiguous() for x in outputs)
    if hasattr(forward, "_opoverload"):
        schema = forward._opoverload._schema
        assert all(arg.alias_info is None for arg in schema.arguments)
        assert all(result.alias_info is None for result in schema.returns)
        torch.library.opcheck(forward, args, test_utils=("test_schema", "test_faketensor"))


@pytest.mark.reference
def test_reference_rounding():
    args = make_case(CASES[0], "cpu")
    updated, output = _call(args, "torch")
    assert torch.equal(updated, args[0])
    assert torch.equal(output, torch.full_like(output, 1.2265625))
    # Rounding the RMS result before the affine operation gives a different value.
    premature = (torch.full_like(output, 1.125) * (1 + args[5]) + args[4]).to(torch.bfloat16)
    assert not torch.equal(output, premature)


@pytest.mark.reference
@pytest.mark.parametrize(
    "fault",
    (
        "dtype",
        "index_dtype",
        "index_shape",
        "weight",
        "branch",
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
    args = list(make_case(dict(CASES[1], rows=3, hidden_size=16), "cpu"))
    if fault == "dtype":
        args[2] = args[2].float()
    elif fault == "index_dtype":
        args[6] = args[6].int()
    elif fault == "index_shape":
        args[6] = args[6][:1]
    elif fault == "weight":
        args[3] = args[3][:-1]
    elif fault == "branch":
        args[2] = args[2][:1]
    elif fault == "table":
        args[4] = args[4][:1]
    elif fault == "conditions":
        args[1] = args[1][:0]
    elif fault == "eps":
        args[7] = float("nan")
    elif fault == "grad":
        args[0].requires_grad_()
    elif fault == "hidden":
        args[0] = torch.empty((1, 8193), dtype=torch.bfloat16)
    elif fault == "device":
        args[6] = args[6].to("meta")
    elif fault == "addressing":
        args[0] = torch.empty((2**28, 16), dtype=torch.bfloat16, device="meta")
    with pytest.raises((ValueError, RuntimeError)):
        _call(args, "torch")


@pytest.mark.reference
@pytest.mark.parametrize("case", MODEL_CASE_PARAMS)
def test_model_layout_metadata(case):
    import hashlib

    indices = torch.empty(case["rows"], dtype=torch.int64)
    coverage = torch.zeros(case["rows"], dtype=torch.int32)
    for segment in case["indices_segments"]:
        start = segment["start"]
        stop = start + segment["rows"]
        indices[start:stop] = segment["condition"]
        coverage[start:stop] += 1
    assert torch.all(coverage == 1)
    assert hashlib.sha256(indices.numpy().tobytes()).hexdigest() == case["indices_sha256"]
    assert torch.bincount(indices, minlength=case["conditions"]).tolist() == case["condition_row_counts"]
    # A small row count checks the same table strides without large activations.
    args = make_model_case(dict(case, rows=1, indices_segments=[dict(start=0, rows=1, condition=0)]), "cpu")
    tables = [(1, "gate"), (4, "shift"), (5, "scale")]
    for slot, name in tables:
        assert list(args[slot].stride()) == case[name + "_stride"]
        assert args[slot].storage_offset() == case[name + "_storage_offset"]
    assert len({args[slot].untyped_storage().data_ptr() for slot, _ in tables}) == 1
