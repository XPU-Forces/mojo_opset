# SPDX-License-Identifier: Apache-2.0
"""Legacy regression cases and 16 pinned MiniMax H3 model-rule fixtures."""

import hashlib

import pytest
import torch

from torch._subclasses.fake_tensor import FakeTensorMode

from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._checks import assert_close
from tests._compile import compile_fullgraph

OP = "indexed_gate_rms_norm_scale_shift_infer"
# Case ID, rows, hidden, conditions, seed, value pattern, index pattern.
_REGRESSION_CASES = (
    ("cuda_affine_boundary", 2, 5376, 1, 0, "fp32_affine_rounding_boundary", "cycle"),
    ("cuda_residual_boundary", 11, 3072, 4, 17, "bf16_residual_boundary", "cycle"),
    ("npu_grid_limit", 69888, 16, 7, 0, "random_normal", "cycle"),
    ("gen_single_row", 1, 5376, 1, 1, "random_normal", "random"),
    ("gen_odd_rows", 513, 3072, 5, 3, "random_normal", "random"),
    ("gen_mid_rows", 2048, 3072, 4, 4, "random_normal", "random"),
    ("gen_mid_wide", 4096, 5376, 7, 5, "random_normal", "random"),
    ("gen_wide_hidden", 8192, 7168, 4, 6, "random_normal", "random"),
    ("gen_narrow_hidden", 16384, 2048, 8, 7, "random_normal", "random"),
    ("gen_odd_hidden", 8192, 5375, 4, 8, "random_normal", "random"),
    ("gen_grid_exact", 65535, 3072, 4, 9, "random_normal", "random"),
    ("gen_grid_two_chunks", 131072, 3072, 8, 10, "random_normal", "random"),
    ("gen_large_prod", 69888, 5376, 7, 11, "random_normal", "random"),
)
CASES = [
    dict(
        id=case_id,
        rows=rows,
        hidden_size=hidden,
        conditions=conditions,
        seed=seed,
        eps=1e-6,
        input_pattern=values,
        indices_pattern=indices,
    )
    for case_id, rows, hidden, conditions, seed, values, indices in _REGRESSION_CASES
]

# MiniMax H3 layouts derived from vLLM-Omni revision
# d613fdb62c432a342681e88b307d36b54f6ed5f0. Activation values are synthetic.
_MODEL_CASES = (
    dict(
        id="model_t2va_smoke_step0_u1",
        rows=4096,
        conditions=3,
        seed=2026111801,
        indices_segments=[
            dict(start=0, rows=128, condition=1),
            dict(start=128, rows=356, condition=2),
            dict(start=484, rows=3612, condition=0),
        ],
        indices_sha256="653e19ed299e5a60716cd8c020a53f5f7993788f8bc9acbf4dc21a4881a7b8b7",
        condition_row_counts=[3612, 128, 356],
    ),
    dict(
        id="model_t2va_smoke_mid_u1",
        rows=4096,
        conditions=6,
        seed=2026111802,
        indices_segments=[
            dict(start=0, rows=128, condition=1),
            dict(start=128, rows=356, condition=5),
            dict(start=484, rows=3612, condition=0),
        ],
        indices_sha256="937a4daa8d36bc0853a9b040275fe508f4ddf5574b6a255fe526c5087959b1ec",
        condition_row_counts=[3612, 128, 0, 0, 0, 356],
    ),
    dict(
        id="model_t2va_smoke_mid_u8_rank0",
        rows=512,
        conditions=6,
        seed=2026111803,
        indices_segments=[
            dict(start=0, rows=128, condition=1),
            dict(start=128, rows=356, condition=5),
            dict(start=484, rows=28, condition=0),
        ],
        indices_sha256="4e09502d472a4a0a574061ce98371f26b6c017f2cfb1fd8768e2b75aeaa164be",
        condition_row_counts=[28, 128, 0, 0, 0, 356],
    ),
    dict(
        id="model_t2va_doc209_u1",
        rows=16448,
        conditions=6,
        seed=2026111804,
        indices_segments=[
            dict(start=0, rows=128, condition=1),
            dict(start=128, rows=696, condition=5),
            dict(start=824, rows=15624, condition=0),
        ],
        indices_sha256="3705a58cd9d2c79890179d7a95c67bfc57e4bd71cfeb2e7171a81e6458ef21b1",
        condition_row_counts=[15624, 128, 0, 0, 0, 696],
    ),
    dict(
        id="model_t2va_doc209_u4_rank0",
        rows=4112,
        conditions=6,
        seed=2026111805,
        indices_segments=[
            dict(start=0, rows=128, condition=1),
            dict(start=128, rows=696, condition=5),
            dict(start=824, rows=3288, condition=0),
        ],
        indices_sha256="1c2f1a30381bf0c8edcf5db3171d6d41582caa28520922ce97d63cb8f015c1c9",
        condition_row_counts=[3288, 128, 0, 0, 0, 696],
    ),
    dict(
        id="model_t2va_doc209_u4_rank3",
        rows=4112,
        conditions=6,
        seed=2026111806,
        indices_segments=[
            dict(start=0, rows=4112, condition=0),
        ],
        indices_sha256="efba72cd933a9a2ccb8b35094e8499f22c2dac508b6b3b2ad5b3d2802b0bf4c7",
        condition_row_counts=[4112, 0, 0, 0, 0, 0],
    ),
    dict(
        id="model_t2va_recipe209_u1",
        rows=63360,
        conditions=6,
        seed=2026111807,
        indices_segments=[
            dict(start=0, rows=128, condition=1),
            dict(start=128, rows=696, condition=5),
            dict(start=824, rows=62536, condition=0),
        ],
        indices_sha256="3b74ae859f078650d7241b35864c931b741353e77e4deb6b519a1310e48c3a15",
        condition_row_counts=[62536, 128, 0, 0, 0, 696],
    ),
    dict(
        id="model_t2va_recipe209_u4_rank0",
        rows=15840,
        conditions=6,
        seed=2026111808,
        indices_segments=[
            dict(start=0, rows=128, condition=1),
            dict(start=128, rows=696, condition=5),
            dict(start=824, rows=15016, condition=0),
        ],
        indices_sha256="600ea6392dc1a2f2ed2afeaad10e8141a736c5419f95311fc411f6360c29d13a",
        condition_row_counts=[15016, 128, 0, 0, 0, 696],
    ),
    dict(
        id="model_fl2va_first_step0_u4",
        rows=15056,
        conditions=6,
        seed=2026111809,
        indices_segments=[
            dict(start=0, rows=256, condition=1),
            dict(start=256, rows=256, condition=0),
            dict(start=512, rows=936, condition=3),
            dict(start=1448, rows=696, condition=2),
            dict(start=2144, rows=12912, condition=0),
        ],
        indices_sha256="612779bcb0d430280beb278a87ac7a6938684cb616c1aae690e3cf0dfb3d67cc",
        condition_row_counts=[13168, 256, 696, 936, 0, 0],
    ),
    dict(
        id="model_fl2va_first_mid_u4",
        rows=15056,
        conditions=9,
        seed=2026111810,
        indices_segments=[
            dict(start=0, rows=256, condition=1),
            dict(start=256, rows=256, condition=0),
            dict(start=512, rows=936, condition=6),
            dict(start=1448, rows=696, condition=5),
            dict(start=2144, rows=12912, condition=0),
        ],
        indices_sha256="d0d7f9f8aa246b5a916913bd81c08471ac3f7fea6d77de727429010ee84c1e4b",
        condition_row_counts=[13168, 256, 0, 0, 0, 696, 936, 0, 0],
    ),
    dict(
        id="model_fl2va_firstlast124_u4",
        rows=10192,
        conditions=9,
        seed=2026111811,
        indices_segments=[
            dict(start=0, rows=512, condition=1),
            dict(start=512, rows=512, condition=0),
            dict(start=1024, rows=2016, condition=6),
            dict(start=3040, rows=414, condition=5),
            dict(start=3454, rows=6738, condition=0),
        ],
        indices_sha256="2f4e4aad6d5e75d2cecef0d078c0705b7827b1678ff0485aaaf5b1a793fc4b4e",
        condition_row_counts=[7250, 512, 0, 0, 0, 414, 2016, 0, 0],
    ),
    dict(
        id="model_t2va_default_limit_u4_rank3",
        rows=27312,
        conditions=6,
        seed=2026111812,
        indices_segments=[
            dict(start=0, rows=27312, condition=0),
        ],
        indices_sha256="d77c340a5939f0d75bfb748f2b0d533c5b924939d3db854bfb3bb5e7161e2b1a",
        condition_row_counts=[27312, 0, 0, 0, 0, 0],
    ),
    dict(
        id="model_ref2va_img_audio_step0_u1",
        rows=10688,
        conditions=9,
        seed=2026111813,
        indices_segments=[
            dict(start=0, rows=384, condition=1),
            dict(start=384, rows=128, condition=0),
            dict(start=512, rows=252, condition=3),
            dict(start=764, rows=160, condition=8),
            dict(start=924, rows=414, condition=2),
            dict(start=1338, rows=9350, condition=0),
        ],
        indices_sha256="68b2e6f30690a1f1df27134f8ac990e8a4cfa6ecebd72cb904156212504da973",
        condition_row_counts=[9478, 384, 414, 252, 0, 0, 0, 0, 160],
    ),
    dict(
        id="model_ref2va_img_audio_mid_u1",
        rows=10688,
        conditions=12,
        seed=2026111814,
        indices_segments=[
            dict(start=0, rows=384, condition=1),
            dict(start=384, rows=128, condition=0),
            dict(start=512, rows=252, condition=6),
            dict(start=764, rows=160, condition=11),
            dict(start=924, rows=414, condition=5),
            dict(start=1338, rows=9350, condition=0),
        ],
        indices_sha256="9b220c29add9766233c644dde6782575133d8eac79f06f3fda49cbb11112396c",
        condition_row_counts=[9478, 384, 0, 0, 0, 414, 252, 0, 0, 0, 0, 160],
    ),
    dict(
        id="model_batch2_same_step_u4_rank0",
        rows=5136,
        conditions=6,
        seed=2026111815,
        indices_segments=[
            dict(start=0, rows=128, condition=1),
            dict(start=128, rows=356, condition=5),
            dict(start=484, rows=3612, condition=0),
            dict(start=4096, rows=128, condition=1),
            dict(start=4224, rows=696, condition=5),
            dict(start=4920, rows=216, condition=0),
        ],
        indices_sha256="139ec14957310713227ef8a1296b78e9f1d69cc9f80507fedd32d46277b7363e",
        condition_row_counts=[3828, 256, 0, 0, 0, 1052],
    ),
    dict(
        id="model_batch2_mixed_step_u4_rank3",
        rows=5136,
        conditions=9,
        seed=2026111816,
        indices_segments=[
            dict(start=0, rows=5136, condition=3),
        ],
        indices_sha256="43d71e4bd6ebc03a0e489142e6896e5ec308c12a38c4494da5423e079f87aa00",
        condition_row_counts=[0, 0, 0, 5136, 0, 0, 0, 0, 0],
    ),
)
MODEL_CASES = [
    dict(
        case,
        hidden_size=5376,
        eps=1e-05,
        gate_stride=[32256, 1],
        shift_stride=[32256, 1],
        scale_stride=[32256, 1],
        gate_storage_offset=10752,
        shift_storage_offset=16128,
        scale_storage_offset=21504,
    )
    for case in _MODEL_CASES
]
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


def test_condition_table_large_row_stride(accuracy_backend):
    implementation, target, device = accuracy_backend
    if target not in ("npu.a2", "npu.a5") or implementation not in (None, "triton"):
        pytest.skip("large row-stride regression targets Ascend Triton")

    hidden, conditions = 5376, 3
    row_stride = 2**30
    num_cores = torch.npu.get_device_properties(torch.npu.current_device()).vector_core_num
    # One full group per core plus a tail row on both providers.
    args = list(make_case(dict(CASES[4], rows=2 * num_cores + 1, hidden_size=hidden, conditions=conditions), device))
    args[6].fill_(conditions - 1)
    wide_args = args.copy()
    table_slots = (1, 4, 5)
    # Share the backing storage and write only the logical rows.
    backing = torch.empty(
        (conditions - 1) * row_stride + len(table_slots) * hidden, dtype=torch.bfloat16, device=device
    )
    for i, slot in enumerate(table_slots):
        wide_table = backing.as_strided(args[slot].shape, (row_stride, 1), storage_offset=i * hidden)
        for row in range(conditions):
            wide_table[row].copy_(args[slot][row])
        wide_args[slot] = wide_table

    _check(_call(wide_args, "triton"), _call(args, "torch"), wide_args)
    for slot in (0, 2, 6):
        args[slot] = wide_args[slot] = args[slot][:1]
    _check(_call(wide_args, "triton"), _call(args, "torch"), wide_args)


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
