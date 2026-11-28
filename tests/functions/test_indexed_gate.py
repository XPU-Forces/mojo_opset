# SPDX-License-Identifier: Apache-2.0
"""Indexed gate correctness, dispatch and graph tests."""

import hashlib

import pytest
import torch

from torch._subclasses.fake_tensor import FakeTensorMode

from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._checks import assert_close
from tests._compile import compile_fullgraph

OP = "indexed_gate_infer"

# R, C, sequence-parallel world size and rank, CPU seed, runs (length, condition),
# and SHA-256 of the reconstructed little-endian int64 indices.
PERF_CASES = (
    (
        4096,
        3,
        1,
        0,
        2026111801,
        ((128, 1), (356, 2), (3612, 0)),
        "653e19ed299e5a60716cd8c020a53f5f7993788f8bc9acbf4dc21a4881a7b8b7",
    ),
    (
        4096,
        6,
        1,
        0,
        2026111802,
        ((128, 1), (356, 5), (3612, 0)),
        "937a4daa8d36bc0853a9b040275fe508f4ddf5574b6a255fe526c5087959b1ec",
    ),
    (
        16448,
        6,
        1,
        0,
        2026111804,
        ((128, 1), (696, 5), (15624, 0)),
        "3705a58cd9d2c79890179d7a95c67bfc57e4bd71cfeb2e7171a81e6458ef21b1",
    ),
    (
        4112,
        6,
        4,
        0,
        2026111805,
        ((128, 1), (696, 5), (3288, 0)),
        "1c2f1a30381bf0c8edcf5db3171d6d41582caa28520922ce97d63cb8f015c1c9",
    ),
    (4112, 6, 4, 3, 2026111806, ((4112, 0),), "efba72cd933a9a2ccb8b35094e8499f22c2dac508b6b3b2ad5b3d2802b0bf4c7"),
    (
        63360,
        6,
        1,
        0,
        2026111807,
        ((128, 1), (696, 5), (62536, 0)),
        "3b74ae859f078650d7241b35864c931b741353e77e4deb6b519a1310e48c3a15",
    ),
    (
        15840,
        6,
        4,
        0,
        2026111808,
        ((128, 1), (696, 5), (15016, 0)),
        "600ea6392dc1a2f2ed2afeaad10e8141a736c5419f95311fc411f6360c29d13a",
    ),
    (
        15056,
        6,
        4,
        0,
        2026111809,
        ((256, 1), (256, 0), (936, 3), (696, 2), (12912, 0)),
        "612779bcb0d430280beb278a87ac7a6938684cb616c1aae690e3cf0dfb3d67cc",
    ),
    (
        15056,
        9,
        4,
        0,
        2026111810,
        ((256, 1), (256, 0), (936, 6), (696, 5), (12912, 0)),
        "d0d7f9f8aa246b5a916913bd81c08471ac3f7fea6d77de727429010ee84c1e4b",
    ),
    (
        10192,
        9,
        4,
        0,
        2026111811,
        ((512, 1), (512, 0), (2016, 6), (414, 5), (6738, 0)),
        "2f4e4aad6d5e75d2cecef0d078c0705b7827b1678ff0485aaaf5b1a793fc4b4e",
    ),
    (27312, 6, 4, 3, 2026111812, ((27312, 0),), "d77c340a5939f0d75bfb748f2b0d533c5b924939d3db854bfb3bb5e7161e2b1a"),
    (
        10688,
        9,
        1,
        0,
        2026111813,
        ((384, 1), (128, 0), (252, 3), (160, 8), (414, 2), (9350, 0)),
        "68b2e6f30690a1f1df27134f8ac990e8a4cfa6ecebd72cb904156212504da973",
    ),
    (
        10688,
        12,
        1,
        0,
        2026111814,
        ((384, 1), (128, 0), (252, 6), (160, 11), (414, 5), (9350, 0)),
        "9b220c29add9766233c644dde6782575133d8eac79f06f3fda49cbb11112396c",
    ),
    (
        5136,
        6,
        4,
        0,
        2026111815,
        ((128, 1), (356, 5), (3612, 0), (128, 1), (696, 5), (216, 0)),
        "139ec14957310713227ef8a1296b78e9f1d69cc9f80507fedd32d46277b7363e",
    ),
    (5136, 9, 4, 3, 2026111816, ((5136, 3),), "43d71e4bd6ebc03a0e489142e6896e5ec308c12a38c4494da5423e079f87aa00"),
)
_MODEL_FUNCTIONAL = (
    (
        512,
        6,
        8,
        0,
        2026111803,
        ((128, 1), (356, 5), (28, 0)),
        "4e09502d472a4a0a574061ce98371f26b6c017f2cfb1fd8768e2b75aeaa164be",
    ),
)
# R, H, C, gate dtype, CPU seed, packed gate table.
_UPSTREAM_CASES = (
    (69888, 16, 7, "bfloat16", 0, False),
    (1, 5376, 12, "bfloat16", 0, True),
    (1, 5376, 12, "float32", 0, True),
    (777, 5376, 12, "bfloat16", 0, True),
    (777, 5376, 12, "float32", 0, True),
    (64, 3072, 12, "bfloat16", 0, True),
    (64, 3072, 12, "float32", 0, True),
    (33, 1000, 12, "bfloat16", 0, True),
    (33, 1000, 12, "float32", 0, True),
    (256, 5376, 12, "float32", 0, True),
)


def shape_id(case):
    rows, conditions, world_size, rank, *_ = case
    shape = f"{rows}x5376x{conditions}"
    if rows == 4112:
        return f"{shape}-sp{world_size}-rank{rank}"
    return shape


def make_model_case(case, device):
    rows, conditions, _, _, seed, runs, digest = case
    hidden = 5376
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, hidden, generator=rng, dtype=torch.bfloat16).to(device)
    other = torch.randn(rows, hidden, generator=rng, dtype=torch.bfloat16).to(device)
    table = torch.randn(conditions, 6 * hidden, generator=rng, dtype=torch.bfloat16).to(device)
    gate = table[:, 5 * hidden :]
    indices = torch.cat([torch.full((length,), condition, dtype=torch.int64) for length, condition in runs])
    assert indices.shape == (rows,)
    assert hashlib.sha256(indices.numpy().tobytes()).hexdigest() == digest
    return x, gate, other, indices.to(device)


def _make_upstream(case, device, seed=None):
    rows, hidden, conditions, dtype, initial_seed, packed = case
    dtype = getattr(torch, dtype)
    rng = torch.Generator().manual_seed(initial_seed if seed is None else seed)
    x = torch.randn(rows, hidden, generator=rng, dtype=torch.bfloat16).to(device)
    if not packed:
        torch.randn(rows, hidden, generator=rng, dtype=torch.bfloat16)
    other = torch.randn(rows, hidden, generator=rng, dtype=torch.bfloat16).to(device)
    if packed:
        table = 0.3 * torch.randn(conditions, 3 * hidden, generator=rng, dtype=torch.float32)
        gate = table.to(device=device, dtype=dtype)[:, :hidden]
        torch.randn(hidden, generator=rng, dtype=torch.float32)
        indices = torch.randint(conditions, (rows,), generator=rng, dtype=torch.int64)
    else:
        torch.randn(hidden, generator=rng, dtype=torch.bfloat16)
        torch.randn(conditions, hidden, generator=rng, dtype=dtype)
        torch.randn(conditions, hidden, generator=rng, dtype=dtype)
        gate = torch.randn(conditions, hidden, generator=rng, dtype=dtype).to(device)
        indices = torch.arange(rows) % conditions
    return x, gate, other, indices.to(device)


def _assert_accuracy(output, args):
    x, gate, other, indices = args
    assert output.shape == x.shape and output.dtype == x.dtype and output.device == x.device
    assert output.is_contiguous()
    if output.numel():
        assert all(output.untyped_storage().data_ptr() != t.untyped_storage().data_ptr() for t in args)
    # Bound the peak memory used by the FP32 Torch reference and its comparison.
    for start in range(0, x.shape[0], 4096):
        window = slice(start, start + 4096)
        ref = F.indexed_gate_infer(x[window], gate, other[window], indices[window], implementation="torch")
        assert_close(output[window], ref)


@pytest.mark.api("functions." + OP)
@pytest.mark.accuracy
@pytest.mark.parametrize("case", PERF_CASES + _MODEL_FUNCTIONAL, ids=shape_id)
def test_model_accuracy(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    args = make_model_case(case, device)
    _assert_accuracy(F.indexed_gate_infer(*args, implementation=implementation), args)


@pytest.mark.api("functions." + OP)
@pytest.mark.accuracy
@pytest.mark.parametrize("case", _UPSTREAM_CASES, ids=lambda c: f"{c[0]}x{c[1]}x{c[2]}-{c[3]}")
def test_upstream_accuracy(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    for seed in (0, 1, 2) if case[0] == 256 else (case[4],):
        args = _make_upstream(case, device, seed)
        _assert_accuracy(F.indexed_gate_infer(*args, implementation=implementation), args)


def _small_args(device, rows=11, hidden=16, conditions=7):
    x = torch.randn(rows, 2 * hidden, dtype=torch.bfloat16, device=device)[:, ::2]
    gate = torch.randn(conditions, 2 * hidden, device=device)[:, 1::2]
    other = torch.randn(rows, 2 * hidden, dtype=torch.bfloat16, device=device)[:, 1::2]
    indices = (torch.arange(2 * rows, device=device) % conditions)[::2]
    return x, gate, other, indices


@pytest.mark.api("functions." + OP)
@pytest.mark.accuracy
@pytest.mark.parametrize("rows,hidden,conditions", [(11, 16, 7), (0, 16, 0), (5, 0, 0), (3, 8192, 3)])
def test_layout_and_empty(accuracy_backend, rows, hidden, conditions):
    implementation, _, device = accuracy_backend
    args = _small_args(device, rows, hidden, max(conditions, 1))
    if conditions == 0:
        args = (args[0], args[1][:0], *args[2:])
    _assert_accuracy(F.indexed_gate_infer(*args, implementation=implementation), args)


@pytest.mark.api("functions." + OP)
@pytest.mark.accuracy
@pytest.mark.parametrize("rows", [65535, 65536])
def test_grid_boundary(accuracy_backend, rows):
    implementation, _, device = accuracy_backend
    args = _small_args(device, rows=rows)
    _assert_accuracy(F.indexed_gate_infer(*args, implementation=implementation), args)


@pytest.mark.api("functions." + OP)
@pytest.mark.accuracy
@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "mixed"])
def test_condition_chunks(accuracy_backend, uniform):
    implementation, _, device = accuracy_backend
    x, gate, other, ids = _small_args(device, rows=79, hidden=128, conditions=73)
    if uniform:
        ids = torch.full_like(ids, 72)
    _assert_accuracy(F.indexed_gate_infer(x, gate, other, ids, implementation=implementation), (x, gate, other, ids))


@pytest.mark.api("functions." + OP)
@pytest.mark.accuracy
def test_default_dispatch(accuracy_backend):
    _, target, device = accuracy_backend
    if not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("requires the default Ascend Triton binding")
    default, _ = load_impl(OP, None)
    selected, _ = load_impl(OP, "triton")
    assert default is selected
    args = _small_args(device)
    _assert_accuracy(F.indexed_gate_infer(*args), args)


@pytest.mark.api("functions." + OP)
@pytest.mark.accuracy
def test_schema_and_fake(accuracy_backend):
    implementation, _, device = accuracy_backend
    implementation = implementation or "triton"
    if implementation != "triton":
        pytest.skip("requires a Triton custom operator")
    args = _small_args(device)
    binding, _ = load_impl(OP, implementation)
    torch.library.opcheck(binding, args, test_utils=("test_schema", "test_faketensor"))
    with FakeTensorMode() as mode:
        fake = F.indexed_gate_infer(*(mode.from_tensor(t) for t in args), implementation=implementation)
    assert fake.is_contiguous() and fake.shape == args[0].shape and fake.dtype == args[0].dtype


@pytest.mark.api("functions." + OP)
@pytest.mark.accuracy
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_fullgraph(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    implementation = implementation or "triton"
    preload(OP, target=target, implementation=implementation)

    def apply(*args):
        return F.indexed_gate_infer(*args, implementation=implementation)

    compiled = compile_fullgraph(apply, backend=backend)
    for _ in range(2):
        args = _small_args(device)
        _assert_accuracy(compiled(*args), args)


@pytest.mark.api("functions." + OP)
@pytest.mark.reference
def test_reference():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    gate = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32)
    indices = torch.tensor([1, 0])
    got = F.indexed_gate_infer(x, gate, x, indices, implementation="torch")
    torch.testing.assert_close(got, torch.tensor([[5, 12], [9, 16]], dtype=x.dtype), rtol=0, atol=0)


@pytest.mark.api("functions." + OP)
@pytest.mark.reference
@pytest.mark.parametrize("fault", ["rank", "other", "gate", "indices", "device", "autograd"])
def test_invalid_metadata(fault):
    x, gate, other, indices = _small_args("cpu")
    if fault == "rank":
        x = x.flatten()
    elif fault == "other":
        other = other.float()
    elif fault == "gate":
        gate = gate[:, :-1]
    elif fault == "indices":
        indices = indices.int()
    elif fault == "device":
        gate = gate.to("meta")
    elif fault == "autograd":
        x.requires_grad_(True)
    with pytest.raises(RuntimeError if fault == "autograd" else ValueError):
        F.indexed_gate_infer(x, gate, other, indices, implementation="torch")
