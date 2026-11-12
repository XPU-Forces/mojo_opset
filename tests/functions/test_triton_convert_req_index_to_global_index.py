import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._compile import compile_fullgraph
from tests._convert_req_index import CASES
from tests._convert_req_index import make_inputs


def _inputs(device):
    return (
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([[5, 7], [2, 9]], dtype=torch.int32, device=device),
        torch.tensor([[0, 7, 8, -1], [1, 8, 25, -1]], dtype=torch.int32, device=device),
        4,
        4,
        True,
        torch.tensor([-1, 0], dtype=torch.int32, device=device),
        torch.tensor([100], dtype=torch.int32, device=device),
    )


@pytest.mark.reference
def test_torch_decode_and_prefill():
    result = F.triton_convert_req_index_to_global_index(*_inputs("cpu"), implementation="torch")
    expected = torch.tensor([[20, 31, -1, -1], [101, 108, 125, -1]], dtype=torch.int32)
    assert torch.equal(result, expected)


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.accuracy
def test_npu_matches_reference(accuracy_backend):
    implementation, _, device = accuracy_backend
    inputs = _inputs(device)
    actual = F.triton_convert_req_index_to_global_index(*inputs, implementation=implementation)
    expected = F.triton_convert_req_index_to_global_index(*inputs, implementation="torch")
    assert torch.equal(actual, expected)


BOUNDARY_CASES = [
    (0, 1, 1, 1, 1, False, 0, 0),
    (0, 1, 3, 4, 4, True, 0, 1),
    (1, 1, 1, 1, 1, False, 0, 0),
    (3, 2, 3, 4, 4, False, 0, 0),
    (5, 2, 3, 4, 128, True, 5, 2),
    (5, 2, 3, 4, 128, True, 0, 2),
    (65, 3, 3, 16, 128, True, 7, 2),
    (129, 3, 8, 16, 1024, False, 0, 0),
    (513, 3, 3, 16, 128, True, 257, 2),
    (9, 2, 1, 1, 1, True, 5, 2),
    (9, 2, 3, 4, 1, False, 0, 0),
    (65, 2, 3, 4, 1, True, 33, 2),
    (9, 2, 1, 4, 2, False, 0, 0),
    (9, 2, 1, 4, 4, True, 5, 2),
    (9, 2, 1, 4, 128, False, 0, 0),
]


def _reference(args):
    cpu_args = tuple(x.cpu() if isinstance(x, torch.Tensor) else x for x in args)
    return F.triton_convert_req_index_to_global_index(*cpu_args, implementation="torch")


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", BOUNDARY_CASES)
@pytest.mark.parametrize("strided", [False, True])
def test_boundaries(accuracy_backend, case, strided):
    implementation, _, device = accuracy_backend
    args = make_inputs(case, device, strided=strided)
    snapshots = [x.clone() for x in args if isinstance(x, torch.Tensor)]
    actual = F.triton_convert_req_index_to_global_index(*args, implementation=implementation)
    assert torch.equal(actual.cpu(), _reference(args))
    assert actual.shape == args[2].shape and actual.dtype == torch.int32 and actual.device == args[2].device
    for tensor, snapshot in zip((x for x in args if isinstance(x, torch.Tensor)), snapshots):
        assert torch.equal(tensor, snapshot)


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.accuracy
def test_page_and_prefill_limits(accuracy_backend):
    implementation, _, device = accuracy_backend
    args = (
        torch.tensor([1, 0, 1], dtype=torch.int32, device=device),
        torch.tensor([[2, 7, 9], [10, 20, 30]], dtype=torch.int32, device=device),
        torch.tensor([[-1, 0, 3, 4, 7, 8, 11, 12]] * 3, dtype=torch.int32, device=device),
        4,
        8,
        True,
        torch.tensor([-1, 1, 0], dtype=torch.int32, device=device),
        torch.tensor([100, 200], dtype=torch.int32, device=device),
    )
    expected = torch.tensor(
        [
            [-1, 40, 43, 80, 83, 120, 123, -1],
            [-1, 200, 203, 204, 207, 208, 211, 212],
            [-1, 100, 103, 104, 107, 108, 111, 112],
        ],
        dtype=torch.int32,
    )
    actual = F.triton_convert_req_index_to_global_index(*args, implementation=implementation)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "invalid",
    [
        "req_dtype",
        "table_dtype",
        "indices_dtype",
        "prefill_ids_dtype",
        "starts_dtype",
        "req_rank",
        "table_rank",
        "indices_rank",
        "tokens",
        "empty_requests",
        "empty_blocks",
        "missing_ids",
        "missing_starts",
        "ids_shape",
        "starts_rank",
        "empty_starts",
        "device",
        "block_zero",
        "block_negative",
        "block_nonpower",
        "topk_zero",
        "topk_negative",
        "topk_nonpower",
    ],
)
def test_invalid_contract(accuracy_backend, invalid):
    implementation, _, device = accuracy_backend
    args = list(_inputs(device))
    dtype_indices = {"req_dtype": 0, "table_dtype": 1, "indices_dtype": 2, "prefill_ids_dtype": 6, "starts_dtype": 7}
    if invalid in dtype_indices:
        i = dtype_indices[invalid]
        args[i] = args[i].to(torch.int64)
        message = "int32"
    elif invalid in ("req_rank", "table_rank", "indices_rank"):
        i = {"req_rank": 0, "table_rank": 1, "indices_rank": 2}[invalid]
        args[i] = args[i].unsqueeze(0)
        message = "expected req_id"
    elif invalid in ("tokens", "empty_requests", "empty_blocks"):
        if invalid == "tokens":
            args[0] = args[0][:1]
        elif invalid == "empty_requests":
            args[1] = args[1][:0]
        else:
            args[1] = args[1][:, :0]
        message = "input shapes"
    elif invalid in ("missing_ids", "missing_starts"):
        args[6 if invalid == "missing_ids" else 7] = None
        message = "prefill requires"
    elif invalid in ("ids_shape", "starts_rank", "empty_starts"):
        if invalid == "ids_shape":
            args[6] = args[6][:1]
        elif invalid == "starts_rank":
            args[7] = args[7].unsqueeze(0)
        else:
            args[7] = args[7][:0]
        message = "prefill tensor shapes"
    elif invalid == "device":
        args[1] = args[1].cpu()
        message = "same device"
    else:
        kind, value = invalid.split("_")
        args[3 if kind == "block" else 4] = {"zero": 0, "negative": -4, "nonpower": 3}[value]
        message = "block_size" if kind == "block" else "num_topk_tokens"
    with pytest.raises(ValueError, match=message):
        F.triton_convert_req_index_to_global_index(*args, implementation=implementation)


GRAPH_CASES = [
    (5, 2, 3, 4, 128, False, 0, 0),
    (5, 2, 3, 4, 128, True, 3, 2),
    (0, 2, 3, 4, 128, True, 0, 2),
]


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.parametrize("case", GRAPH_CASES)
@pytest.mark.parametrize("strided", [False, True])
def test_fake_metadata(accuracy_backend, case, strided, monkeypatch):
    from importlib import import_module

    from torch._subclasses.fake_tensor import FakeTensorMode

    implementation, target, device = accuracy_backend
    preload("triton_convert_req_index_to_global_index", implementation=implementation)
    args = make_inputs(case, device, strided=strided)
    actual = F.triton_convert_req_index_to_global_index(*args, implementation=implementation)
    if implementation in (None, "triton") and target.startswith(("npu.a2", "npu.a5")):
        provider = import_module(
            f"mojo_opset.kernels.npu_{target.split('.')[1]}_triton.triton_convert_req_index_to_global_index"
        )

        def unexpected_execution(*args, **kwargs):
            pytest.fail("FakeTensor metadata must not execute the device implementation")

        monkeypatch.setattr(provider.ModelNew, "forward", unexpected_execution)
    with FakeTensorMode() as mode:
        fake_args = tuple(mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in args)
        fake = F.triton_convert_req_index_to_global_index(*fake_args, implementation=implementation)
    assert (fake.shape, fake.dtype, fake.device, fake.stride()) == (
        actual.shape,
        actual.dtype,
        actual.device,
        actual.stride(),
    )


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.parametrize("case", GRAPH_CASES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_fullgraph(accuracy_backend, case, strided, backend):
    implementation, _, device = accuracy_backend
    preload("triton_convert_req_index_to_global_index", implementation=implementation)

    def run(*args):
        result = F.triton_convert_req_index_to_global_index(*args, implementation=implementation)
        return result, result.stride()

    try:
        compiled = compile_fullgraph(run, backend=backend)
        for seed in (42, 123):
            args = make_inputs(case, device, seed=seed, strided=strided)
            actual, stride = compiled(*args)
            eager, expected_stride = run(*args)
            assert torch.equal(actual.cpu(), _reference(args))
            assert torch.equal(actual, eager)
            assert tuple(stride) == expected_stride == actual.stride()
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.parametrize("case", GRAPH_CASES)
def test_registered_schema(accuracy_backend, case):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton registration")
    preload("triton_convert_req_index_to_global_index", implementation="triton")
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    leaf = namespace.triton_convert_req_index_to_global_index.default
    # SchemaCheckMode's clone_preserve_strides cannot clone an empty NPU
    # view with oversized strides. Empty strided metadata is covered above.
    args = make_inputs(case, device, strided=case[0] != 0)
    torch.library.opcheck(leaf, args, test_utils=("test_schema", "test_faketensor"))


@pytest.mark.reference
def test_rejects_unsupported_shapes():
    args = list(_inputs("cpu"))
    args[3] = 3
    with pytest.raises(ValueError, match="block_size"):
        F.triton_convert_req_index_to_global_index(*args, implementation="torch")
    args[3], args[4] = 4, 3
    with pytest.raises(ValueError, match="num_topk_tokens"):
        F.triton_convert_req_index_to_global_index(*args, implementation="torch")


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", CASES)
def test_matrix(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    args = make_inputs(case, device)
    actual = F.triton_convert_req_index_to_global_index(*args, implementation=implementation)
    expected = F.triton_convert_req_index_to_global_index(*args, implementation="torch")
    assert torch.equal(actual, expected)
