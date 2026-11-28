import numpy as np
import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._compile import compile_fullgraph

# T, requests, blocks/request, block size, K, prefill, prefill tokens, prefill requests.
REGRESSION_CASES = [
    (8, 4, 10, 16, 128, False, 0, 0),
    (8, 4, 10, 16, 256, False, 0, 0),
    (8, 4, 10, 16, 512, False, 0, 0),
    (8, 4, 10, 64, 128, False, 0, 0),
    (8, 4, 10, 64, 256, False, 0, 0),
    (8, 4, 10, 64, 512, False, 0, 0),
    (8, 4, 10, 128, 128, False, 0, 0),
    (8, 4, 10, 128, 256, False, 0, 0),
    (8, 4, 10, 128, 512, False, 0, 0),
    (8, 4, 10, 8, 128, False, 0, 0),
    (12, 4, 8, 16, 128, True, 6, 2),
    (1, 1, 10, 16, 128, False, 0, 0),
    (64, 8, 10, 16, 128, False, 0, 0),
    (512, 32, 64, 64, 128, False, 0, 0),
    (2048, 64, 128, 128, 128, False, 0, 0),
    (8192, 256, 1024, 64, 512, False, 0, 0),
    (512, 16, 64, 16, 512, False, 0, 0),
    (2048, 128, 1024, 64, 512, False, 0, 0),
    (2048, 128, 1024, 128, 2048, False, 0, 0),
    (512, 32, 64, 128, 2048, False, 0, 0),
    (8192, 512, 1024, 128, 2048, False, 0, 0),
    (2048, 64, 1024, 64, 512, True, 1024, 8),
]


def make_inputs(case, device, *, seed=42, strided=False):
    tokens, requests, blocks, block_size, topk, has_prefill, prefill_tokens, num_prefill = case
    generator = torch.Generator().manual_seed(seed)
    req = torch.randint(requests, (tokens,), dtype=torch.int32, generator=generator)
    table = torch.randint(4096, (requests, blocks), dtype=torch.int32, generator=generator)
    indices = torch.randint(blocks * block_size, (tokens, topk), dtype=torch.int32, generator=generator)
    if tokens:
        indices[0, :4] = -1
    if tokens > 1:
        indices[1, :4] = blocks * block_size
    ids = starts = None
    if has_prefill:
        ids = torch.full((tokens,), -1, dtype=torch.int32)
        if prefill_tokens:
            ids[-prefill_tokens:] = torch.arange(prefill_tokens, dtype=torch.int32) * num_prefill // prefill_tokens
        starts = torch.arange(num_prefill, dtype=torch.int32) * 100

    def transfer(tensor):
        if tensor is None:
            return None
        tensor = tensor.to(device)
        if not strided:
            return tensor
        # Exercise nonzero storage offsets and strides in every tensor dimension.
        storage = torch.full(tuple(2 * n + 1 for n in tensor.shape), -777, dtype=tensor.dtype, device=device)
        view = storage[tuple(slice(1, 2 * n + 1, 2) for n in tensor.shape)]
        view.copy_(tensor)
        return view

    return (
        transfer(req),
        transfer(table),
        transfer(indices),
        block_size,
        topk,
        has_prefill,
        transfer(ids),
        transfer(starts),
    )


# Model-shaped inputs for the paged mapping branch. The fixed input
# recipe also supplies the performance suite and its L20 baseline.
def _paged_case(name, seed, seq_lens, query_lens):
    return {
        "name": name,
        "seed": seed,
        "seq_lens": seq_lens,
        "query_lens": query_lens,
        "num_requests": len(seq_lens),
        "num_tokens": sum(query_lens),
        "max_blocks_per_req": 3168,
        "block_size": 64,
        "num_topk_tokens": 2048,
        "has_prefill": False,
        "decode_tokens": query_lens.count(1),
        "prefill_tokens": sum(q for q in query_lens if q > 1),
    }


_RAGGED_CONTEXT = (2048, 4096, 8192, 16384)
_PERFORMANCE_CONTEXT = 202752
PERFORMANCE_CASES = (
    _paged_case("mixed_decode_and_prefill", 21008, _RAGGED_CONTEXT * 16 + (_PERFORMANCE_CONTEXT,), (1,) * 64 + (8128,)),
    _paged_case("full_prefill_8192", 21009, (8192,), (8192,)),
    _paged_case("full_prefill_32768", 21010, (32768,), (32768,)),
    _paged_case("full_prefill_131072", 21011, (131072,), (131072,)),
    _paged_case("full_prefill_202752", 21012, (_PERFORMANCE_CONTEXT,), (_PERFORMANCE_CONTEXT,)),
    _paged_case("long_context_single_chunk", 21013, (_PERFORMANCE_CONTEXT,), (8192,)),
    _paged_case("long_context_four_chunks", 21014, (_PERFORMANCE_CONTEXT,) * 4, (8192,) * 4),
)

_ACCURACY_ONLY_CASES = (
    _paged_case("decode_r1_long_context", 21001, (202751,), (1,)),
    _paged_case("decode_r8_ragged_context", 21002, _RAGGED_CONTEXT * 2, (1,) * 8),
    _paged_case("decode_r32_context8192", 21003, (8192,) * 32, (1,) * 32),
    _paged_case("decode_r64_ragged_context", 21004, _RAGGED_CONTEXT * 16, (1,) * 64),
    _paged_case("decode_r128_ragged_context_stress", 21005, _RAGGED_CONTEXT * 32, (1,) * 128),
    _paged_case("mixed_decode32_prefill32_t64", 21006, _RAGGED_CONTEXT * 8 + (32768,), (1,) * 32 + (32,)),
    _paged_case("mixed_decode64_prefill192_t256", 21007, _RAGGED_CONTEXT * 16 + (65536,), (1,) * 64 + (192,)),
)
BALANCED_CASES = (*_ACCURACY_ONLY_CASES, *PERFORMANCE_CASES)


def _small_inputs(c):
    seq = c["seq_lens"]
    query = c["query_lens"]
    r = len(seq)
    m = c["max_blocks_per_req"]
    bs = c["block_size"]
    req = np.repeat(np.arange(r, dtype=np.int32), query)
    positions = np.concatenate([np.arange(s - q, s, dtype=np.int64) for s, q in zip(seq, query)])
    table = np.zeros((r, m), dtype=np.int32)
    physical = 1
    for i, s in enumerate(seq):
        blocks = (s + bs - 1) // bs
        table[i, :blocks] = np.arange(physical, physical + blocks, dtype=np.int32)
        physical += blocks
    assert len(req) == c["num_tokens"] and int(table.max()) * bs + bs < 2**31
    return req, table, positions


def _token_chunk(c, req, positions, start, end):
    p = positions[start:end, None]
    length = p + 1
    k = c["num_topk_tokens"]
    j = np.arange(k, dtype=np.int64)[None, :]
    offset = (
        np.int64(c["seed"]) * 1103515245 + p * 12345 + req[start:end, None].astype(np.int64) * 2654435761
    ) % length
    base = np.where(length >= k, (j * length) // k, j)
    result = np.where(j < np.minimum(k, length), (base + offset) % length, -1).astype(np.int32)
    assert np.all((result == -1) | ((result >= 0) & (result <= p)))
    return np.ascontiguousarray(result)


def build_balanced_inputs(c, device):
    req, table, positions = _small_inputs(c)
    k = c["num_topk_tokens"]
    t = c["num_tokens"]
    tok = torch.empty((t, k), device=device, dtype=torch.int32)
    for start in range(0, t, 1024):
        end = min(t, start + 1024)
        chunk = _token_chunk(c, req, positions, start, end)
        tok[start:end].copy_(torch.from_numpy(chunk))
    return [
        torch.from_numpy(req).to(device),
        torch.from_numpy(table).to(device),
        tok,
        c["block_size"],
        k,
        False,
        None,
        None,
    ]


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


@pytest.mark.reference
@pytest.mark.parametrize(
    ("tokens", "topk", "requests", "blocks"),
    [
        pytest.param(2**20, 2048, 1, 1, id="token-offset-limit"),
        pytest.param(1, 1, 2**16, 2**15, id="page-table-offset-limit"),
        pytest.param(2**30, 1, 1, 1, id="request-scan-limit"),
    ],
)
def test_rejects_int32_index_overflow(tokens, topk, requests, blocks):
    # Meta tensors carry the oversized shapes without allocating any device storage.
    req = torch.empty((tokens,), dtype=torch.int32, device="meta")
    table = torch.empty((requests, blocks), dtype=torch.int32, device="meta")
    indices = torch.empty((tokens, topk), dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="signed int32 addressing"):
        F.triton_convert_req_index_to_global_index(req, table, indices, 64, topk, implementation="triton")


GRAPH_CASES = [
    (129, 1, 3168, 64, 2048, False, 0, 0),
    (8193, 1, 3168, 64, 2048, False, 0, 0),
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
@pytest.mark.parametrize("case", REGRESSION_CASES)
def test_matrix(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    args = make_inputs(case, device)
    actual = F.triton_convert_req_index_to_global_index(*args, implementation=implementation)
    expected = F.triton_convert_req_index_to_global_index(*args, implementation="torch")
    assert torch.equal(actual, expected)


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", BALANCED_CASES, ids=[c["name"] for c in BALANCED_CASES])
def test_balanced_matrix(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    args = build_balanced_inputs(case, device)
    actual = F.triton_convert_req_index_to_global_index(*args, implementation=implementation)
    # Bound CPU/NPU temporary memory while checking every element against the
    # independent Torch implementation, including invalid padding and page tails.
    for start in range(0, case["num_tokens"], 1024):
        chunk = [args[0][start : start + 1024].cpu(), args[1].cpu(), args[2][start : start + 1024].cpu(), *args[3:]]
        expected = F.triton_convert_req_index_to_global_index(*chunk, implementation="torch")
        assert torch.equal(actual[start : start + 1024].cpu(), expected)
