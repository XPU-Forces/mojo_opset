"""Accuracy suite for compressor_from_projected (token-aligned C2 pooling).

The oracle is an independent FP32 PyTorch loop over the request metadata, so it
does not share indexing code with the kernel or with the device-independent
provider. Completed pairs use the BF16 2^-7 mixed tolerance; untouched output
rows, ring rows and gathered-path ring state are compared byte for byte.
"""

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset.functions._dispatch import load_impl

from tests._checks import assert_bitwise_equal

STATE_RING_ROWS = 32
TOLERANCE = 2**-7
POISON = 17.0

pytestmark = [
    pytest.mark.api("functions.compressor_from_projected"),
    pytest.mark.accuracy,
]


def _platform_cores():
    properties = torch.npu.get_device_properties(torch.npu.current_device())
    return getattr(properties, "vector_core_num", None) or properties.multi_processor_count


def _build(dim, lengths, starts, blocks, *, ring_rows=6, seed=0, capacity=None, gathered=None):
    requests = len(lengths)
    capacity = capacity or (max(lengths) + 2 if lengths else 1)
    tokens = requests * capacity + 2
    generator = torch.Generator().manual_seed(seed)
    kv = torch.randn(tokens, dim, generator=generator)
    scores = torch.randn(tokens, dim, generator=generator) * 3
    state = torch.randn(ring_rows, STATE_RING_ROWS, 2 * dim, generator=torch.Generator().manual_seed(seed + 1))
    metadata = torch.zeros(5, requests, dtype=torch.int32)
    metadata[0] = torch.tensor(starts, dtype=torch.int32)
    metadata[1] = torch.tensor(lengths, dtype=torch.int32)
    metadata[3] = torch.arange(requests, dtype=torch.int32) * capacity
    metadata[4] = torch.tensor(blocks, dtype=torch.int32)
    prev_rows = prev_indices = None
    if gathered is not None:
        prev_rows = torch.randn(requests, 2 * dim, generator=torch.Generator().manual_seed(seed + 2))
        if gathered == "valid":
            prev_indices = torch.arange(requests, dtype=torch.int64)
        elif gathered == "ring":
            prev_indices = torch.full((requests,), -1, dtype=torch.int64)
        else:
            prev_indices = torch.tensor([-1 if r % 2 == 0 else r // 2 for r in range(requests)], dtype=torch.int64)
    return kv, scores, state, metadata, prev_rows, prev_indices


def _expected(kv, scores, state, metadata, prev_rows, prev_indices):
    width = kv.shape[1]
    expected = torch.zeros_like(kv, dtype=torch.bfloat16)
    updated = state.clone()
    written = torch.zeros(kv.shape[0], dtype=torch.bool)
    for request in range(metadata.shape[1]):
        start, used, _, base, block = (int(value) for value in metadata[:, request])
        if not 0 < block < state.shape[0]:
            continue
        for offset in range(used):
            position = start + offset
            if position % 2:
                if offset:
                    previous = torch.cat((kv[base + offset - 1], scores[base + offset - 1]))
                elif prev_rows is not None and int(prev_indices[request]) >= 0:
                    previous = prev_rows[int(prev_indices[request])]
                else:
                    previous = state[block, (position - 1) % STATE_RING_ROWS]
                pair_k = torch.stack((previous[:width], kv[base + offset]))
                pair_s = torch.stack((previous[width:], scores[base + offset]))
                expected[base + offset] = (pair_s.softmax(0) * pair_k).sum(0).to(torch.bfloat16)
                written[base + offset] = True
            if prev_rows is None:
                updated[block, position % STATE_RING_ROWS] = torch.cat((kv[base + offset], scores[base + offset]))
    return expected, updated, written


def _execute(kv, scores, state, metadata, prev_rows, prev_indices, implementation, *, max_query_len=None, cores=None):
    tokens, dim = kv.shape
    if max_query_len is None:
        max_query_len = 0 if metadata.shape[1] == 0 else int(metadata[1].max())
    cores = cores or _platform_cores()
    out = torch.full((tokens, dim), POISON, dtype=torch.bfloat16)
    device_state = state.npu()
    device_out = out.npu()
    device_prev = prev_rows.npu() if prev_rows is not None else None
    device_indices = prev_indices.npu() if prev_indices is not None else None
    result = F.compressor_from_projected(
        kv.npu(),
        scores.npu(),
        device_state,
        metadata.npu(),
        device_out,
        max_query_len=max_query_len,
        num_cores=cores,
        prev_rows=device_prev,
        prev_row_indices=device_indices,
        implementation=implementation,
    )
    assert result.data_ptr() == device_out.data_ptr()
    return device_out.cpu(), device_state.cpu()


def _check(kv, scores, state, metadata, actual_out, actual_state, prev_rows, prev_indices):
    expected, updated, written = _expected(kv, scores, state, metadata, prev_rows, prev_indices)
    torch.testing.assert_close(actual_out, expected, rtol=TOLERANCE, atol=TOLERANCE)
    assert_bitwise_equal(actual_out[~written], expected[~written], phase="untouched-out")
    assert_bitwise_equal(actual_state, state if prev_rows is not None else updated, phase="state")


RING_CASES = [
    (128, [1, 1, 1, 1], [0, 0, 0, 0]),
    (128, [31, 29, 17, 1], [5, 5, 5, 5]),
    (128, [32, 32, 32, 32], [0, 0, 0, 0]),
    (128, [33, 32, 31, 30], [0, 1, 2, 3]),
    (128, [63, 60, 59, 58], [1, 0, 1, 0]),
    (128, [65, 64, 63, 62], [0, 0, 0, 0]),
    (512, [1, 1, 1, 1], [1, 1, 1, 1]),
    (512, [31, 31, 31, 31], [0, 0, 0, 0]),
    (512, [33, 33, 33, 33], [1, 1, 1, 1]),
    (512, [64, 64, 64, 64], [0, 0, 0, 0]),
    (512, [65, 65, 65, 65], [0, 0, 0, 0]),
    (512, [1024, 7, 0, 1], [0, 3, 0, 0]),
]


@pytest.mark.parametrize(
    "dim,lengths,starts",
    RING_CASES,
    ids=[f"d{dim}-r{len(lengths)}-u{lengths[0]}" for dim, lengths, _ in RING_CASES],
)
def test_ring_path(dim, lengths, starts, accuracy_backend):
    implementation, _, _ = accuracy_backend
    blocks = [1, 2, 3, 4]
    kv, scores, state, metadata, _, _ = _build(dim, lengths, starts, blocks, seed=11)
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


GATHERED_CASES = [
    (128, [32, 32, 32, 32], [0, 0, 0, 0], "valid"),
    (128, [33, 32, 31, 30], [1, 0, 1, 0], "mixed"),
    (128, [31, 31, 31, 31], [0, 0, 0, 0], "ring"),
    (128, [65, 64, 63, 62], [1, 0, 1, 0], "valid"),
    (128, [1, 2, 3, 4], [1, 1, 1, 1], "mixed"),
    (128, [64, 64, 64, 64], [1, 0, 1, 0], "ring"),
    (512, [32, 32, 32, 32], [0, 0, 0, 0], "valid"),
    (512, [33, 32, 31, 30], [1, 0, 1, 0], "mixed"),
    (512, [31, 31, 31, 31], [0, 0, 0, 0], "ring"),
    (512, [65, 64, 63, 62], [1, 0, 1, 0], "valid"),
    (512, [1, 2, 3, 4], [1, 1, 1, 1], "mixed"),
    (512, [64, 64, 64, 64], [1, 0, 1, 0], "ring"),
]


@pytest.mark.parametrize(
    "dim,lengths,starts,kind",
    GATHERED_CASES,
    ids=[f"d{dim}-{kind}-u{lengths[0]}" for dim, lengths, _, kind in GATHERED_CASES],
)
def test_gathered_path(dim, lengths, starts, kind, accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, prev_rows, prev_indices = _build(
        dim, lengths, starts, [1, 2, 3, 4], seed=23, gathered=kind
    )
    out, actual_state = _execute(kv, scores, state, metadata, prev_rows, prev_indices, implementation)
    _check(kv, scores, state, metadata, out, actual_state, prev_rows, prev_indices)


def test_empty_tokens(accuracy_backend):
    implementation, _, _ = accuracy_backend
    dim = 128
    kv = torch.zeros(0, dim)
    scores = torch.zeros(0, dim)
    state = torch.randn(4, STATE_RING_ROWS, 2 * dim, generator=torch.Generator().manual_seed(1))
    metadata = torch.zeros(5, 2, dtype=torch.int32)
    metadata[4] = torch.tensor([1, 2], dtype=torch.int32)
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation, max_query_len=0)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


def test_empty_requests(accuracy_backend):
    implementation, _, _ = accuracy_backend
    dim = 128
    kv = torch.randn(4, dim, generator=torch.Generator().manual_seed(2))
    scores = torch.randn(4, dim, generator=torch.Generator().manual_seed(3))
    state = torch.randn(4, STATE_RING_ROWS, 2 * dim, generator=torch.Generator().manual_seed(4))
    metadata = torch.zeros(5, 0, dtype=torch.int32)
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation, max_query_len=0)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


def test_zero_max_query_len(accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, _, _ = _build(128, [0, 0], [3, 5], [1, 2], seed=31)
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation, max_query_len=0)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


@pytest.mark.parametrize("block", [0, 6], ids=["block-zero", "block-above"])
def test_invalid_page(block, accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, _, _ = _build(128, [32, 32], [0, 0], [block, 3], seed=41)
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


def test_inactive_requests_mixed(accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, _, _ = _build(512, [0, 33, 0, 5], [0, 1, 7, 3], [1, 2, 3, 4], seed=43)
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


def test_odd_start_reads_ring(accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, _, _ = _build(128, [1], [7], [2], ring_rows=4, seed=47)
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation)
    _check(kv, scores, state, metadata, out, actual_state, None, None)
    assert out[0].abs().sum() > 0


def test_odd_start_gathered_fallback(accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, prev_rows, prev_indices = _build(
        512, [3], [9], [2], ring_rows=4, seed=53, gathered="ring"
    )
    out, actual_state = _execute(kv, scores, state, metadata, prev_rows, prev_indices, implementation)
    _check(kv, scores, state, metadata, out, actual_state, prev_rows, prev_indices)
    assert out[0].abs().sum() > 0


def test_ring_wrap_long_tail(accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, _, _ = _build(128, [100], [31], [4], ring_rows=5, seed=59)
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


def test_special_inf(accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, _, _ = _build(128, [4], [0], [2], ring_rows=4, seed=61)
    kv[1] = float("inf")
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation)
    _check(kv, scores, state, metadata, out, actual_state, None, None)
    assert torch.isinf(out[1]).all()


def test_special_nan_idle(accuracy_backend):
    implementation, _, _ = accuracy_backend
    kv, scores, state, metadata, _, _ = _build(128, [16, 0], [0, 0], [1, 2], seed=67)
    kv[18:] = float("nan")
    scores[18:] = float("nan")
    state[2] = float("nan")
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


def test_int32_page_boundary(accuracy_backend):
    implementation, _, _ = accuracy_backend
    dim, length, page = 512, 40, 65536
    state = torch.zeros(page + 1, STATE_RING_ROWS, 2 * dim)
    state[0].fill_(1.0)
    state[page].fill_(0.5)
    tokens = length + 1
    kv = torch.linspace(-1.0, 1.0, tokens * dim).reshape(tokens, dim)
    scores = torch.linspace(1.0, -1.0, tokens * dim).reshape(tokens, dim)
    metadata = torch.zeros(5, 1, dtype=torch.int32)
    metadata[0, 0] = 0
    metadata[1, 0] = length
    metadata[4, 0] = page
    out, actual_state = _execute(kv, scores, state, metadata, None, None, implementation, max_query_len=length)
    _check(kv, scores, state, metadata, out, actual_state, None, None)


GRAPH_CASES = [(128, False), (128, True), (512, False), (512, True)]


@pytest.mark.parametrize(
    "dim,gathered",
    GRAPH_CASES,
    ids=[f"d{dim}-{'gather' if g else 'ring'}" for dim, g in GRAPH_CASES],
)
def test_graph_replay(dim, gathered, accuracy_backend):
    implementation, _, _ = accuracy_backend
    forward, _ = load_impl("compressor_from_projected", implementation)
    lengths, starts = [65, 35, 7, 0, 9, 3], [0, 29, 1, 13, 31, 3]
    blocks, capacity = [1, 2, 3, 4, 5, 6], 96
    kv, scores, state, metadata, prev_rows, prev_indices = _build(
        dim, lengths, starts, blocks, ring_rows=7, seed=307, capacity=capacity,
        gathered="mixed" if gathered else None,
    )
    tokens = kv.shape[0]
    kv_d, scores_d, metadata_d = kv.npu(), scores.npu(), metadata.npu()
    state_initial = state.npu()
    state_d, out_d = state_initial.clone(), torch.full((tokens, dim), POISON, dtype=torch.bfloat16, device="npu")
    prev_d = prev_rows.npu() if prev_rows is not None else None
    indices_d = prev_indices.npu() if prev_indices is not None else None
    cores = _platform_cores()

    def run():
        return forward(kv_d, scores_d, state_d, metadata_d, out_d, 65, cores, prev_d, indices_d)

    run()
    torch.npu.synchronize()
    eager_out, eager_state = out_d.clone(), state_d.clone()
    state_d.copy_(state_initial)
    out_d.fill_(POISON)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    # A ring residual may read a slot that K3 overwrites, so every replay must
    # start from the same pre-call state as the eager baseline.
    state_d.copy_(state_initial)
    out_d.fill_(POISON)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    assert_bitwise_equal(out_d.cpu(), eager_out.cpu(), phase="graph-out")
    assert_bitwise_equal(state_d.cpu(), eager_state.cpu(), phase="graph-state")
    _check(kv, scores, state, metadata, out_d.cpu(), state_d.cpu(), prev_rows, prev_indices)

    kv2, scores2, state2, metadata2, prev2, indices2 = _build(
        dim, [7, 9, 1, 0, 4, 1], [2, 3, 5, 7, 11, 13], blocks, ring_rows=7, seed=401,
        capacity=capacity, gathered="mixed" if gathered else None,
    )
    kv2_d, scores2_d, metadata2_d = kv2.npu(), scores2.npu(), metadata2.npu()
    state2_initial = state2.npu()
    state2_d = state2_initial.clone()
    out2_d = torch.full((tokens, dim), POISON, dtype=torch.bfloat16, device="npu")
    prev2_d = prev2.npu() if prev2 is not None else None
    indices2_d = indices2.npu() if indices2 is not None else None
    forward(kv2_d, scores2_d, state2_d, metadata2_d, out2_d, 65, cores, prev2_d, indices2_d)
    torch.npu.synchronize()

    kv_d.copy_(kv2)
    scores_d.copy_(scores2)
    metadata_d.copy_(metadata2)
    state_d.copy_(state2_initial)
    out_d.fill_(POISON)
    if gathered:
        prev_d.copy_(prev2)
        indices_d.copy_(indices2)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    assert_bitwise_equal(out_d.cpu(), out2_d.cpu(), phase="replay-out")
    assert_bitwise_equal(state_d.cpu(), state2_d.cpu(), phase="replay-state")


INVALID_CASES = [
    "kv-dtype",
    "scores-shape",
    "state-shape",
    "out-dtype",
    "metadata-shape",
    "metadata-dtype",
    "prev-without-indices",
    "non-contiguous",
    "device-mismatch",
    "launch-bounds",
]


@pytest.mark.parametrize("case", INVALID_CASES)
def test_rejects_invalid_inputs(case, accuracy_backend):
    implementation, _, _ = accuracy_backend
    dim = 128
    kv, scores, state, metadata, prev_rows, prev_indices = _build(
        128, [4, 4], [0, 1], [1, 2], seed=71, gathered="valid"
    )
    tokens = kv.shape[0]
    device = "npu"
    args = {
        "kv": kv.to(device),
        "scores": scores.to(device),
        "state_cache": state.to(device),
        "metadata": metadata.to(device),
        "out": torch.zeros(tokens, dim, dtype=torch.bfloat16, device=device),
    }
    kwargs = {"max_query_len": 4, "num_cores": 8, "implementation": implementation}
    if case == "kv-dtype":
        args["kv"] = args["kv"].half()
    elif case == "scores-shape":
        args["scores"] = args["scores"][:, : dim // 2]
    elif case == "state-shape":
        args["state_cache"] = args["state_cache"][:, : STATE_RING_ROWS - 1]
    elif case == "out-dtype":
        args["out"] = args["out"].float()
    elif case == "metadata-shape":
        args["metadata"] = args["metadata"][:4]
    elif case == "metadata-dtype":
        args["metadata"] = args["metadata"].long()
    elif case == "prev-without-indices":
        kwargs["prev_rows"] = prev_rows.to(device)
    elif case == "non-contiguous":
        wide = torch.randn(tokens, 2 * dim, device=device)
        args["kv"] = wide[:, ::2]
    elif case == "device-mismatch":
        args["kv"] = kv
    else:
        with pytest.raises(ValueError):
            F.compressor_from_projected(
                args["kv"], args["scores"], args["state_cache"], args["metadata"], args["out"],
                max_query_len=4, num_cores=0, implementation=implementation,
            )
        kwargs["max_query_len"] = tokens + 1
    with pytest.raises(ValueError):
        F.compressor_from_projected(**args, **kwargs)
