"""Tests for functions.get_compressed_slot_mapping.

The op emits int64 slot indices, so docs/operator_accuracy_standard.md §1 does
not apply (integer compute is out of scope) and every case is pinned bit-exactly
against the torch reference (``|diff| == 0``).
"""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

# Merged path: decode 64/256/2048 requests x 1 token and spec-decode 256x4.
# Split path: prefill 1x1024, mixed 2x2048, prefill 1x8192, mixed 4x2048.
SHAPES = [
    ("decode_64x1_cr2", [1] * 64, 4096, 64, 2),
    ("decode_256x1_cr2", [1] * 256, 8192, 64, 2),
    ("decode_2048x1_cr2", [1] * 2048, 8192, 64, 2),
    ("decode_256x4_cr2", [4] * 256, 8192, 64, 2),
    ("prefill_1x1024_cr2", [1024], 4096, 64, 2),
    ("mixed_2x2048_cr2", [2048] * 2, 8192, 64, 2),
    ("prefill_1x8192_cr2", [8192], 16384, 64, 2),
    ("mixed_4x2048_cr4", [2048] * 4, 16384, 64, 4),
]

VARLEN = [
    [1],
    [0, 257, 1, 0, 3],
    [6] * 64,
    [2, 6, 4, 0],
    [0, 0, 0],
    # num_computed=240, q=40, cr=4, bs=64 -> compressed states 60..69 cross the
    # storage block boundary at 64 (vllm regression :548).
    [40],
]

PARAM_GRID = [(cr, bs) for cr in (2, 4, 16, 32, 128) for bs in (2, 64, 128)]


def _loop_reference(
    num_tokens,
    slot_mapping,
    query_start_loc,
    seq_lens,
    block_table,
    block_size,
    compress_ratio,
    dcp_world_size=1,
    cp_interleave=1,
    buffer_size=None,
):
    """Literal per-request replica of the vllm kernel (CPU oracle for the torch provider)."""
    buffer_size = num_tokens if buffer_size is None else buffer_size
    out = torch.full((buffer_size,), -1, dtype=torch.int64)
    block_table = block_table.to(torch.int64)
    for request in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        if end <= start:
            continue
        query_len = end - start
        token = torch.arange(start, end, dtype=torch.int64)
        pos = int(seq_lens[request]) - query_len + (token - start)
        slot = slot_mapping[start:end].to(torch.int64)
        valid = ((pos + 1) % compress_ratio == 0) & (slot != -1)
        local_pos = pos // (cp_interleave * dcp_world_size) * cp_interleave + pos % cp_interleave
        pac = local_pos // compress_ratio
        block_number = block_table[request][pac // block_size]
        out[start:end] = torch.where(valid, block_number * block_size + pac % block_size, -1)
    return out


def _make_case(query_lens, seq_upper, block_size, compress_ratio, seed):
    """Build CPU inputs; ``seq_lens`` never drops below the request's query length."""
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    torch.manual_seed(seed)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(torch.tensor(query_lens, dtype=torch.int64), 0).to(torch.int32)
    seq_lens = torch.randint(seq_upper // 2, seq_upper + 1, (num_reqs,))
    seq_lens = torch.maximum(seq_lens, torch.tensor(query_lens, dtype=torch.int64))
    max_states = int(seq_lens.max()) // compress_ratio + 1
    width = (max_states + block_size - 1) // block_size
    block_table = torch.randint(0, 4096, (num_reqs, width), dtype=torch.int32)
    slot_mapping = torch.randint(0, 1 << 30, (num_tokens,), dtype=torch.int64)
    slot_mapping[::17] = -1  # SWA bounded replay PAD inheritance
    # per-token positions in query order within each request
    positions = torch.cat([
        torch.arange(int(seq_lens[i]) - query_lens[i], int(seq_lens[i]))
        for i in range(num_reqs)
    ]).to(torch.int64)
    return {
        "num_tokens": num_tokens,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens.to(torch.int32),
        "block_table": block_table,
        "block_size": block_size,
        "compress_ratio": compress_ratio,
        "max_query_len": max(query_lens) if query_lens else 0,
    }


def _run(implementation, case, device, buffer_size=None, use_max_query_len=True, tag=""):
    kwargs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in case.items()}
    if not use_max_query_len:
        kwargs["max_query_len"] = None
    expected_kwargs = dict(kwargs)
    expected_kwargs.pop("out", None)
    expected_kwargs.pop("positions", None)
    if buffer_size is not None:
        actual = torch.full((buffer_size,), 0x7F7F7F7F, dtype=torch.int64, device=device)
        expected = torch.full((buffer_size,), 0x7F7F7F7F, dtype=torch.int64, device=device)
        returned = F.get_compressed_slot_mapping(
            **kwargs, out=actual, implementation=implementation
        )
        assert returned.data_ptr() == actual.data_ptr(), "persistent buffer must be reused"
        F.get_compressed_slot_mapping(**expected_kwargs, out=expected, implementation="torch")
        assert torch.equal(actual, expected), f"buffer mismatch: {tag}"
    else:
        actual = F.get_compressed_slot_mapping(**kwargs, implementation=implementation)
        expected = F.get_compressed_slot_mapping(**expected_kwargs, implementation="torch")
        assert torch.equal(actual, expected), f"mismatch: {tag}"
    return actual


@pytest.mark.reference
@pytest.mark.parametrize("query_lens,seq_upper,block_size,compress_ratio", [
    ([1] * 8, 64, 64, 2),
    ([3, 17, 1, 0, 64], 512, 2, 2),
    ([40], 280, 64, 4),
])
def test_reference_matches_loop_semantics(query_lens, seq_upper, block_size, compress_ratio):
    """The vectorised torch provider must equal the literal per-request loop."""
    case = _make_case(query_lens, seq_upper, block_size, compress_ratio, seed=11)
    actual = F.get_compressed_slot_mapping(**case, implementation="torch")
    expected = _loop_reference(
        case["num_tokens"],
        case["slot_mapping"],
        case["query_start_loc"],
        case["seq_lens"],
        case["block_table"],
        block_size,
        compress_ratio,
    )
    assert torch.equal(actual, expected)


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
@pytest.mark.parametrize("name,query_lens,seq_upper,block_size,compress_ratio", SHAPES)
@pytest.mark.parametrize("buffer_size", [None, 8192])
def test_get_compressed_slot_mapping_shapes(
    accuracy_backend, name, query_lens, seq_upper, block_size, compress_ratio, buffer_size
):
    implementation, _, device = accuracy_backend
    case = _make_case(query_lens, seq_upper, block_size, compress_ratio, seed=0)
    _run(implementation, case, device, buffer_size=buffer_size, tag=f"{name}+buf{buffer_size}")


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
@pytest.mark.parametrize("query_lens", VARLEN, ids=lambda lens: f"{len(lens)}x{max(lens)}")
def test_get_compressed_slot_mapping_varlen(accuracy_backend, query_lens):
    implementation, _, device = accuracy_backend
    max_query = max(query_lens) if query_lens else 1
    case = _make_case(query_lens, max(max_query, 4096), 64, 4, seed=1)
    _run(implementation, case, device, tag=f"varlen{query_lens}")


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
@pytest.mark.parametrize("query_len", [128, 129])
@pytest.mark.parametrize("use_max_query_len", [True, False])
def test_get_compressed_slot_mapping_dispatch_threshold(
    accuracy_backend, query_len, use_max_query_len
):
    """128 uses the merged tile, 129 the split path; no hint uses the original structure.

    96 requests keep cdiv(num_reqs, rpp) >= min_programs satisfied (24 programs
    at block_q=128), so the q_len threshold alone flips 128 -> merged, 129 -> split.
    """
    implementation, _, device = accuracy_backend
    case = _make_case([query_len] * 96, 4096, 64, 4, seed=2)
    _run(
        implementation,
        case,
        device,
        use_max_query_len=use_max_query_len,
        tag=f"q{query_len}-mq{use_max_query_len}",
    )


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
@pytest.mark.parametrize("compress_ratio,block_size", PARAM_GRID)
def test_get_compressed_slot_mapping_parameter_grid(accuracy_backend, compress_ratio, block_size):
    implementation, _, device = accuracy_backend
    case = _make_case([3, 17, 1, 0, 64], 4096, block_size, compress_ratio, seed=3)
    _run(implementation, case, device, tag=f"cr{compress_ratio}-bs{block_size}")


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
@pytest.mark.parametrize("name,query_lens,seq_upper,block_size,compress_ratio", SHAPES[:4])
def test_get_compressed_slot_mapping_garbage_reuse(
    accuracy_backend, name, query_lens, seq_upper, block_size, compress_ratio
):
    """``out=None`` must overwrite every element of a recycled allocation."""
    implementation, _, device = accuracy_backend
    case = _make_case(query_lens, seq_upper, block_size, compress_ratio, seed=4)
    junk = torch.full((case["num_tokens"],), 0x7F7F7F7F, dtype=torch.int64, device=device)
    del junk  # hand the dirty block back to the caching allocator
    actual = F.get_compressed_slot_mapping(
        **{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in case.items()},
        implementation=implementation,
    )
    expected = F.get_compressed_slot_mapping(
        **case, implementation="torch"
    )
    assert torch.equal(actual, expected.to(device)), f"garbage alloc: {name}"


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
def test_get_compressed_slot_mapping_swa_replay(accuracy_backend):
    """SWA bounded replay: a PAD token slot propagates PAD to the compressed slot."""
    implementation, _, device = accuracy_backend
    actual = F.get_compressed_slot_mapping(
        8,
        torch.tensor([-1, -1, -1, -1, 0, 0, 0, 0], dtype=torch.int64, device=device),
        torch.tensor([0, 8], dtype=torch.int32, device=device),
        torch.tensor([8], dtype=torch.int32, device=device),
        torch.tensor([[3]], dtype=torch.int32, device=device),
        block_size=4,
        compress_ratio=2,
        implementation=implementation,
    )
    assert actual.tolist() == [-1, -1, -1, -1, -1, 3 * 4 + 2, -1, 3 * 4 + 3]


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("decode", [False, True])
def test_get_compressed_slot_mapping_dcp(accuracy_backend, rank, decode):
    """DCP rank-local state positions, on both the merged and the decode path."""
    implementation, _, device = accuracy_backend
    dcp, interleave, ratio, block_size, num_tokens = 2, 4, 2, 4, 32
    pos = torch.arange(num_tokens, device=device)
    owned = (pos // interleave) % dcp == rank
    slot_mapping = torch.where(owned, 0, -1).to(torch.int64)
    if decode:
        # Decode path: one request per token, so each block-table row is offset
        # by its request index.
        block_table = torch.tensor([[5, 7]]) + torch.arange(num_tokens)[:, None]
        actual = F.get_compressed_slot_mapping(
            num_tokens,
            slot_mapping,
            torch.arange(num_tokens + 1, dtype=torch.int32, device=device),
            torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device),
            block_table.to(torch.int32).to(device),
            block_size=block_size,
            compress_ratio=ratio,
            dcp_world_size=dcp,
            cp_interleave=interleave,
            max_query_len=1,
            implementation=implementation,
        )
    else:
        # Merged path: a single request covering all tokens.
        block_table = torch.tensor([[5, 7]]).expand(num_tokens, 2)
        actual = F.get_compressed_slot_mapping(
            num_tokens,
            slot_mapping,
            torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
            torch.tensor([num_tokens], dtype=torch.int32, device=device),
            torch.tensor([[5, 7]], dtype=torch.int32, device=device),
            block_size=block_size,
            compress_ratio=ratio,
            dcp_world_size=dcp,
            cp_interleave=interleave,
            max_query_len=num_tokens,
            implementation=implementation,
        )
    local_pos = pos.cpu() // (interleave * dcp) * interleave + pos.cpu() % interleave
    state = local_pos // ratio
    block_number = block_table.to(torch.int64)[torch.arange(num_tokens), state // block_size]
    expected = torch.where(
        owned.cpu() & ((pos.cpu() + 1) % ratio == 0),
        block_number * block_size + state % block_size,
        torch.tensor(-1, dtype=torch.int64),
    )
    assert actual.cpu().tolist() == expected.tolist()


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
def test_get_compressed_slot_mapping_block_cross(accuracy_backend):
    """vllm :548 — compressed states 60..69 straddle the block boundary at 64."""
    implementation, _, device = accuracy_backend
    actual = F.get_compressed_slot_mapping(
        40,
        torch.full((40,), -123, dtype=torch.int64, device=device),
        torch.tensor([0, 40], dtype=torch.int32, device=device),
        torch.tensor([280], dtype=torch.int32, device=device),
        torch.tensor([[5, 7]], dtype=torch.int32, device=device),
        block_size=64,
        compress_ratio=4,
        max_query_len=40,
        implementation=implementation,
    )
    valid = actual[actual >= 0]
    expected = [5 * 64 + 60, 5 * 64 + 61, 5 * 64 + 62, 5 * 64 + 63] + [
        7 * 64 + i for i in range(6)
    ]
    assert valid.numel() == 10 and valid.tolist() == expected


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
def test_get_compressed_slot_mapping_draft_decode(accuracy_backend):
    """vllm :106 — cr=16 draft decode is all-PAD unless the sequence divides."""
    implementation, _, device = accuracy_backend
    common = dict(
        slot_mapping=torch.arange(3, dtype=torch.int64, device=device),
        query_start_loc=torch.arange(4, dtype=torch.int32, device=device),
        block_table=torch.arange(12, dtype=torch.int32, device=device).view(3, 4),
        block_size=64,
        compress_ratio=16,
        max_query_len=1,
    )
    padded = F.get_compressed_slot_mapping(
        3,
        seq_lens=torch.tensor([17, 33, 65], dtype=torch.int32, device=device),
        implementation=implementation,
        **common,
    )
    assert (padded == -1).all()
    divisible = F.get_compressed_slot_mapping(
        3,
        seq_lens=torch.tensor([32, 48, 80], dtype=torch.int32, device=device),
        implementation=implementation,
        **common,
    )
    expected = torch.tensor([0, 4, 8], dtype=torch.int64) * 64 + torch.tensor([1, 2, 4])
    assert torch.equal(divisible.cpu(), expected)


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
def test_get_compressed_slot_mapping_empty_batch(accuracy_backend):
    """Empty tensor (docs §1.4): nothing is written, the persistent buffer stays -1."""
    implementation, _, device = accuracy_backend
    out = torch.full((8,), 0x7F7F7F7F, dtype=torch.int64, device=device)
    F.get_compressed_slot_mapping(
        0,
        torch.zeros(0, dtype=torch.int64, device=device),
        torch.tensor([0, 0], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([[1]], dtype=torch.int32, device=device),
        block_size=64,
        compress_ratio=2,
        out=out,
        max_query_len=0,
        implementation=implementation,
    )
    assert (out == -1).all()


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
@pytest.mark.parametrize("name,query_lens,seq_upper,block_size,compress_ratio", [
    ("prefill_1x8192_cr2", [8192], 16384, 64, 2),
    ("mixed_2x2048_cr2", [2048] * 2, 8192, 64, 2),
    ("decode_256x1_cr2", [1] * 256, 8192, 64, 2),
])
def test_get_compressed_slot_mapping_paired_slot_contig(accuracy_backend, name, query_lens,
                                                        seq_upper, block_size, compress_ratio):
    """Paired line: kernel-produced slot_contig equals the elementwise contract."""
    implementation, _, device = accuracy_backend
    case = _make_case(query_lens, seq_upper, block_size, compress_ratio, seed=9)
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in case.items()}
    contig = torch.zeros(case["num_tokens"], dtype=torch.uint8, device=device)
    F.get_compressed_slot_mapping(
        num_tokens=kwargs["num_tokens"],
        slot_mapping=kwargs["slot_mapping"],
        query_start_loc=kwargs["query_start_loc"],
        seq_lens=kwargs["seq_lens"],
        block_table=kwargs["block_table"],
        block_size=kwargs["block_size"],
        compress_ratio=kwargs["compress_ratio"],
        dcp_world_size=1,
        cp_interleave=1,
        max_query_len=kwargs["max_query_len"],
        positions=kwargs["positions"],
        slot_contig=contig,
        implementation=implementation,
    )
    slots = F.get_compressed_slot_mapping(
        num_tokens=kwargs["num_tokens"],
        slot_mapping=kwargs["slot_mapping"],
        query_start_loc=kwargs["query_start_loc"],
        seq_lens=kwargs["seq_lens"],
        block_table=kwargs["block_table"],
        block_size=kwargs["block_size"],
        compress_ratio=kwargs["compress_ratio"],
        dcp_world_size=1,
        cp_interleave=1,
        max_query_len=kwargs["max_query_len"],
        implementation=implementation,
    )
    pos = kwargs["positions"]
    prev_slot = torch.cat((torch.tensor([-2], device=device), slots[:-1]))
    prev_pos = torch.cat((torch.tensor([-2], device=device), pos[:-1]))
    expected = (
        (slots != -1)
        & (slots == prev_slot + 1)
        & (pos == prev_pos + 1)
        & ((pos + 1) % compress_ratio == 0)
    ).to(torch.uint8)
    assert torch.equal(contig, expected), name


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.bitwise
def test_get_compressed_slot_mapping_repeatable(accuracy_backend):
    implementation, _, device = accuracy_backend
    case = _make_case([128] * 4, 4096, 64, 4, seed=5)
    kwargs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in case.items()
    }
    assert_repeatable(
        partial(F.get_compressed_slot_mapping, **kwargs, implementation=implementation),
        (),
        backward=False,
    )


@pytest.mark.api("functions.get_compressed_slot_mapping")
def test_get_compressed_slot_mapping_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("get_compressed_slot_mapping", implementation="triton", target=target)
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    case = _make_case([4] * 8, 512, 64, 2, seed=6)
    args = (
        case["num_tokens"],
        case["slot_mapping"].to(device),
        case["query_start_loc"].to(device),
        case["seq_lens"].to(device),
        case["block_table"].to(device),
        case["block_size"],
        case["compress_ratio"],
        torch.empty(case["num_tokens"], dtype=torch.int64, device=device),
        1,
        1,
        case["max_query_len"],
        torch.empty(case["num_tokens"], dtype=torch.uint8, device=device),
        case["positions"].to(device),
    )
    torch.library.opcheck(
        namespace.get_compressed_slot_mapping_fwd.default,
        args,
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_get_compressed_slot_mapping_compiled(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU Triton compile contract")
    implementation = implementation or "triton"
    preload("get_compressed_slot_mapping", implementation=implementation)
    case = _make_case([4] * 16, 512, 64, 2, seed=7)
    kwargs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in case.items()
    }

    def run(slot_mapping, query_start_loc, seq_lens, block_table):
        output = F.get_compressed_slot_mapping(
            case["num_tokens"],
            slot_mapping,
            query_start_loc,
            seq_lens,
            block_table,
            case["block_size"],
            case["compress_ratio"],
            None,
            1,
            1,
            case["max_query_len"],
            implementation=implementation,
        )
        return (output,)
    inputs = (
        kwargs["slot_mapping"],
        kwargs["query_start_loc"],
        kwargs["seq_lens"],
        kwargs["block_table"],
    )
    try:
        (eager,) = run(*inputs)
        (actual,) = compile_fullgraph(run, backend=backend)(*inputs)
        assert torch.equal(actual, eager)
        expected = F.get_compressed_slot_mapping(**case, implementation="torch")
        assert torch.equal(actual, expected.to(device))
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
@pytest.mark.parametrize("use_max_query_len", [True, False], ids=["batched", "native"])
def test_get_compressed_slot_mapping_block_table_row_stride(accuracy_backend, use_max_query_len):
    """Page-table rows padded to a wider stride: only stride(0) is read, so the
    row offset must stay correct well past the compact table's size."""
    implementation, _, device = accuracy_backend
    case = _make_case([32] * 128, 4096, 64, 2, seed=12)
    width = case["block_table"].shape[1]
    pages = torch.empty(128, 4096, dtype=torch.int32, device=device)
    block_table = pages[:, :width]
    assert block_table.stride() == (4096, 1)
    block_table.copy_(case["block_table"].to(device))
    case["block_table"] = block_table
    _run(implementation, case, device, use_max_query_len=use_max_query_len, tag="row_stride")


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
def test_get_compressed_slot_mapping_rejects_block_table_column_stride(accuracy_backend):
    """A page-table view strided along the page axis is rejected at the entry."""
    implementation, _, device = accuracy_backend
    case = _make_case([4], 64, 4, 2, seed=13)
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in case.items()}
    width = kwargs["block_table"].shape[1]
    wide = torch.zeros(kwargs["block_table"].shape[0], width * 2, dtype=torch.int32, device=device)
    kwargs["block_table"] = wide[:, ::2]
    assert kwargs["block_table"].stride(1) == 2
    with pytest.raises(ValueError, match="page dimension"):
        F.get_compressed_slot_mapping(**kwargs, implementation=implementation)


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
def test_get_compressed_slot_mapping_rejects_bad_positions(accuracy_backend):
    """``positions`` must be int32/int64 with one entry per token."""
    implementation, _, device = accuracy_backend
    case = _make_case([8], 64, 4, 2, seed=14)
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in case.items()}
    contig = torch.zeros(case["num_tokens"], dtype=torch.uint8, device=device)
    for positions in (kwargs["positions"].to(torch.float32), kwargs["positions"][:-1]):
        with pytest.raises(ValueError, match="positions"):
            F.get_compressed_slot_mapping(
                **{**kwargs, "positions": positions, "slot_contig": contig},
                implementation=implementation,
            )


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.accuracy
def test_get_compressed_slot_mapping_rejects_bad_slot_contig(accuracy_backend):
    """``slot_contig`` must be uint8/int32/int64 with one entry per token."""
    implementation, _, device = accuracy_backend
    case = _make_case([8], 64, 4, 2, seed=15)
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in case.items()}
    for contig in (
        torch.zeros(case["num_tokens"], dtype=torch.bool, device=device),
        torch.zeros(case["num_tokens"] - 1, dtype=torch.uint8, device=device),
    ):
        with pytest.raises(ValueError, match="slot_contig"):
            F.get_compressed_slot_mapping(
                **{**kwargs, "slot_contig": contig}, implementation=implementation
            )
