"""Tests for communication-fused GEMM operators.

Includes:
  - Single-rank correctness tests (dist not initialised → comm is identity).
  - Multi-process distributed tests that exercise AllReduce / AllGather /
    All2All / ReduceScatter.  Automatically selects **hccl + npu** when
    Ascend NPUs are available, otherwise falls back to **gloo + cpu**.

NOTE: ASCEND_RT_VISIBLE_DEVICES must be unset at the **shell level** before
running pytest, so that the session-scoped conftest fixture initialises the
NPU runtime with full device visibility.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from mojo_opset.tests.utils import bypass_not_implemented

from mojo_opset import MojoAllGatherGemm
from mojo_opset import MojoGemmAll2All
from mojo_opset import MojoGemmAllReduce
from mojo_opset import MojoGemmReduceScatter
from mojo_opset import MojoParallelEmbedding
from mojo_opset.utils.platform import get_dist_backend, get_platform

torch.manual_seed(42)

dtypes = [torch.float16, torch.bfloat16]

# ---------------------------------------------------------------------------
# Platform / backend config (reuse mojo_opset.utils.platform)
# ---------------------------------------------------------------------------
_PLATFORM = get_platform()
COMM_BACKEND = get_dist_backend()
DEVICE = _PLATFORM if _PLATFORM in ("npu", "mlu") else "cpu"

if _PLATFORM == "npu":
    _NPU_COUNT = torch.npu.device_count()
    _TEST_NPU_IDS = [int(x) for x in os.environ.get("MOJO_TEST_NPU_IDS", "4,5").split(",")]
    WORLD_SIZE = min(len(_TEST_NPU_IDS), _NPU_COUNT)
else:
    _TEST_NPU_IDS = []
    WORLD_SIZE = 2


# ===========================================================================
# Helpers
# ===========================================================================

def _make_weight_and_bias(k, n, trans_weight, dtype):
    if trans_weight:
        w = torch.randn(k, n, dtype=dtype)
    else:
        w = torch.randn(n, k, dtype=dtype)
    b = torch.randn(n, dtype=dtype)
    return w, b


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_pg(rank, world_size, master_port):
    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    if _PLATFORM == "npu":
        torch.npu.set_device(_TEST_NPU_IDS[rank])
    dist.init_process_group(
        backend=COMM_BACKEND,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def _destroy_pg():
    dist.destroy_process_group()


def _to_dev(t: torch.Tensor) -> torch.Tensor:
    """Move a CPU tensor to the current rank's device."""
    return t.to(DEVICE) if DEVICE != "cpu" else t


@pytest.fixture()
def master_port():
    return _free_port()


# ===========================================================================
# Single-rank correctness (dist NOT initialised → comm ops are identity)
# ===========================================================================

@pytest.mark.parametrize(
    "m, k, n",
    [(64, 256, 512), (128, 1024, 2048), (7, 4096, 4096)],
)
@pytest.mark.parametrize("trans_weight", [False, True])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_gemm_all_reduce(m, k, n, trans_weight, has_bias, dtype):
    x = torch.randn(m, k, dtype=dtype)
    w, b = _make_weight_and_bias(k, n, trans_weight, dtype)
    bias = b if has_bias else None
    op = MojoGemmAllReduce(weight=w, bias=bias, trans_weight=trans_weight)
    op_ref = MojoGemmAllReduce._registry.get("torch")(
        weight=w, bias=bias, trans_weight=trans_weight
    )
    op.forward_diff_with(op_ref, x, atol=0, rtol=0)


@pytest.mark.parametrize(
    "m, k, n",
    [(64, 256, 512), (128, 1024, 2048), (7, 4096, 4096)],
)
@pytest.mark.parametrize("trans_weight", [False, True])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_all_gather_gemm(m, k, n, trans_weight, has_bias, dtype):
    x = torch.randn(m, k, dtype=dtype)
    w, b = _make_weight_and_bias(k, n, trans_weight, dtype)
    bias = b if has_bias else None
    op = MojoAllGatherGemm(weight=w, bias=bias, trans_weight=trans_weight)
    op_ref = MojoAllGatherGemm._registry.get("torch")(
        weight=w, bias=bias, trans_weight=trans_weight
    )
    op.forward_diff_with(op_ref, x, atol=0, rtol=0)


@pytest.mark.parametrize(
    "m, k, n",
    [(64, 256, 512), (128, 1024, 2048), (7, 4096, 4096)],
)
@pytest.mark.parametrize("trans_weight", [False, True])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_gemm_all2all(m, k, n, trans_weight, has_bias, dtype):
    x = torch.randn(m, k, dtype=dtype)
    w, b = _make_weight_and_bias(k, n, trans_weight, dtype)
    bias = b if has_bias else None
    op = MojoGemmAll2All(weight=w, bias=bias, trans_weight=trans_weight)
    op_ref = MojoGemmAll2All._registry.get("torch")(
        weight=w, bias=bias, trans_weight=trans_weight
    )
    op.forward_diff_with(op_ref, x, atol=0, rtol=0)


@pytest.mark.parametrize(
    "m, k, n",
    [(64, 256, 512), (128, 1024, 2048), (7, 4096, 4096)],
)
@pytest.mark.parametrize("trans_weight", [False, True])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_gemm_reduce_scatter(m, k, n, trans_weight, has_bias, dtype):
    x = torch.randn(m, k, dtype=dtype)
    w, b = _make_weight_and_bias(k, n, trans_weight, dtype)
    bias = b if has_bias else None
    op = MojoGemmReduceScatter(weight=w, bias=bias, trans_weight=trans_weight)
    op_ref = MojoGemmReduceScatter._registry.get("torch")(
        weight=w, bias=bias, trans_weight=trans_weight
    )
    op.forward_diff_with(op_ref, x, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_single_rank_all_ops_equivalent(dtype):
    """In a single-rank environment all four TP-fused GEMMs equal F.linear."""
    m, k, n = 32, 64, 128
    x = torch.randn(m, k, dtype=dtype)
    w = torch.randn(n, k, dtype=dtype)
    b = torch.randn(n, dtype=dtype)
    ref = F.linear(x, w, b)
    for OpClass in (MojoGemmAllReduce, MojoAllGatherGemm, MojoGemmAll2All, MojoGemmReduceScatter):
        op = OpClass._registry.get("torch")(weight=w, bias=b, trans_weight=False)
        out = op(x)
        torch.testing.assert_close(
            out, ref, atol=0, rtol=0, msg=f"{OpClass.__name__} output mismatch"
        )


# ===========================================================================
# Multi-card distributed communication tests
#   backend : hccl (NPU) / gloo (CPU)
#   world_size : WORLD_SIZE  (auto-detected)
#
# All tensors are created on CPU in the main process, serialised via
# mp.spawn, then moved to the rank-local device inside each worker.
# Results are pulled back to CPU for assertion against the CPU reference.
# ===========================================================================

# --- GemmAllReduce (row-parallel) ---
# Each rank holds column-shard of input & corresponding weight rows.
# allreduce(sum of partial GEMMs) should equal the full GEMM.

def _worker_gemm_all_reduce(rank, world_size, port, x_full, w_full, ref):
    _init_pg(rank, world_size, port)
    try:
        K = x_full.shape[-1]
        k_local = K // world_size
        x_local = _to_dev(x_full[:, rank * k_local:(rank + 1) * k_local].contiguous())
        w_local = _to_dev(w_full[:, rank * k_local:(rank + 1) * k_local].contiguous())

        op = MojoGemmAllReduce._registry.get("torch")(
            weight=w_local, bias=None, trans_weight=False
        )
        out = op(x_local).cpu()
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    finally:
        _destroy_pg()


def test_gemm_all_reduce_comm(master_port):
    M, K, N = 32, 64, 128
    x = torch.randn(M, K, device="cpu")
    w = torch.randn(N, K, device="cpu")
    ref = F.linear(x, w)
    mp.spawn(
        _worker_gemm_all_reduce,
        args=(WORLD_SIZE, master_port, x, w, ref),
        nprocs=WORLD_SIZE,
        join=True,
    )


# --- AllGatherGemm (sequence-parallel) ---
# Each rank holds a sequence shard. AllGather reconstructs full seq, then GEMM.

def _worker_all_gather_gemm(rank, world_size, port, x_full, w, bias, ref):
    _init_pg(rank, world_size, port)
    try:
        M = x_full.shape[0]
        m_local = M // world_size
        x_local = _to_dev(x_full[rank * m_local:(rank + 1) * m_local].contiguous())
        w_dev = _to_dev(w)
        b_dev = _to_dev(bias)

        op = MojoAllGatherGemm._registry.get("torch")(
            weight=w_dev, bias=b_dev, trans_weight=False, gather_dim=0
        )
        out = op(x_local).cpu()
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    finally:
        _destroy_pg()


def test_all_gather_gemm_comm(master_port):
    M, K, N = 32, 64, 128
    x = torch.randn(M, K, device="cpu")
    w = torch.randn(N, K, device="cpu")
    b = torch.randn(N, device="cpu")
    ref = F.linear(x, w, b)
    mp.spawn(
        _worker_all_gather_gemm,
        args=(WORLD_SIZE, master_port, x, w, b, ref),
        nprocs=WORLD_SIZE,
        join=True,
    )


# --- GemmReduceScatter (TP row-parallel → SP) ---
# Each rank holds column-shard of input & weight.
# After GEMM + ReduceScatter: rank i gets shard[i] of the full reduced output.

def _worker_gemm_reduce_scatter(rank, world_size, port, x_full, w_full, ref_full):
    _init_pg(rank, world_size, port)
    try:
        K = x_full.shape[-1]
        M = x_full.shape[0]
        k_local = K // world_size
        m_local = M // world_size

        x_local = _to_dev(x_full[:, rank * k_local:(rank + 1) * k_local].contiguous())
        w_local = _to_dev(w_full[:, rank * k_local:(rank + 1) * k_local].contiguous())

        op = MojoGemmReduceScatter._registry.get("torch")(
            weight=w_local, bias=None, trans_weight=False, scatter_dim=0
        )
        out = op(x_local).cpu()

        ref_local = ref_full[rank * m_local:(rank + 1) * m_local]
        torch.testing.assert_close(out, ref_local, atol=1e-4, rtol=1e-4)
    finally:
        _destroy_pg()


def test_gemm_reduce_scatter_comm(master_port):
    M, K, N = 32, 64, 128
    x = torch.randn(M, K, device="cpu")
    w = torch.randn(N, K, device="cpu")
    ref = F.linear(x, w)
    mp.spawn(
        _worker_gemm_reduce_scatter,
        args=(WORLD_SIZE, master_port, x, w, ref),
        nprocs=WORLD_SIZE,
        join=True,
    )


# --- GemmAll2All (Ulysses-style, scatter_dim=0 gather_dim=0) ---
# Each rank computes GEMM on its sequence shard, then All2All redistributes
# chunks among ranks.

def _worker_gemm_all2all(rank, world_size, port, x_shards, w, bias, expected):
    _init_pg(rank, world_size, port)
    try:
        x_dev = _to_dev(x_shards[rank])
        w_dev = _to_dev(w)
        b_dev = _to_dev(bias)

        op = MojoGemmAll2All._registry.get("torch")(
            weight=w_dev, bias=b_dev, trans_weight=False,
            scatter_dim=0, gather_dim=0,
        )
        out = op(x_dev).cpu()
        torch.testing.assert_close(out, expected[rank], atol=1e-4, rtol=1e-4)
    finally:
        _destroy_pg()


def test_gemm_all2all_comm(master_port):
    M, K, N = 32, 64, 128
    m_local = M // WORLD_SIZE
    w = torch.randn(N, K, device="cpu")
    b = torch.randn(N, device="cpu")

    x_full = torch.randn(M, K, device="cpu")
    x_shards = [
        x_full[i * m_local:(i + 1) * m_local].contiguous()
        for i in range(WORLD_SIZE)
    ]

    gemm_outputs = [F.linear(x_shards[j], w, b) for j in range(WORLD_SIZE)]

    expected = []
    for i in range(WORLD_SIZE):
        chunks_for_rank_i = [
            gemm_outputs[j].chunk(WORLD_SIZE, dim=0)[i]
            for j in range(WORLD_SIZE)
        ]
        expected.append(torch.cat(chunks_for_rank_i, dim=0))

    mp.spawn(
        _worker_gemm_all2all,
        args=(WORLD_SIZE, master_port, x_shards, w, b, expected),
        nprocs=WORLD_SIZE,
        join=True,
    )


# --- ParallelEmbedding (vocab-parallel) ---
# Full embedding table is split along num_embeddings.  Each rank stores
# rows [start, end).  all_reduce(sum) reassembles the full lookup.

def _worker_parallel_embedding(rank, world_size, port, full_weight, ids, ref):
    _init_pg(rank, world_size, port)
    try:
        import math
        V, D = full_weight.shape
        local_size = math.ceil(V / world_size)
        start = rank * local_size
        end = min(start + local_size, V)
        local_weight = full_weight[start:end].contiguous()

        op = MojoParallelEmbedding._registry.get("torch")(
            num_embeddings=V,
            embedding_dim=D,
        )
        op.vocab_start_index = start
        op.vocab_end_index = end
        op.local_num_embeddings = end - start
        with torch.no_grad():
            op.weight = torch.nn.Parameter(local_weight)

        if DEVICE != "cpu":
            op = op.to(DEVICE)
        ids_dev = _to_dev(ids)

        out = op(ids_dev).cpu()
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    finally:
        _destroy_pg()


def test_parallel_embedding_comm(master_port):
    V, D = 128, 64
    full_weight = torch.randn(V, D, device="cpu")
    ids = torch.randint(0, V, (8, 16), device="cpu")
    ref = F.embedding(ids, full_weight)
    mp.spawn(
        _worker_parallel_embedding,
        args=(WORLD_SIZE, master_port, full_weight, ids, ref),
        nprocs=WORLD_SIZE,
        join=True,
    )
