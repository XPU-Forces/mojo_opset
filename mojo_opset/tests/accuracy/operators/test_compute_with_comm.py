"""Tests for communication-fused GEMM operators.

Multi-process distributed tests use torchrun (subprocess) to exercise
AllReduce / AllGather / All2All / ReduceScatter with real HCCL/gloo comm.

Run with:
    torchrun --nproc-per-node=2 -m pytest <this_file> -v
Or via the helper that auto-launches torchrun for comm tests.
"""

import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
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

_PLATFORM = get_platform()
COMM_BACKEND = get_dist_backend()
DEVICE = _PLATFORM if _PLATFORM in ("npu", "mlu") else "cpu"
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


def _is_dist_env():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _run_torchrun_test(test_fn_name, nproc=2, timeout=600):
    """Launch a distributed test via torchrun as subprocess."""
    import random
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            f"from mojo_opset.tests.accuracy.operators.test_compute_with_comm import {test_fn_name}\n"
            f"{test_fn_name}()\n"
        )
        script_path = f.name
    try:
        port = random.randint(29500, 39999)
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc-per-node", str(nproc),
            "--master-addr", "127.0.0.1",
            "--master-port", str(port),
            script_path,
        ]
        env = os.environ.copy()
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            pytest.fail(f"torchrun test {test_fn_name} failed:\n{result.stderr[-3000:]}")
    finally:
        os.unlink(script_path)


def _init_dist():
    """Initialize dist for torchrun-launched processes."""
    rank = int(os.environ["LOCAL_RANK"])
    if _PLATFORM == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(rank)
    dist.init_process_group(backend=COMM_BACKEND)
    return rank, dist.get_world_size()


def _to_dev(t: torch.Tensor) -> torch.Tensor:
    return t.to(DEVICE) if DEVICE != "cpu" else t


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
# Multi-card distributed tests (torchrun-based)
# ===========================================================================

def _dist_all_gather_gemm():
    """Worker: each rank holds sequence shard, AllGather + GEMM produces full result."""
    rank, world_size = _init_dist()

    test_cases = [
        (4096, 4096, 4096, torch.float16, True),
        (2048, 4096, 8192, torch.float16, True),
        (8192, 2048, 4096, torch.float16, True),
        # bf16 without bias: torch_npu F.linear fused bias differs from unfused add
        (4096, 4096, 4096, torch.bfloat16, False),
    ]

    for M, K, N, dtype, use_bias in test_cases:
        torch.manual_seed(42)
        x_full = torch.randn(M, K, dtype=dtype)
        w = torch.randn(N, K, dtype=dtype)
        b = torch.randn(N, dtype=dtype) if use_bias else None

        m_local = M // world_size
        x_local = _to_dev(x_full[rank * m_local:(rank + 1) * m_local].contiguous())
        w_dev = _to_dev(w)
        b_dev = _to_dev(b) if b is not None else None

        torch_cls = MojoAllGatherGemm._registry.get("torch")
        ref_op = torch_cls(weight=w_dev, bias=b_dev, trans_weight=False, gather_dim=0)
        ref = ref_op(x_local)

        op = MojoAllGatherGemm(weight=w_dev, bias=b_dev, trans_weight=False, gather_dim=0)
        out = op(x_local)

        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
        if rank == 0:
            print(f"[PASS] AllGatherGemm M={M} K={K} N={N} dtype={dtype}, shape={out.shape}")

    dist.destroy_process_group()


# TODO: Remove skip once triton-dist is added to mojo CI dependencies
@pytest.mark.skip(reason="requires triton-dist which is not yet in CI")
def test_all_gather_gemm_comm():
    if _is_dist_env():
        _dist_all_gather_gemm()
    else:
        _run_torchrun_test("_dist_all_gather_gemm")


def _dist_gemm_all_reduce():
    """Worker: each rank holds column-shard, GEMM + AllReduce produces full result."""
    rank, world_size = _init_dist()

    test_cases = [
        (4096, 4096, 4096, torch.float16),
        (2048, 8192, 4096, torch.float16),
        (8192, 4096, 2048, torch.float16),
        (4096, 4096, 4096, torch.bfloat16),
    ]

    for M, K, N, dtype in test_cases:
        k_local = K // world_size

        torch.manual_seed(42 + rank)
        x_local = _to_dev(torch.randn(M, k_local, dtype=dtype))
        w_local = _to_dev(torch.randn(k_local, N, dtype=dtype))

        torch_cls = MojoGemmAllReduce._registry.get("torch")
        ref_op = torch_cls(weight=w_local, bias=None, trans_weight=True)
        ref = ref_op(x_local)

        op = MojoGemmAllReduce(weight=w_local, bias=None, trans_weight=True)
        out = op(x_local)

        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
        if rank == 0:
            print(f"[PASS] GemmAllReduce M={M} K={K} N={N} dtype={dtype}, shape={out.shape}")

    dist.destroy_process_group()


# TODO: Remove skip once triton-dist is added to mojo CI dependencies
@pytest.mark.skip(reason="requires triton-dist which is not yet in CI")
def test_gemm_all_reduce_comm():
    if _is_dist_env():
        _dist_gemm_all_reduce()
    else:
        _run_torchrun_test("_dist_gemm_all_reduce")


def _dist_gemm_reduce_scatter():
    """Worker: each rank holds column-shard, GEMM + ReduceScatter scatters result."""
    rank, world_size = _init_dist()

    test_cases = [
        (4096, 4096, 4096, torch.float16),
        (2048, 8192, 4096, torch.float16),
        (8192, 4096, 2048, torch.float16),
        (4096, 4096, 4096, torch.bfloat16),
    ]

    for M, K, N, dtype in test_cases:
        k_local = K // world_size

        torch.manual_seed(42 + rank)
        x_local = _to_dev(torch.randn(M, k_local, dtype=dtype))
        w_local = _to_dev(torch.randn(k_local, N, dtype=dtype))

        torch_cls = MojoGemmReduceScatter._registry.get("torch")
        ref_op = torch_cls(weight=w_local, bias=None, trans_weight=True, scatter_dim=0)
        ref = ref_op(x_local)

        op = MojoGemmReduceScatter(weight=w_local, bias=None, trans_weight=True, scatter_dim=0)
        out = op(x_local)

        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
        if rank == 0:
            print(f"[PASS] GemmReduceScatter M={M} K={K} N={N} dtype={dtype}, shape={out.shape}")

    dist.destroy_process_group()


# TODO: Remove skip once triton-dist is added to mojo CI dependencies
@pytest.mark.skip(reason="requires triton-dist which is not yet in CI")
def test_gemm_reduce_scatter_comm():
    if _is_dist_env():
        _dist_gemm_reduce_scatter()
    else:
        _run_torchrun_test("_dist_gemm_reduce_scatter")


def _dist_gemm_all2all():
    """Worker: each rank computes GEMM on its shard, All2All redistributes."""
    rank, world_size = _init_dist()

    torch.manual_seed(42)
    M, K, N = 32, 64, 128
    dtype = torch.float32
    m_local = M // world_size
    x_full = torch.randn(M, K, dtype=dtype)
    w = torch.randn(N, K, dtype=dtype)
    b = torch.randn(N, dtype=dtype)

    x_shards = [x_full[i * m_local:(i + 1) * m_local].contiguous() for i in range(world_size)]
    gemm_outputs = [F.linear(x_shards[j], w, b) for j in range(world_size)]
    expected = []
    for i in range(world_size):
        chunks = [gemm_outputs[j].chunk(world_size, dim=0)[i] for j in range(world_size)]
        expected.append(torch.cat(chunks, dim=0))

    x_dev = _to_dev(x_shards[rank])
    w_dev = _to_dev(w)
    b_dev = _to_dev(b)

    op = MojoGemmAll2All(weight=w_dev, bias=b_dev, trans_weight=False, scatter_dim=0, gather_dim=0)
    out = op(x_dev).cpu()

    torch.testing.assert_close(out, expected[rank], atol=1e-4, rtol=1e-4)
    if rank == 0:
        print(f"[PASS] GemmAll2All comm test, shape={out.shape}")
    dist.destroy_process_group()


def test_gemm_all2all_comm():
    if _is_dist_env():
        _dist_gemm_all2all()
    else:
        _run_torchrun_test("_dist_gemm_all2all")


def _dist_parallel_embedding():
    """Worker: vocab-parallel embedding with allreduce."""
    rank, world_size = _init_dist()
    import math

    torch.manual_seed(42)
    V, D = 128, 64
    full_weight = torch.randn(V, D)
    ids = torch.randint(0, V, (8, 16))
    ref = F.embedding(ids, full_weight)

    local_size = math.ceil(V / world_size)
    start = rank * local_size
    end = min(start + local_size, V)
    local_weight = full_weight[start:end].contiguous()

    op = MojoParallelEmbedding(num_embeddings=V, embedding_dim=D)
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
    if rank == 0:
        print(f"[PASS] ParallelEmbedding comm test, shape={out.shape}")
    dist.destroy_process_group()


def test_parallel_embedding_comm():
    if _is_dist_env():
        _dist_parallel_embedding()
    else:
        _run_torchrun_test("_dist_parallel_embedding")
