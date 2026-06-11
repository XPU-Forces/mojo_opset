"""Distributed correctness tests for MojoAllGatherQuantGemm and MojoQuantGemmReduceScatter.

Each worker constructs the op (current backend) and a torch-backend reference,
``forward_diff_with`` compares the two. When neither op has a device-specific
backend registered, all cases skip via ``bypass_not_implemented``.

Shape coverage targets the M13 prefill seq=4k Megatron path:
  * AllGatherQuantGemm: QKV projection (input shard along seq, weight unsharded)
  * QuantGemmReduceScatter: Out projection (input column-shard, output seq-scatter)
"""

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mojo_opset import MojoAllGatherQuantGemm
from mojo_opset import MojoQuantGemmReduceScatter
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_dist_backend, get_platform, get_torch_device

torch.manual_seed(42)

dtypes = [torch.bfloat16]

_PLATFORM = get_platform()
DEVICE = get_torch_device()
COMM_BACKEND = "nccl" if DEVICE == "cuda" else get_dist_backend()

if _PLATFORM == "npu":
    _NPU_COUNT = torch.npu.device_count()
    _TEST_NPU_IDS = [int(x) for x in os.environ.get("MOJO_TEST_NPU_IDS", "4,5").split(",")]
    WORLD_SIZE = min(len(_TEST_NPU_IDS), _NPU_COUNT)
else:
    _TEST_NPU_IDS = []
    WORLD_SIZE = 2


# (seq_full, in_features, out_features) for AllGather + QuantGemm.
# seq_full and in_features divisible by WORLD_SIZE; in_features for the
# AllGather case is the full hidden dim H since AG is on the seq axis.
_AG_SHAPES = [
    (8, 16, 32),                # tiny smoke
    (32, 256, 512),             # small
    (128, 1024, 2048),          # medium
    (4096, 3584, 4096),         # M13 prefill QKV: H=3584 → N_qkv=4096 (full attn)
    (4096, 3584, 8192),         # SWA QKV variant: larger N
    (4096, 3584, 3072),         # GQA QKV variant: smaller N
    (1, 3584, 4096),            # decode batch=1
    (128, 3584, 4096),          # small batch M13
]

# (seq_full, in_features, out_features) for QuantGemm + ReduceScatter.
# seq_full divisible by WORLD_SIZE; in_features is K_local (per-rank shard
# of K_global), out_features is the unsharded hidden dim H.
_RS_SHAPES = [
    (8, 16, 32),                # tiny smoke
    (32, 256, 512),             # small
    (128, 1024, 2048),          # medium
    (4096, 4096, 3584),         # M13 prefill Out proj: K_local=4096 → H=3584
    (4096, 3072, 3584),         # GQA Out proj
    (4096, 8192, 3584),         # SWA Out proj
    (1, 4096, 3584),            # decode
    (128, 4096, 3584),          # small batch
]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_pg(rank, world_size, master_port):
    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    if _PLATFORM == "npu":
        torch.npu.set_device(_TEST_NPU_IDS[rank])
        dist.init_process_group(
            backend=COMM_BACKEND,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    elif DEVICE == "cuda":
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend=COMM_BACKEND,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device("cuda", rank),
        )
    else:
        dist.init_process_group(
            backend=COMM_BACKEND,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )


def _destroy_pg():
    dist.destroy_process_group()


def _to_dev(t: torch.Tensor) -> torch.Tensor:
    return t.to(DEVICE) if DEVICE != "cpu" else t


@pytest.fixture()
def master_port():
    return _free_port()


def _skip_if_no_specific_backend(op_cls):
    default_cls = op_cls._registry.get(None)
    torch_cls = op_cls._registry.get("torch")
    if default_cls is torch_cls:
        raise NotImplementedError(
            f"No device-specific backend for {op_cls.__name__}; "
            "both operands resolve to the same implementation."
        )


def _build_pair_ag(in_features, out_features, trans_weight, weight_int8, weight_scale):
    op = MojoAllGatherQuantGemm(
        in_features=in_features,
        out_features=out_features,
        trans_weight=trans_weight,
        process_group=dist.group.WORLD,
        gather_dim=0,
    )
    op = op.to(DEVICE) if DEVICE != "cpu" else op
    op_ref = MojoAllGatherQuantGemm._registry.get("torch")(
        in_features=in_features,
        out_features=out_features,
        trans_weight=trans_weight,
        process_group=dist.group.WORLD,
        gather_dim=0,
    )
    op_ref = op_ref.to(DEVICE) if DEVICE != "cpu" else op_ref
    state = {
        "weight": _to_dev(weight_int8),
        "weight_scale": _to_dev(weight_scale),
    }
    op.load_state_dict(state)
    op_ref.load_state_dict(state)
    return op, op_ref

def _build_pair_rs(in_features, out_features, trans_weight, weight_int8, weight_scale):
    op = MojoQuantGemmReduceScatter(
        in_features=in_features,
        out_features=out_features,
        trans_weight=trans_weight,
        process_group=dist.group.WORLD,
        scatter_dim=0,
    )
    op = op.to(DEVICE) if DEVICE != "cpu" else op
    op_ref = MojoQuantGemmReduceScatter._registry.get("torch")(
        in_features=in_features,
        out_features=out_features,
        trans_weight=trans_weight,
        process_group=dist.group.WORLD,
        scatter_dim=0,
    )
    op_ref = op_ref.to(DEVICE) if DEVICE != "cpu" else op_ref
    state = {
        "weight": _to_dev(weight_int8),
        "weight_scale": _to_dev(weight_scale),
    }
    op.load_state_dict(state)
    op_ref.load_state_dict(state)
    return op, op_ref


# ===========================================================================
# AllGatherQuantGemm — multi-rank
# ===========================================================================

def _worker_ag(rank, world_size, port, shape, trans_weight, non_contig):
    _init_pg(rank, world_size, port)
    exit_code = 1
    try:
        seq_full, in_features, out_features = shape
        seq_local = seq_full // world_size

        weight_shape = (
            (out_features, in_features) if trans_weight else (in_features, out_features)
        )
        weight_int8 = torch.randint(-128, 127, weight_shape, dtype=torch.int8)
        weight_scale = torch.rand(out_features, dtype=torch.bfloat16) + 0.5

        op, op_ref = _build_pair_ag(
            in_features, out_features, trans_weight, weight_int8, weight_scale,
        )

        if non_contig:
            big = torch.randint(-128, 127, (seq_local, in_features * 2), dtype=torch.int8)
            x_local = _to_dev(big)[:, ::2]
            assert not x_local.is_contiguous()
        else:
            x_local = _to_dev(
                torch.randint(-128, 127, (seq_local, in_features), dtype=torch.int8)
            )
        scale_local = _to_dev(torch.rand(seq_local, dtype=torch.float32) + 0.1)

        expected = op_ref(x_local, scale_local)
        actual = op(x_local, scale_local)
        torch.testing.assert_close(
            actual.to(torch.float32),
            expected.to(torch.float32),
            atol=1,
            rtol=2e-3,
        )

        exit_code = 0
    finally:
        try:
            if dist.is_initialized():
                _destroy_pg()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        # Avoid fragile NVSHMEM / symmetric memory Python destructors under mp.spawn.
        os._exit(exit_code)


@pytest.mark.parametrize("shape", _AG_SHAPES)
@pytest.mark.parametrize("trans_weight", [False, True])
@bypass_not_implemented
def test_all_gather_quant_gemm_comm(master_port, shape, trans_weight):
    _skip_if_no_specific_backend(MojoAllGatherQuantGemm)
    seq_full, in_features, _ = shape
    if seq_full % WORLD_SIZE != 0:
        pytest.skip(f"seq_full={seq_full} not divisible by world_size={WORLD_SIZE}")
    if _PLATFORM == "ilu" and (seq_full // WORLD_SIZE) % 256 != 0:
        pytest.skip("Ixformer allgather_quant_matmul requires M_per_rank divisible by BM=256.")
    if _PLATFORM == "ilu" and in_features % 64 != 0:
        pytest.skip("Ixformer allgather_quant_matmul requires K divisible by BK=64.")

    mp.spawn(
        _worker_ag,
        args=(WORLD_SIZE, master_port, shape, trans_weight, False),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.parametrize(
    "shape",
    [(128, 1024, 2048), (4096, 3584, 4096), (4096, 3584, 8192)],
)
@bypass_not_implemented
def test_all_gather_quant_gemm_comm_non_contiguous(master_port, shape):
    _skip_if_no_specific_backend(MojoAllGatherQuantGemm)
    if _PLATFORM == "ilu":
        pytest.skip("Ixformer allgather_quant_matmul requires contiguous input.")
    seq_full, _, _ = shape
    if seq_full % WORLD_SIZE != 0:
        pytest.skip(f"seq_full={seq_full} not divisible by world_size={WORLD_SIZE}")

    mp.spawn(
        _worker_ag,
        args=(WORLD_SIZE, master_port, shape, False, True),
        nprocs=WORLD_SIZE,
        join=True,
    )


# ===========================================================================
# QuantGemmReduceScatter — multi-rank
# ===========================================================================

def _worker_rs(rank, world_size, port, shape, trans_weight, non_contig):
    _init_pg(rank, world_size, port)
    exit_code = 1
    try:
        seq_full, in_features, out_features = shape

        weight_shape = (
            (out_features, in_features) if trans_weight else (in_features, out_features)
        )
        weight_int8 = torch.randint(-128, 127, weight_shape, dtype=torch.int8)
        weight_scale = torch.rand(out_features, dtype=torch.bfloat16) + 0.5

        op, op_ref = _build_pair_rs(
            in_features, out_features, trans_weight, weight_int8, weight_scale,
        )

        if non_contig:
            big = torch.randint(-128, 127, (seq_full, in_features * 2), dtype=torch.int8)
            x_full = _to_dev(big)[:, ::2]
            assert not x_full.is_contiguous()
        else:
            x_full = _to_dev(
                torch.randint(-128, 127, (seq_full, in_features), dtype=torch.int8)
            )
        scale_full = _to_dev(torch.rand(seq_full, dtype=torch.float32) + 0.1)

        op.forward_diff_with(
            op_ref, x_full, scale_full,
            atol=1, rtol=2e-3,
            )
        exit_code = 0
    finally:
        try:
            if dist.is_initialized():
                _destroy_pg()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        # Avoid fragile NVSHMEM / symmetric memory Python destructors under mp.spawn.
        os._exit(exit_code)


@pytest.mark.parametrize("shape", _RS_SHAPES)
@pytest.mark.parametrize("trans_weight", [False, True])
@bypass_not_implemented
def test_quant_gemm_reduce_scatter_comm(master_port, shape, trans_weight):
    _skip_if_no_specific_backend(MojoQuantGemmReduceScatter)
    seq_full, _, _ = shape
    if seq_full % WORLD_SIZE != 0:
        pytest.skip(f"seq_full={seq_full} not divisible by world_size={WORLD_SIZE}")
    if _PLATFORM == "ilu" and (seq_full // WORLD_SIZE) % 256 != 0:
        pytest.skip("Ixformer quant_matmul_reducescatter requires M / world_size divisible by BM=256.")

    mp.spawn(
        _worker_rs,
        args=(WORLD_SIZE, master_port, shape, trans_weight, False),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.parametrize(
    "shape",
    [(128, 1024, 2048), (4096, 4096, 3584), (4096, 8192, 3584)],
)
@bypass_not_implemented
def test_quant_gemm_reduce_scatter_comm_non_contiguous(master_port, shape):
    _skip_if_no_specific_backend(MojoQuantGemmReduceScatter)
    if _PLATFORM == "ilu":
        pytest.skip("Ixformer quant_matmul_reducescatter requires contiguous input.")
    seq_full, _, _ = shape
    if seq_full % WORLD_SIZE != 0:
        pytest.skip(f"seq_full={seq_full} not divisible by world_size={WORLD_SIZE}")

    mp.spawn(
        _worker_rs,
        args=(WORLD_SIZE, master_port, shape, False, True),
        nprocs=WORLD_SIZE,
        join=True,
    )


# ===========================================================================
# Single-rank (no dist initialised) fallback paths.
# ===========================================================================

@pytest.mark.parametrize(
    "in_features, out_features",
    [(64, 128), (1024, 2048), (3584, 4096)],
)
@pytest.mark.parametrize("trans_weight", [False, True])
@bypass_not_implemented
def test_all_gather_quant_gemm_no_dist(in_features, out_features, trans_weight):
    _skip_if_no_specific_backend(MojoAllGatherQuantGemm)
    seq = 32
    weight_shape = (
        (out_features, in_features) if trans_weight else (in_features, out_features)
    )
    weight_int8 = torch.randint(-128, 127, weight_shape, dtype=torch.int8)
    weight_scale = torch.rand(out_features, dtype=torch.bfloat16) + 0.5

    op = MojoAllGatherQuantGemm(
        in_features=in_features, out_features=out_features,
        trans_weight=trans_weight, process_group=None, gather_dim=0,
    )
    op = op.to(DEVICE) if DEVICE != "cpu" else op
    op_ref = MojoAllGatherQuantGemm._registry.get("torch")(
        in_features=in_features, out_features=out_features,
        trans_weight=trans_weight, process_group=None, gather_dim=0,
    )
    op_ref = op_ref.to(DEVICE) if DEVICE != "cpu" else op_ref
    state = {
        "weight": _to_dev(weight_int8),
        "weight_scale": _to_dev(weight_scale),
    }
    op.load_state_dict(state)
    op_ref.load_state_dict(state)

    x = _to_dev(torch.randint(-128, 127, (seq, in_features), dtype=torch.int8))
    scale = _to_dev(torch.rand(seq, dtype=torch.float32) + 0.1)
    op.forward_diff_with(op_ref, x, scale, atol=1, rtol=2e-3)


@pytest.mark.parametrize(
    "in_features, out_features",
    [(64, 128), (1024, 2048), (3584, 4096)],
)
@pytest.mark.parametrize("trans_weight", [False, True])
@bypass_not_implemented
def test_quant_gemm_reduce_scatter_no_dist(in_features, out_features, trans_weight):
    _skip_if_no_specific_backend(MojoQuantGemmReduceScatter)
    seq = 32
    weight_shape = (
        (out_features, in_features) if trans_weight else (in_features, out_features)
    )
    weight_int8 = torch.randint(-128, 127, weight_shape, dtype=torch.int8)
    weight_scale = torch.rand(out_features, dtype=torch.bfloat16) + 0.5

    op = MojoQuantGemmReduceScatter(
        in_features=in_features, out_features=out_features,
        trans_weight=trans_weight, process_group=None, scatter_dim=0,
    )
    op = op.to(DEVICE) if DEVICE != "cpu" else op
    op_ref = MojoQuantGemmReduceScatter._registry.get("torch")(
        in_features=in_features, out_features=out_features,
        trans_weight=trans_weight, process_group=None, scatter_dim=0,
    )
    op_ref = op_ref.to(DEVICE) if DEVICE != "cpu" else op_ref
    state = {
        "weight": _to_dev(weight_int8),
        "weight_scale": _to_dev(weight_scale),
    }
    op.load_state_dict(state)
    op_ref.load_state_dict(state)

    x = _to_dev(torch.randint(-128, 127, (seq, in_features), dtype=torch.int8))
    scale = _to_dev(torch.rand(seq, dtype=torch.float32) + 0.1)
    op.forward_diff_with(op_ref, x, scale, atol=1, rtol=2e-3)
