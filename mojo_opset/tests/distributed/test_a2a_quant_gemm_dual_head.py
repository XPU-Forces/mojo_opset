"""Distributed correctness tests for MojoA2AQuantGemmDualHead.

Mirrors mojo_opset/tests/accuracy/operators/test_compute_with_comm.py. Each
worker constructs the op (current backend) and a torch-backend reference, then
``forward_diff_with`` compares the two. When MojoA2AQuantGemmDualHead has no
device-specific backend registered, all cases skip via ``bypass_not_implemented``
(both operands resolve to the same auto-generated Torch class).

Cases cover:
  * identity head split — rank R holds heads ``[R*per_rank, (R+1)*per_rank)``
  * GQA-strided head split — rank R holds ``(R, R+tp, R+2*tp, ...)`` mimicking
    yoco's KV_SHUFFLE layout. argsort permute path is exercised.
  * M13.4 12B prefill seq=4k shapes (layer-52 type and layer-66 mixed type)
  * non-contiguous attn_int8 input — sliced strided view
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mojo_opset.experimental import MojoA2AQuantGemmDualHead
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_dist_backend, get_platform

torch.manual_seed(42)

dtypes = [torch.bfloat16]  # the op output dtype is bf16; only one to vary

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


# (n_pad, full_nh_global, swa_nh_global, head_dim, hidden_size).
# n_pad must be divisible by WORLD_SIZE (op assumes pre-padded by caller).
# *_nh_global must be divisible by WORLD_SIZE.
_DUAL_HEAD_SHAPES = [
    (8, 4, 4, 8, 16),               # tiny smoke
    (32, 4, 4, 16, 64),             # small
    (128, 8, 8, 32, 256),           # medium
    (4096, 32, 32, 128, 3584),      # M13.4 12B layer-52 prefill seq=4k
    (4096, 16, 48, 128, 3584),      # M13 layer-66 mixed full/SWA
    (1024, 16, 48, 128, 3584),      # smaller M13 batch
    (2, 32, 32, 128, 3584),         # decode-ish, tp=2 local
    (4096, 4, 28, 128, 3584),       # very lopsided full vs swa
]


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
    return t.to(DEVICE) if DEVICE != "cpu" else t


@pytest.fixture()
def master_port():
    return _free_port()


def _skip_if_no_specific_backend():
    """Skip when MojoA2AQuantGemmDualHead only has the auto-generated torch fallback."""
    default_cls = MojoA2AQuantGemmDualHead._registry.get(None)
    torch_cls = MojoA2AQuantGemmDualHead._registry.get("torch")
    if default_cls is torch_cls:
        raise NotImplementedError(
            "No device-specific backend for MojoA2AQuantGemmDualHead; "
            "both operands resolve to the same implementation."
        )


def _identity_head_indices(num_heads_global, tp_size, tp_rank):
    """Continuous block: rank R -> [R*per, (R+1)*per)."""
    per = num_heads_global // tp_size
    return tuple(range(tp_rank * per, (tp_rank + 1) * per))


def _strided_head_indices(num_heads_global, tp_size, tp_rank):
    """Strided: rank R -> (R, R+tp, R+2*tp, ...). Mimics yoco's KV_SHUFFLE-derived
    query head distribution at q_per_kv=1 (the simplest non-trivial perm)."""
    return tuple(range(tp_rank, num_heads_global, tp_size))


def _build_op_pair(
    *,
    tp_size, tp_rank,
    full_nh_global, swa_nh_global,
    head_dim, hidden_size,
    full_indices, swa_indices,
):
    """Construct (current-backend, torch-backend) pair sharing same o_proj weights."""
    op = MojoA2AQuantGemmDualHead(
        tp_size=tp_size, tp_rank=tp_rank,
        full_nh_global=full_nh_global, swa_nh_global=swa_nh_global,
        head_dim=head_dim, hidden_size=hidden_size,
        full_query_global_head_indices=full_indices,
        swa_query_global_head_indices=swa_indices,
        tp_group=dist.group.WORLD,
    )
    op = op.to(DEVICE) if DEVICE != "cpu" else op

    op_ref = MojoA2AQuantGemmDualHead._registry.get("torch")(
        tp_size=tp_size, tp_rank=tp_rank,
        full_nh_global=full_nh_global, swa_nh_global=swa_nh_global,
        head_dim=head_dim, hidden_size=hidden_size,
        full_query_global_head_indices=full_indices,
        swa_query_global_head_indices=swa_indices,
        tp_group=dist.group.WORLD,
    )
    op_ref = op_ref.to(DEVICE) if DEVICE != "cpu" else op_ref

    # Identical o_proj weights on both
    weight = torch.randint(-128, 127, op.o_proj.weight.shape, dtype=torch.int8)
    weight_scale = torch.rand(hidden_size, dtype=torch.bfloat16) + 0.5
    op.o_proj.weight.copy_(_to_dev(weight))
    op.o_proj.weight_scale.copy_(_to_dev(weight_scale))
    op_ref.o_proj.weight.copy_(_to_dev(weight))
    op_ref.o_proj.weight_scale.copy_(_to_dev(weight_scale))
    return op, op_ref


def _worker_dual_head(
    rank, world_size, port, shape, head_layout, non_contig,
):
    _init_pg(rank, world_size, port)
    try:
        n_pad, full_nh, swa_nh, head_dim, hidden_size = shape
        full_local = full_nh // world_size
        swa_local = swa_nh // world_size
        ch_local = (full_local + swa_local) * head_dim

        if head_layout == "identity":
            full_idx = _identity_head_indices(full_nh, world_size, rank)
            swa_idx = _identity_head_indices(swa_nh, world_size, rank)
        elif head_layout == "gqa_strided":
            full_idx = _strided_head_indices(full_nh, world_size, rank)
            swa_idx = _strided_head_indices(swa_nh, world_size, rank)
        else:
            raise ValueError(f"unknown head_layout: {head_layout}")

        op, op_ref = _build_op_pair(
            tp_size=world_size, tp_rank=rank,
            full_nh_global=full_nh, swa_nh_global=swa_nh,
            head_dim=head_dim, hidden_size=hidden_size,
            full_indices=full_idx, swa_indices=swa_idx,
        )

        if non_contig:
            big = torch.randint(-128, 127, (n_pad, ch_local * 2), dtype=torch.int8)
            attn_int8 = _to_dev(big)[:, ::2]
            assert not attn_int8.is_contiguous()
        else:
            attn_int8 = _to_dev(
                torch.randint(-128, 127, (n_pad, ch_local), dtype=torch.int8)
            )
        unified_scale = _to_dev(torch.rand(n_pad, dtype=torch.float32) + 0.1)

        op.forward_diff_with(
            op_ref, attn_int8, unified_scale,
            atol=(1, 2e-3), rtol=(0, 2e-3),
        )
    finally:
        _destroy_pg()


@pytest.mark.parametrize("shape", _DUAL_HEAD_SHAPES)
@pytest.mark.parametrize("head_layout", ["identity", "gqa_strided"])
@bypass_not_implemented
def test_a2a_quant_gemm_dual_head_comm(master_port, shape, head_layout):
    _skip_if_no_specific_backend()
    n_pad, full_nh, swa_nh, head_dim, hidden_size = shape
    if (
        n_pad % WORLD_SIZE != 0
        or full_nh % WORLD_SIZE != 0
        or swa_nh % WORLD_SIZE != 0
    ):
        pytest.skip(f"shape not divisible by world_size={WORLD_SIZE}")

    mp.spawn(
        _worker_dual_head,
        args=(WORLD_SIZE, master_port, shape, head_layout, False),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (128, 8, 8, 32, 256),
        (4096, 32, 32, 128, 3584),
        (1024, 16, 48, 128, 3584),
    ],
)
@pytest.mark.parametrize("head_layout", ["identity", "gqa_strided"])
@bypass_not_implemented
def test_a2a_quant_gemm_dual_head_comm_non_contiguous(
    master_port, shape, head_layout
):
    """M13 callers feed sliced views post-attention; ensure non-contig works."""
    _skip_if_no_specific_backend()
    n_pad, full_nh, swa_nh, head_dim, hidden_size = shape
    if (
        n_pad % WORLD_SIZE != 0
        or full_nh % WORLD_SIZE != 0
        or swa_nh % WORLD_SIZE != 0
    ):
        pytest.skip(f"shape not divisible by world_size={WORLD_SIZE}")

    mp.spawn(
        _worker_dual_head,
        args=(WORLD_SIZE, master_port, shape, head_layout, True),
        nprocs=WORLD_SIZE,
        join=True,
    )
