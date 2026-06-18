"""Distributed correctness tests for MojoDistFusedConcatAttnGateQuant.

Mirrors mojo_opset/tests/accuracy/operators/test_compute_with_comm.py: each
worker constructs the op (current backend) and a torch-backend reference, and
``forward_diff_with`` compares the two. When MojoDistFusedConcatAttnGateQuant has no
device-specific backend registered, all cases skip via ``bypass_not_implemented``
(both operands resolve to the same auto-generated Torch class).

Cases cover M13 prefill + decode shapes, mixed full/SWA head counts, and a
non-contiguous attention output to exercise the strided-view path the M13
caller produces post-attention.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mojo_opset.experimental import MojoDistFusedConcatAttnGateQuant
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_dist_backend, get_platform, get_torch_device

torch.manual_seed(42)

dtypes = [torch.float16, torch.bfloat16]

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


# (T, N_full, N_swa, head_dim, hidden_size) covering M13 prefill seq=4k +
# decode-like small T.  All shapes are picked so T % WORLD_SIZE == 0 (or
# the op pads internally).
_M13_SHAPES = [
    (8, 2, 2, 16, 32),          # tiny smoke
    (32, 4, 4, 32, 128),        # small
    (128, 8, 8, 32, 256),       # medium
    (1, 16, 16, 128, 3584),     # decode batch=1, M13 layer-52 type
    (2, 16, 16, 128, 3584),     # decode batch=2, M13 layer-52
    (128, 16, 16, 128, 3584),   # small batch M13
    (1024, 16, 16, 128, 3584),  # half-prefill
    (4096, 16, 16, 128, 3584),  # M13 layer-52 prefill seq=4k, tp=2 local
    (4096, 8, 24, 128, 3584),   # mixed full/SWA proportions
    (4096, 4, 28, 128, 3584),   # very lopsided full vs swa
    (4096, 16, 16, 64, 3584),   # smaller head_dim variant
    (257, 4, 4, 32, 128),       # uneven T → exercises internal padding path
    (4097, 16, 16, 128, 3584),  # uneven prefill T → padding under M13 shape
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


def _skip_if_no_specific_backend():
    """Skip when MojoDistFusedConcatAttnGateQuant only has the auto-generated torch fallback."""
    default_cls = MojoDistFusedConcatAttnGateQuant._registry.get(None)
    torch_cls = MojoDistFusedConcatAttnGateQuant._registry.get("torch")
    if default_cls is torch_cls:
        raise NotImplementedError(
            "No device-specific backend for MojoDistFusedConcatAttnGateQuant; "
            "both operands resolve to the same implementation."
        )


def _skip_if_ixformer_unsupported_gate_shape(num_heads_total):
    if _PLATFORM == "ilu" and num_heads_total < 8:
        pytest.skip("Ixformer mixed_type_linear requires at least 8 gate outputs.")


def _build_inputs(T, N_full, N_swa, head_dim, hidden_size, dtype):
    hidden_states = torch.randn(T, hidden_size, dtype=dtype)
    full_attn = torch.randn(T, N_full, head_dim, dtype=dtype)
    swa_attn = torch.randn(T, N_swa, head_dim, dtype=dtype)
    inv_smooth = torch.rand((N_full + N_swa) * head_dim, dtype=torch.float32) + 0.1
    return hidden_states, full_attn, swa_attn, inv_smooth


def _load_op_params(op, inv_smooth, gate_weight_full, gate_weight_swa, gate_bias_full, gate_bias_swa):
    state_dict = op.state_dict()
    state_dict["inv_smooth_scale"] = inv_smooth
    state_dict["attn_gate.full_gate_weight"] = gate_weight_full
    state_dict["attn_gate.swa_gate_weight"] = gate_weight_swa
    if gate_bias_full is not None:
        state_dict["attn_gate.full_gate_bias"] = gate_bias_full
    if gate_bias_swa is not None:
        state_dict["attn_gate.swa_gate_bias"] = gate_bias_swa
    op.load_state_dict(state_dict)


# ===========================================================================
# Single-rank correctness — fallback path (tp_group=None) on device.
# ===========================================================================

@pytest.mark.parametrize(
    "T, N_full, N_swa, head_dim, hidden_size",
    [
        (8, 2, 2, 16, 32),
        (32, 4, 4, 32, 128),
        (128, 8, 8, 32, 256),
        (1, 16, 16, 128, 3584),
        (1024, 16, 16, 128, 3584),
        (4096, 16, 16, 128, 3584),
        (4096, 8, 24, 128, 3584),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dist_fused_attn_gate_quant_no_dist(T, N_full, N_swa, head_dim, hidden_size, dtype):
    _skip_if_no_specific_backend()
    _skip_if_ixformer_unsupported_gate_shape(N_full + N_swa)
    hidden_states, full_attn, swa_attn, inv_smooth = _build_inputs(
        T, N_full, N_swa, head_dim, hidden_size, dtype
    )
    hidden_states = _to_dev(hidden_states)
    full_attn = _to_dev(full_attn)
    swa_attn = _to_dev(swa_attn)
    gw_full = torch.randn(N_full, hidden_size, dtype=dtype)
    gw_swa = torch.randn(N_swa, hidden_size, dtype=dtype)

    op = MojoDistFusedConcatAttnGateQuant(
        hidden_size=hidden_size,
        num_heads_full=N_full,
        num_heads_swa=N_swa,
        head_dim=head_dim,
        bias=False,
        tp_group=None,
    ).to(DEVICE)
    _load_op_params(op, inv_smooth, gw_full, gw_swa, None, None)

    op_ref = MojoDistFusedConcatAttnGateQuant._registry.get("torch")(
        hidden_size=hidden_size,
        num_heads_full=N_full,
        num_heads_swa=N_swa,
        head_dim=head_dim,
        bias=False,
        tp_group=None,
    ).to(DEVICE)
    _load_op_params(op_ref, inv_smooth, gw_full, gw_swa, None, None)

    op.forward_diff_with(
        op_ref, hidden_states, full_attn, swa_attn,
        atol=(1, 2e-3), rtol=(0, 2e-3),
    )


@pytest.mark.parametrize(
    "T, N_full, N_swa, head_dim, hidden_size",
    [
        (32, 4, 4, 32, 128),
        (128, 8, 8, 32, 256),
        (1024, 16, 16, 128, 3584),
        (4096, 16, 16, 128, 3584),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dist_fused_attn_gate_quant_no_dist_non_contiguous(
    T, N_full, N_swa, head_dim, hidden_size, dtype
):
    """M13 callers feed strided slices for full / SWA attention output."""
    _skip_if_no_specific_backend()
    _skip_if_ixformer_unsupported_gate_shape(N_full + N_swa)
    hidden_states, _, _, inv_smooth = _build_inputs(T, N_full, N_swa, head_dim, hidden_size, dtype)
    hidden_states = _to_dev(hidden_states)
    big_full = torch.randn(T, N_full * 2, head_dim, dtype=dtype)
    big_swa = torch.randn(T, N_swa * 2, head_dim, dtype=dtype)
    full_attn = big_full[:, ::2, :]
    swa_attn = big_swa[:, ::2, :]
    assert not full_attn.is_contiguous() and not swa_attn.is_contiguous()
    full_attn = _to_dev(full_attn)
    swa_attn = _to_dev(swa_attn)

    gw_full = torch.randn(N_full, hidden_size, dtype=dtype)
    gw_swa = torch.randn(N_swa, hidden_size, dtype=dtype)

    op = MojoDistFusedConcatAttnGateQuant(
        hidden_size=hidden_size, num_heads_full=N_full, num_heads_swa=N_swa,
        head_dim=head_dim, bias=False, tp_group=None,
    ).to(DEVICE)
    _load_op_params(op, inv_smooth, gw_full, gw_swa, None, None)

    op_ref = MojoDistFusedConcatAttnGateQuant._registry.get("torch")(
        hidden_size=hidden_size, num_heads_full=N_full, num_heads_swa=N_swa,
        head_dim=head_dim, bias=False, tp_group=None,
    ).to(DEVICE)
    _load_op_params(op_ref, inv_smooth, gw_full, gw_swa, None, None)

    op.forward_diff_with(
        op_ref, hidden_states, full_attn, swa_attn,
        atol=(1, 2e-3), rtol=(0, 2e-3),
    )


# ===========================================================================
# Multi-rank distributed test — exercises the unified-scale AllGather path.
# ===========================================================================

def _worker_dist_gate_quant(
    rank, world_size, port, T, N_full, N_swa, head_dim, hidden_size, dtype,
    full_attn_global, swa_attn_global, hidden_states, inv_smooth_global,
    gw_full, gw_swa,
):
    _init_pg(rank, world_size, port)
    try:
        n_full_local = N_full // world_size
        n_swa_local = N_swa // world_size

        # Each rank holds its head slice
        full_local = _to_dev(
            full_attn_global[:, rank * n_full_local : (rank + 1) * n_full_local, :].contiguous()
        )
        swa_local = _to_dev(
            swa_attn_global[:, rank * n_swa_local : (rank + 1) * n_swa_local, :].contiguous()
        )
        hidden = _to_dev(hidden_states)
        inv_smooth_local_full = inv_smooth_global[: N_full * head_dim].view(N_full, head_dim)[
            rank * n_full_local : (rank + 1) * n_full_local
        ].reshape(-1)
        inv_smooth_local_swa = inv_smooth_global[N_full * head_dim :].view(N_swa, head_dim)[
            rank * n_swa_local : (rank + 1) * n_swa_local
        ].reshape(-1)
        inv_smooth_local = _to_dev(
            torch.cat([inv_smooth_local_full, inv_smooth_local_swa], dim=0).contiguous()
        )
        gw_full_local = _to_dev(gw_full[rank * n_full_local : (rank + 1) * n_full_local].contiguous())
        gw_swa_local = _to_dev(gw_swa[rank * n_swa_local : (rank + 1) * n_swa_local].contiguous())

        op = MojoDistFusedConcatAttnGateQuant(
            hidden_size=hidden_size,
            num_heads_full=n_full_local,
            num_heads_swa=n_swa_local,
            head_dim=head_dim,
            bias=False,
            tp_group=dist.group.WORLD,
        )
        op = op.to(DEVICE) if DEVICE != "cpu" else op
        _load_op_params(op, inv_smooth_local, gw_full_local, gw_swa_local, None, None)

        op_ref = MojoDistFusedConcatAttnGateQuant._registry.get("torch")(
            hidden_size=hidden_size,
            num_heads_full=n_full_local,
            num_heads_swa=n_swa_local,
            head_dim=head_dim,
            bias=False,
            tp_group=dist.group.WORLD,
        )
        op_ref = op_ref.to(DEVICE) if DEVICE != "cpu" else op_ref
        _load_op_params(op_ref, inv_smooth_local, gw_full_local, gw_swa_local, None, None)

        op.forward_diff_with(
            op_ref, hidden, full_local, swa_local,
            atol=(1, 2e-3), rtol=(0, 2e-3),
        )
    finally:
        _destroy_pg()


@pytest.mark.parametrize("shape", _M13_SHAPES)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dist_fused_attn_gate_quant_comm(master_port, shape, dtype):
    _skip_if_no_specific_backend()
    T, N_full, N_swa, head_dim, hidden_size = shape
    if N_full % WORLD_SIZE != 0 or N_swa % WORLD_SIZE != 0:
        pytest.skip(f"head counts not divisible by world_size={WORLD_SIZE}")
    _skip_if_ixformer_unsupported_gate_shape((N_full + N_swa) // WORLD_SIZE)
    full_attn = torch.randn(T, N_full, head_dim, dtype=dtype)
    swa_attn = torch.randn(T, N_swa, head_dim, dtype=dtype)
    hidden_states = torch.randn(T, hidden_size, dtype=dtype)
    inv_smooth = torch.rand((N_full + N_swa) * head_dim, dtype=torch.float32) + 0.1
    gw_full = torch.randn(N_full, hidden_size, dtype=dtype)
    gw_swa = torch.randn(N_swa, hidden_size, dtype=dtype)

    mp.spawn(
        _worker_dist_gate_quant,
        args=(
            WORLD_SIZE, master_port, T, N_full, N_swa, head_dim, hidden_size, dtype,
            full_attn, swa_attn, hidden_states, inv_smooth, gw_full, gw_swa,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
