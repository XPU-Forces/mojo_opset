import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
import torch.distributed as dist
import torch.nn as nn

from tests.dist_common import dist_test
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.placement_types import Partial
from torch.distributed.tensor.placement_types import Replicate
from torch.distributed.tensor.placement_types import Shard

from mojo_opset import MojoSwiGLUMLP
from mojo_opset.distributed.parallel import MojoColwiseParallel
from mojo_opset.distributed.parallel import MojoQKVColwiseParallel
from mojo_opset.distributed.parallel import MojoRowwiseParallel
from mojo_opset.distributed.parallel import MojoSwiGLUParallel
from mojo_opset.distributed.parallel import mojo_parallelize_module

TP_SIZE = 2
TP_SIZE = 2
DP_SIZE = 4


def verify_dp_consistency(dist, output_data, mesh, test_name):
    """Each TP group gathers within itself via mesh["tp"] and verifies in parallel."""
    tp_rank = mesh.get_local_rank("tp")
    dp_rank = mesh.get_local_rank("dp")
    tp_mesh = mesh["tp"]
    tp_group = tp_mesh.get_group()
    tp_size = tp_mesh.size()

    if tp_rank == 0:
        gathered = [torch.zeros_like(output_data) for _ in range(tp_size)]
        dist.gather(output_data, gathered, group=tp_group, group_dst=0)
        match = all(torch.allclose(gathered[0], gathered[i]) for i in range(1, tp_size))
        if not match:
            for i, o in enumerate(gathered):
                print(f"  tp_rank={i}: {o}")
        assert match, f"{test_name}: dp_rank={dp_rank} output mismatch across tp ranks!"
    else:
        dist.gather(output_data, group=tp_group, group_dst=0)


@dist_test(world_size=TP_SIZE * DP_SIZE, backend="gloo")
def test_basic():
    mesh = init_device_mesh(device_type="cpu", mesh_shape=(TP_SIZE, DP_SIZE), mesh_dim_names=["tp", "dp"])

    x = MojoRowwiseParallel(input_layouts=(Shard(1),), output_layouts=(Replicate(),))(
        torch.nn.Linear(128, 128, bias=False), mesh["tp"]
    ).to("cpu")

    torch.manual_seed(mesh.get_local_rank("dp") * 42)
    inputs = torch.randn(1, 64, device="cpu")
    output = x(inputs)
    verify_dp_consistency(dist, output, mesh, "test_basic")


@dist_test(world_size=TP_SIZE * DP_SIZE, backend="gloo")
def test_multi_layer():
    mesh = init_device_mesh(device_type="cpu", mesh_shape=(TP_SIZE, DP_SIZE), mesh_dim_names=["tp", "dp"])

    torch.manual_seed(42)
    x = torch.nn.Sequential(
        MojoRowwiseParallel(
            input_layouts=(Shard(1),),
            output_layouts=(Partial(),),
            use_local_output=False,
        )(torch.nn.Linear(128, 128, bias=False), mesh["tp"]),
        MojoColwiseParallel(
            output_layouts=[Shard(1)],
            use_local_output=False,
        )(torch.nn.Linear(128, 128, bias=False), mesh["tp"]),
        MojoRowwiseParallel(
            input_layouts=(Shard(1),),
            output_layouts=(Partial(),),
            use_local_output=False,
        )(torch.nn.Linear(128, 128, bias=False), mesh["tp"]),
        MojoColwiseParallel(
            output_layouts=(Replicate(),),
            use_local_output=True,
        )(torch.nn.Linear(128, 128, bias=False), mesh["tp"]),
    ).to("cpu")

    torch.manual_seed(mesh.get_local_rank("dp") * 42)
    inputs = torch.randn(1, 64, device="cpu")
    output = x(inputs)
    verify_dp_consistency(dist, output, mesh, "test_multi_layer")


# ──────────────────────────────────────────────────
# SwiGLU parallel tests
# ──────────────────────────────────────────────────


@dist_test(world_size=TP_SIZE, backend="gloo")
def test_swiglu_parallel_basic():
    """Apply MojoSwiGLUParallel to MojoSwiGLUMLP and verify output matches the
    non-parallelized reference."""
    mesh = init_device_mesh("cpu", (TP_SIZE,))
    input_size, output_size, hidden_size = 128, 128, 64

    torch.manual_seed(42)
    ref = MojoSwiGLUMLP(input_size, output_size, hidden_size).to("cpu")

    torch.manual_seed(42)
    par = MojoSwiGLUMLP(input_size, output_size, hidden_size).to("cpu")
    par = mojo_parallelize_module(par, mesh, MojoSwiGLUParallel())

    x = torch.randn(4, input_size)
    out_ref = ref(x)
    out_par = par(x)

    assert torch.allclose(out_ref, out_par, rtol=1e-4, atol=1e-4), (
        f"SwiGLU output mismatch on rank {dist.get_rank()}: max diff = {(out_ref - out_par).abs().max().item()}"
    )


@dist_test(world_size=TP_SIZE, backend="gloo")
def test_swiglu_parallel_weight_sharding():
    """Verify that fc1 and fc2 weights are correctly sharded after
    parallelization."""
    mesh = init_device_mesh("cpu", (TP_SIZE,))
    input_size, output_size, hidden_size = 128, 128, 64

    torch.manual_seed(42)
    par = MojoSwiGLUMLP(input_size, output_size, hidden_size).to("cpu")
    par = mojo_parallelize_module(par, mesh, MojoSwiGLUParallel())

    mod = par._mod
    fc1_w = mod.fc1.weight
    fc2_w = mod.fc2.weight

    assert fc1_w.shape == (hidden_size * 2 // TP_SIZE, input_size), f"fc1 weight shape mismatch: {fc1_w.shape}"
    assert fc2_w.shape == (output_size, hidden_size // TP_SIZE), f"fc2 weight shape mismatch: {fc2_w.shape}"


@dist_test(world_size=4, backend="gloo")
def test_swiglu_parallel_tp4():
    """SwiGLU parallel with TP=4."""
    tp = 4
    mesh = init_device_mesh("cpu", (tp,))
    input_size, output_size, hidden_size = 256, 256, 128

    torch.manual_seed(42)
    ref = MojoSwiGLUMLP(input_size, output_size, hidden_size).to("cpu")

    torch.manual_seed(42)
    par = MojoSwiGLUMLP(input_size, output_size, hidden_size).to("cpu")
    par = mojo_parallelize_module(par, mesh, MojoSwiGLUParallel())

    x = torch.randn(8, input_size)
    out_ref = ref(x)
    out_par = par(x)

    assert torch.allclose(out_ref, out_par, rtol=1e-4, atol=1e-4), (
        f"SwiGLU TP=4 output mismatch on rank {dist.get_rank()}: max diff = {(out_ref - out_par).abs().max().item()}"
    )


# ──────────────────────────────────────────────────
# QKV ColwiseParallel tests
# ──────────────────────────────────────────────────


@dist_test(world_size=TP_SIZE, backend="gloo")
def test_qkv_parallel_basic():
    """Apply MojoQKVColwiseParallel and verify per-rank output matches the
    corresponding slice of the reference full output."""
    mesh = init_device_mesh("cpu", (TP_SIZE,))
    num_q_heads, num_kv_heads, head_dim = 4, 2, 32
    hidden_size = 128
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_out_dim = q_dim + 2 * kv_dim

    torch.manual_seed(42)
    ref_qkv = nn.Linear(hidden_size, qkv_out_dim, bias=False).to("cpu")

    torch.manual_seed(42)
    par_qkv = nn.Linear(hidden_size, qkv_out_dim, bias=False).to("cpu")
    par_qkv = mojo_parallelize_module(
        par_qkv,
        mesh,
        MojoQKVColwiseParallel(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ),
    )

    x = torch.randn(4, hidden_size)
    out_ref = ref_qkv(x)
    out_par = par_qkv(x)

    rank = dist.get_rank()
    tp = TP_SIZE
    q_per_rank = num_q_heads // tp
    q_start = rank * q_per_rank * head_dim
    q_end = q_start + q_per_rank * head_dim
    ref_q = out_ref[:, q_start:q_end]

    replicate = max(1, tp // num_kv_heads)
    kv_idx = rank // replicate
    k_offset = q_dim
    ref_k = out_ref[:, k_offset + kv_idx * head_dim : k_offset + (kv_idx + 1) * head_dim]
    v_offset = q_dim + kv_dim
    ref_v = out_ref[:, v_offset + kv_idx * head_dim : v_offset + (kv_idx + 1) * head_dim]

    ref_local = torch.cat([ref_q, ref_k, ref_v], dim=-1)
    assert out_par.shape == ref_local.shape, (
        f"QKV shape mismatch on rank {rank}: got {out_par.shape}, expected {ref_local.shape}"
    )
    assert torch.allclose(ref_local, out_par, rtol=1e-4, atol=1e-4), (
        f"QKV output mismatch on rank {rank}: max diff = {(ref_local - out_par).abs().max().item()}"
    )


@dist_test(world_size=TP_SIZE, backend="gloo")
def test_qkv_parallel_with_bias():
    """QKV parallel with bias enabled."""
    mesh = init_device_mesh("cpu", (TP_SIZE,))
    num_q_heads, num_kv_heads, head_dim = 4, 2, 16
    hidden_size = 64
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_out_dim = q_dim + 2 * kv_dim

    torch.manual_seed(42)
    ref_qkv = nn.Linear(hidden_size, qkv_out_dim, bias=True).to("cpu")

    torch.manual_seed(42)
    par_qkv = nn.Linear(hidden_size, qkv_out_dim, bias=True).to("cpu")
    par_qkv = mojo_parallelize_module(
        par_qkv,
        mesh,
        MojoQKVColwiseParallel(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ),
    )

    x = torch.randn(2, hidden_size)
    out_ref = ref_qkv(x)
    out_par = par_qkv(x)

    rank = dist.get_rank()
    q_per_rank = num_q_heads // TP_SIZE
    q_start = rank * q_per_rank * head_dim
    q_end = q_start + q_per_rank * head_dim
    ref_q = out_ref[:, q_start:q_end]

    replicate = max(1, TP_SIZE // num_kv_heads)
    kv_idx = rank // replicate
    k_offset = q_dim
    ref_k = out_ref[:, k_offset + kv_idx * head_dim : k_offset + (kv_idx + 1) * head_dim]
    v_offset = q_dim + kv_dim
    ref_v = out_ref[:, v_offset + kv_idx * head_dim : v_offset + (kv_idx + 1) * head_dim]

    ref_local = torch.cat([ref_q, ref_k, ref_v], dim=-1)
    assert torch.allclose(ref_local, out_par, rtol=1e-4, atol=1e-4), (
        f"QKV with bias mismatch on rank {rank}: max diff = {(ref_local - out_par).abs().max().item()}"
    )


@dist_test(world_size=4, backend="gloo")
def test_qkv_parallel_kv_replicate():
    """When TP_SIZE > num_kv_heads, KV heads should be replicated across
    multiple ranks."""
    tp = 4
    mesh = init_device_mesh("cpu", (tp,))
    num_q_heads, num_kv_heads, head_dim = 8, 2, 32
    hidden_size = 256
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_out_dim = q_dim + 2 * kv_dim

    torch.manual_seed(42)
    ref_qkv = nn.Linear(hidden_size, qkv_out_dim, bias=False).to("cpu")

    torch.manual_seed(42)
    par_qkv = nn.Linear(hidden_size, qkv_out_dim, bias=False).to("cpu")
    par_qkv = mojo_parallelize_module(
        par_qkv,
        mesh,
        MojoQKVColwiseParallel(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ),
    )

    x = torch.randn(4, hidden_size)
    out_ref = ref_qkv(x)
    out_par = par_qkv(x)

    rank = dist.get_rank()
    q_per_rank = num_q_heads // tp
    q_start = rank * q_per_rank * head_dim
    q_end = q_start + q_per_rank * head_dim
    ref_q = out_ref[:, q_start:q_end]

    replicate = max(1, tp // num_kv_heads)
    kv_idx = rank // replicate
    k_offset = q_dim
    ref_k = out_ref[:, k_offset + kv_idx * head_dim : k_offset + (kv_idx + 1) * head_dim]
    v_offset = q_dim + kv_dim
    ref_v = out_ref[:, v_offset + kv_idx * head_dim : v_offset + (kv_idx + 1) * head_dim]

    ref_local = torch.cat([ref_q, ref_k, ref_v], dim=-1)
    assert torch.allclose(ref_local, out_par, rtol=1e-4, atol=1e-4), (
        f"QKV KV-replicate mismatch on rank {rank}: max diff = {(ref_local - out_par).abs().max().item()}"
    )
