import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from mojo_opset.distributed.parallel import MojoRowwiseParallel, MojoColwiseParallel
from torch.distributed.tensor.placement_types import Shard, Replicate, Partial
from tests.dist_common import dist_test

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
        match = all(
            torch.allclose(gathered[0], gathered[i]) for i in range(1, tp_size)
        )
        if not match:
            for i, o in enumerate(gathered):
                print(f"  tp_rank={i}: {o}")
        assert match, f"{test_name}: dp_rank={dp_rank} output mismatch across tp ranks!"
    else:
        dist.gather(output_data, group=tp_group, group_dst=0)


@dist_test(world_size=TP_SIZE * DP_SIZE, backend="gloo")
def test_basic():
    mesh = init_device_mesh(
        device_type="cpu", mesh_shape=(TP_SIZE, DP_SIZE), mesh_dim_names=["tp", "dp"]
    )

    x = MojoRowwiseParallel(input_layouts=(Shard(1),), output_layouts=(Replicate(),))(
        torch.nn.Linear(128, 128, bias=False), mesh["tp"]
    ).to('cpu')

    torch.manual_seed(mesh.get_local_rank("dp") * 42)
    inputs = torch.randn(1, 64, device='cpu')
    output = x(inputs)
    verify_dp_consistency(dist, output, mesh, "test_basic")


@dist_test(world_size=TP_SIZE * DP_SIZE, backend="gloo")
def test_multi_layer():
    mesh = init_device_mesh(
        device_type="cpu", mesh_shape=(TP_SIZE, DP_SIZE), mesh_dim_names=["tp", "dp"]
    )

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
    ).to('cpu')

    torch.manual_seed(mesh.get_local_rank("dp") * 42)
    inputs = torch.randn(1, 64, device='cpu')
    output = x(inputs)
    verify_dp_consistency(dist, output, mesh, "test_multi_layer")


if __name__ == '__main__':
    test_multi_layer()
