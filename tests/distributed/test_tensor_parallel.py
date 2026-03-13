import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from mojo_opset.distributed.parallel import MojoRowwiseParallel, MojoColwiseParallel
from torch.distributed.tensor.placement_types import Shard, Replicate, Partial

def test_basic():
    dist.init_process_group(backend="gloo")
    mesh = init_device_mesh(device_type="cpu", mesh_shape=(2, 4), mesh_dim_names=["tp", "dp"])
    x = MojoRowwiseParallel(input_layouts=(Shard(1),), output_layouts=(Replicate(),))(
        torch.nn.Linear(128, 128, bias=False), mesh["tp"]
    )
    print(x.state_dict().keys())
    inputs = torch.ones(1, 64)
    output = x(inputs)
    if mesh.get_local_rank('dp') == 0:
        print(output)
        print(type(output))
        print(output.shape)

def test_multi_layer():
    dist.init_process_group(backend="gloo")
    mesh = init_device_mesh(device_type="cpu", mesh_shape=(2, 4), mesh_dim_names=["tp", "dp"])
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
    )
    print(x.state_dict().keys())
    inputs = torch.ones(1, 64)
    output = x(inputs)
    if mesh.get_local_rank('dp') == 0:
        print(output)
        print(output.shape)

if __name__ == '__main__':
    test_multi_layer()
