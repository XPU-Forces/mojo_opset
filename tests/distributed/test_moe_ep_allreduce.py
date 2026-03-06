import os

import pytest
import torch

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from mojo_opset.core.operators.moe import MojoMoE
from mojo_opset.distributed.parallel import MojoMoEExpertParallel


def _get_world_size():
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size <= 0:
        pytest.skip("This test requires launching with torchrun (WORLD_SIZE must be set).")
    return world_size


def _get_device_type():
    forced = os.environ.get("MOJO_OPSET_TEST_DEVICE_TYPE", "").strip()
    if forced:
        return forced
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def test_moe_ep_allreduce():
    world_size = _get_world_size()
    device_type = _get_device_type()
    device_mesh = init_device_mesh(device_type, (world_size,))

    hidden_size = 512
    intermediate_size = 128
    num_experts = 16 * world_size
    top_k = 2

    torch.manual_seed(0)
    ref = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device_type)

    torch.manual_seed(0)
    moe = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device_type)
    moe.load_state_dict(ref.state_dict())

    moe = parallelize_module(
        moe,
        device_mesh=device_mesh,
        parallelize_plan=MojoMoEExpertParallel(),
    )

    x = torch.randn(32, hidden_size, device=device_type)
    out_parallel = moe(x)
    out_ref = ref(x)
    assert torch.allclose(out_parallel, out_ref, rtol=1e-4, atol=1e-4)
