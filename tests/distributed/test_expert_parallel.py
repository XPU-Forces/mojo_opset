import os

import pytest
import torch

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from mojo_opset.core.operators.moe import MojoMoE
from mojo_opset.distributed.parallel import MojoExpertParallel
from mojo_opset.utils.platform import get_platform


def _get_world_size():
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size <= 0:
        pytest.skip("This test requires launching with torchrun (WORLD_SIZE must be set).")
    return world_size


def _set_current_device(device_type: str, world_size: int) -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "npu":
        n_devices = torch.npu.device_count()
        if n_devices < world_size:
            pytest.skip(
                f"Not enough NPU devices: need {world_size}, but only {n_devices} visible. "
                f"Set ASCEND_RT_VISIBLE_DEVICES to expose more devices "
                f"(e.g. ASCEND_RT_VISIBLE_DEVICES=0,1)."
            )
        torch.npu.set_device(local_rank)
        return f"npu:{local_rank}"
    if device_type == "mlu":
        n_devices = torch.mlu.device_count()
        if n_devices < world_size:
            pytest.skip(
                f"Not enough MLU devices: need {world_size}, but only {n_devices} visible."
            )
        torch.mlu.set_device(local_rank)
        return f"mlu:{local_rank}"
    raise ValueError(f"Unsupported device type for distributed test: {device_type}")


def _init_weights(module):
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.02)


def _fx_graph_contains(graph_texts, needle: str) -> bool:
    return any(needle in g for g in graph_texts)

def _compile_capture_fx_and_run(module, example_inputs):
    graph_texts = []

    def _backend(gm: torch.fx.GraphModule, inputs):
        graph_texts.append(str(gm.graph))
        return gm.forward

    compiled = torch.compile(module, backend=_backend)
    out = compiled(*example_inputs)
    return out, graph_texts


def test_moe_ep_allreduce():
    world_size = _get_world_size()
    device_type = get_platform()
    device = _set_current_device(device_type, world_size)
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
    ).to(device)
    _init_weights(ref)

    moe = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device)
    moe.load_state_dict(ref.state_dict())

    moe = parallelize_module(
        moe,
        device_mesh=device_mesh,
        parallelize_plan=MojoExpertParallel(),
    )

    x = torch.randn(32, hidden_size, device=device)
    out_parallel = moe(x)
    out_ref = ref(x)
    assert torch.allclose(out_parallel, out_ref, rtol=1e-4, atol=1e-4)


def test_moe_ep_allreduce_compile_fx_contains_allreduce():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available.")

    world_size = _get_world_size()
    device_type = get_platform()
    if device_type not in ("npu", "mlu"):
        pytest.skip(f"Only npu/mlu are supported for this test, got {device_type}.")

    device = _set_current_device(device_type, world_size)
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
    ).to(device)
    _init_weights(ref)

    moe = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device)
    moe.load_state_dict(ref.state_dict())
    moe = parallelize_module(moe, device_mesh=device_mesh, parallelize_plan=MojoExpertParallel())

    x = torch.randn(32, hidden_size, device=device)
    out_eager = moe(x)

    out_compiled, graph_texts = _compile_capture_fx_and_run(moe, (x,))
    assert torch.allclose(out_compiled, out_eager, rtol=1e-4, atol=1e-4)
    assert _fx_graph_contains(graph_texts, "_c10d_functional.all_reduce")
