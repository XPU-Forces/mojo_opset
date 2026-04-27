import os

import pytest
import torch
import torch.nn as nn

from mojo_opset import MojoMoE
from mojo_opset import MojoMoEGating
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import bypass_not_implemented


@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, intermediate_size, num_tokens",
    [
        (16, 4, 1024, 2048, 64),
        (32, 8, 1024, 4096, 128),
        (64, 8, 1024, 4096, 256),
        (64, 8, 1024, 4096, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe(num_experts, top_k, hidden_size, intermediate_size, num_tokens, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    for p in moe.parameters():
        nn.init.normal_(p, std=0.02)

    moe_ref = MojoMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    moe = moe.to(dtype).to(device)
    moe_ref = moe_ref.to(dtype).to(device)
    moe_ref.load_state_dict(moe.state_dict())

    # FIXME: moe.gating.gate_weight.data should not be casted to float32
    moe.gating.gate_weight.data = moe.gating.gate_weight.data.float()
    moe_ref.gating.gate_weight.data = moe_ref.gating.gate_weight.data.float()

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    moe.forward_diff_with(moe_ref, x, mixed_tol=True)


@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, num_tokens",
    [
        (16, 4, 1024, 64),
        (32, 8, 1024, 128),
        (64, 8, 1024, 256),
        (64, 8, 1024, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_gating(num_experts, top_k, hidden_size, num_tokens, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_gating = MojoMoEGating(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    for p in moe_gating.parameters():
        nn.init.normal_(p, std=0.02)

    moe_gating_ref = MojoMoEGating._registry.get("torch")(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    moe_gating = moe_gating.to(device)
    moe_gating_ref = moe_gating_ref.to(device)
    moe_gating_ref.load_state_dict(moe_gating.state_dict())

    assert moe_gating.gate_weight.dtype == torch.float32 and moe_gating_ref.gate_weight.dtype == torch.float32

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    moe_gating.forward_diff_with(moe_gating_ref, x, mixed_tol=True)
