import os

import pytest
import torch
import torch.nn as nn

from mojo_opset import MojoMoE
from mojo_opset.utils.platform import get_platform
from tests.utils import bypass_not_implemented


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
    device = get_platform()
    torch.manual_seed(0)

    moe = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    for p in moe.parameters():
        nn.init.normal_(p, std=0.02)

    old_backend = os.environ.get("MOJO_BACKEND")
    os.environ["MOJO_BACKEND"] = "torch"
    moe_ref = MojoMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if old_backend is None:
        os.environ.pop("MOJO_BACKEND", None)
    else:
        os.environ["MOJO_BACKEND"] = old_backend

    moe = moe.to(dtype).to(device)
    moe_ref = moe_ref.to(dtype).to(device)
    moe_ref.load_state_dict(moe.state_dict())

    moe.gating.gate_weight.data = moe.gating.gate_weight.data.float()
    moe_ref.gating.gate_weight.data = moe_ref.gating.gate_weight.data.float()

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    moe.forward_diff_with(moe_ref, x, mixed_tol=True)
