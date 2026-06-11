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

    moe_ref = MojoMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    for p in moe_ref.parameters():
        nn.init.normal_(p, std=0.02)

    moe = moe.to(dtype).to(device)
    moe_ref = moe_ref.to(dtype).to(device)
    moe.load_state_dict(moe_ref.state_dict())

    # FIXME: moe.gating.gate_weight.data should not be casted to float32
    moe.gating.gate_weight.data = moe.gating.gate_weight.data.float()
    moe_ref.gating.gate_weight.data = moe_ref.gating.gate_weight.data.float()

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    moe.forward_diff_with(moe_ref, x, mixed_tol=True)


@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, intermediate_size, num_tokens",
    [
        (16, 4, 1024, 2048, 64),
        (32, 8, 1024, 4096, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_forced_expert_ids_matches_topk_route(
    num_experts, top_k, hidden_size, intermediate_size, num_tokens, dtype
):
    device = get_torch_device()
    torch.manual_seed(0)

    moe = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    moe_ref = MojoMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    for p in moe_ref.parameters():
        nn.init.normal_(p, std=0.02)

    moe = moe.to(dtype).to(device)
    moe_ref = moe_ref.to(dtype).to(device)
    moe.load_state_dict(moe_ref.state_dict())

    # FIXME: moe.gating.gate_weight.data should not be casted to float32
    moe.gating.gate_weight.data = moe.gating.gate_weight.data.float()
    moe_ref.gating.gate_weight.data = moe_ref.gating.gate_weight.data.float()

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    forced_expert_ids, _ = moe_ref.gating(x)

    out = moe(x, forced_expert_ids=forced_expert_ids)
    out_ref = moe_ref(x)
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


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

    moe_gating_ref = MojoMoEGating._registry.get("torch")(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    for p in moe_gating_ref.parameters():
        nn.init.normal_(p, std=0.02)

    moe_gating = moe_gating.to(device)
    moe_gating_ref = moe_gating_ref.to(device)
    moe_gating.load_state_dict(moe_gating_ref.state_dict())

    assert moe_gating.gate_weight.dtype == torch.float32 and moe_gating_ref.gate_weight.dtype == torch.float32

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    moe_gating.forward_diff_with(
        moe_gating_ref, x,
        atol=(0, 1e-2),
        rtol=(0, 1e-2),
        ptol=(0.999, 1.0),
    )


@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, num_tokens",
    [
        (16, 4, 1024, 64),
        (32, 8, 1024, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_gating_forced_expert_ids_matches_topk_route(num_experts, top_k, hidden_size, num_tokens, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_gating = MojoMoEGating(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    for p in moe_gating.parameters():
        nn.init.normal_(p, std=0.02)

    moe_gating = moe_gating.to(device)
    assert moe_gating.gate_weight.dtype == torch.float32

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    top_k_indices, top_k_gates = moe_gating(x)
    forced_indices, forced_gates = moe_gating(x, forced_expert_ids=top_k_indices)

    torch.testing.assert_close(forced_indices, top_k_indices, atol=0, rtol=0)
    torch.testing.assert_close(forced_gates, top_k_gates, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_gating_forced_expert_ids_gathers_requested_routes(dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_gating = MojoMoEGating(
        hidden_size=8,
        num_experts=4,
        top_k=2,
    )

    for p in moe_gating.parameters():
        nn.init.normal_(p, std=0.02)

    moe_gating = moe_gating.to(device)
    assert moe_gating.gate_weight.dtype == torch.float32

    x = torch.rand(3, 8, dtype=dtype, device=device)
    forced_expert_ids = torch.tensor([[0, 1], [2, 3], [3, 0]], dtype=torch.int64, device=device)
    forced_indices, forced_gates = moe_gating(x, forced_expert_ids=forced_expert_ids)

    gate_probs = torch.softmax(torch.matmul(x.float(), moe_gating.gate_weight), dim=-1)
    expected_gates = torch.gather(gate_probs, dim=-1, index=forced_expert_ids)
    expected_gates = expected_gates / torch.sum(expected_gates, dim=-1, keepdim=True)

    torch.testing.assert_close(forced_indices, forced_expert_ids.to(torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(forced_gates, expected_gates, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_gating_forced_expert_ids_negative_one_keeps_topk_route(dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_gating = MojoMoEGating(
        hidden_size=8,
        num_experts=4,
        top_k=2,
    )

    for p in moe_gating.parameters():
        nn.init.normal_(p, std=0.02)

    moe_gating = moe_gating.to(device)
    assert moe_gating.gate_weight.dtype == torch.float32

    x = torch.rand(3, 8, dtype=dtype, device=device)
    top_k_indices, _ = moe_gating(x)
    forced_expert_ids = torch.tensor([[0, -1], [-1, 3], [2, -1]], dtype=torch.int64, device=device)
    forced_indices, forced_gates = moe_gating(x, forced_expert_ids=forced_expert_ids)

    forced_expert_mask = forced_expert_ids >= 0
    expected_indices = torch.where(forced_expert_mask, forced_expert_ids, top_k_indices.to(torch.int64))
    gate_probs = torch.softmax(torch.matmul(x.float(), moe_gating.gate_weight), dim=-1)
    expected_gates = torch.gather(gate_probs, dim=-1, index=expected_indices)
    expected_gates = expected_gates / torch.sum(expected_gates, dim=-1, keepdim=True)

    torch.testing.assert_close(forced_indices, expected_indices.to(torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(forced_gates, expected_gates, atol=1e-5, rtol=1e-5)
