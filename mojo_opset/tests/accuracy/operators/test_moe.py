import os

import pytest
import torch
import torch.nn as nn

from mojo_opset import MojoExperts
from mojo_opset import MojoMoE
from mojo_opset import MojoMoECombine
from mojo_opset import MojoMoEDispatch
from mojo_opset import MojoMoEGating
from mojo_opset import MojoExperts
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
def test_experts(num_experts, top_k, hidden_size, intermediate_size, num_tokens, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    # Note: use 2 * num_experts to mimic EP scenarios
    expert_indices = torch.randint(0, num_experts * 2, (num_tokens, top_k))

    token_count = torch.bincount(expert_indices.flatten(), minlength=num_experts)[:num_experts].to(torch.int32).to(device)
    total_tokens = int(token_count.sum().item())
    input_fp = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)

    moe = MojoExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    moe_ref = MojoExperts._registry.get("torch")(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    moe = moe.to(dtype).to(device)
    moe_ref = moe_ref.to(dtype).to(device)

    for p in moe_ref.parameters():
        nn.init.normal_(p, std=0.02)

    moe.load_state_dict(moe_ref.state_dict())

    moe.forward_diff_with(moe_ref, input_fp, token_count, mixed_tol=True)


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

    moe = moe.to(dtype).to(device)
    moe_ref = moe_ref.to(dtype).to(device)
    # FIXME: moe.gating.gate_weight.data should not be casted to float32
    moe.gating.gate_weight.data = moe.gating.gate_weight.data.float()
    moe_ref.gating.gate_weight.data = moe_ref.gating.gate_weight.data.float()

    for p in moe_ref.parameters():
        nn.init.normal_(p, std=0.02)

    moe.load_state_dict(moe_ref.state_dict())

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
        (64, 8, 1024, 256),
        (384, 8, 3584, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_dispatch(num_experts, top_k, hidden_size, num_tokens, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_dispatch = MojoMoEDispatch(num_experts=num_experts)
    moe_dispatch_ref = MojoMoEDispatch._registry.get("torch")(num_experts=num_experts)

    moe_dispatch = moe_dispatch.to(device)
    moe_dispatch_ref = moe_dispatch_ref.to(device)

    hidden_states = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    gate_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    gate_probs = torch.softmax(gate_logits, dim=-1)
    top_k_gates, top_k_indices = torch.topk(gate_probs, top_k, dim=-1)
    top_k_gates = (top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)).contiguous()
    top_k_indices = top_k_indices.to(torch.int32).contiguous()

    sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = moe_dispatch(
        hidden_states,
        top_k_gates,
        top_k_indices,
    )
    _ref_hidden_states, ref_tokens_per_expert, _ref_gates, _ref_token_indices = moe_dispatch_ref(
        hidden_states,
        top_k_gates,
        top_k_indices,
    )

    torch.testing.assert_close(tokens_per_expert.to(device), ref_tokens_per_expert.to(device), atol=0, rtol=0)
    torch.testing.assert_close(sorted_hidden_states, hidden_states[token_indices.to(torch.int64)], atol=0, rtol=0)

    expert_offsets = torch.cumsum(tokens_per_expert.to(torch.int64), dim=0)
    expert_starts = torch.cat((expert_offsets.new_zeros(1), expert_offsets[:-1]))
    for expert_idx, (start, end) in enumerate(zip(expert_starts.tolist(), expert_offsets.tolist())):
        if start == end:
            continue
        token_slice = token_indices[start:end].to(torch.int64)
        expert_match = top_k_indices[token_slice] == expert_idx
        assert torch.all(expert_match.any(dim=-1))
        gate_pos = expert_match.to(torch.int64).argmax(dim=-1)
        expected_gates = top_k_gates[token_slice, gate_pos].unsqueeze(-1)
        torch.testing.assert_close(sorted_gates[start:end], expected_gates, atol=0, rtol=0)


@pytest.mark.parametrize(
    "num_experts, hidden_size, intermediate_size, tokens_per_expert",
    [
        (4, 256, 512, [3, 0, 5, 4]),
        (8, 512, 1024, [2, 1, 0, 3, 4, 0, 5, 2]),
        (384, 3584, 64, [2, 1, 0, 3] + [0] * 380),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_experts(num_experts, hidden_size, intermediate_size, tokens_per_expert, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_experts = MojoExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    moe_experts_ref = MojoExperts._registry.get("torch")(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    for p in moe_experts_ref.parameters():
        nn.init.normal_(p, std=0.02)

    moe_experts = moe_experts.to(dtype).to(device)
    moe_experts_ref = moe_experts_ref.to(dtype).to(device)
    moe_experts.load_state_dict(moe_experts_ref.state_dict())

    token_count = torch.tensor(tokens_per_expert, dtype=torch.int32, device=device)
    sorted_hidden_states = torch.rand(int(token_count.sum().item()), hidden_size, dtype=dtype, device=device)

    moe_experts.forward_diff_with(
        moe_experts_ref,
        sorted_hidden_states,
        token_count,
        mixed_tol=True,
    )


@pytest.mark.parametrize(
    "num_tokens, top_k, hidden_size",
    [
        (64, 4, 1024),
        (128, 8, 1024),
        (256, 8, 1024),
        (128, 8, 3584),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_combine(num_tokens, top_k, hidden_size, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_combine = MojoMoECombine(multiply_by_gates=True)
    moe_combine_ref = MojoMoECombine._registry.get("torch")(multiply_by_gates=True)

    moe_combine = moe_combine.to(device)
    moe_combine_ref = moe_combine_ref.to(device)

    expanded_tokens = num_tokens * top_k
    output_buffer = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
    expert_outputs = torch.randn(expanded_tokens, hidden_size, dtype=dtype, device=device)
    sorted_gates = torch.rand(expanded_tokens, 1, dtype=torch.float32, device=device)
    token_indices = (
        torch.arange(num_tokens, device=device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
        .contiguous()
    )

    perm = torch.randperm(expanded_tokens, device=device)
    expert_outputs = expert_outputs[perm].contiguous()
    sorted_gates = sorted_gates[perm].contiguous()
    token_indices = token_indices[perm].contiguous()

    moe_combine.forward_diff_with(
        moe_combine_ref,
        output_buffer,
        expert_outputs,
        sorted_gates,
        token_indices,
        mixed_tol=True,
    )
