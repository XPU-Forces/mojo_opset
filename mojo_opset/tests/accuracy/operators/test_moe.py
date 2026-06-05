import os

import pytest
import torch
import torch.nn as nn

from mojo_opset import MojoExperts
from mojo_opset import MojoMoE
from mojo_opset import MojoMoECombine
from mojo_opset import MojoMoEDispatch
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


def _canonicalize_dispatch(sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices):
    sorted_hidden = []
    sorted_gate = []
    sorted_token = []
    start = 0
    for count in tokens_per_expert.tolist():
        end = start + count
        if count > 0:
            seg_tokens = token_indices[start:end].to(torch.int64)
            order = torch.argsort(seg_tokens, stable=True)
            sorted_hidden.append(sorted_hidden_states[start:end][order])
            sorted_gate.append(sorted_gates[start:end][order])
            sorted_token.append(seg_tokens[order])
        start = end
    return (
        torch.cat(sorted_hidden, dim=0),
        torch.cat(sorted_gate, dim=0),
        torch.cat(sorted_token, dim=0),
    )

@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, num_tokens",
    [
        (16, 4, 1024, 64),
        (32, 8, 1024, 128),
        (64, 8, 1024, 256),
        (128, 8, 4096, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_dispatch(num_experts, top_k, hidden_size, num_tokens, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_dispatch = MojoMoEDispatch(num_experts=num_experts)
    moe_dispatch_ref = MojoMoEDispatch._registry.get("torch")(num_experts=num_experts)

    hidden_states = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    gate_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    probs = torch.softmax(gate_logits, dim=-1)
    top_k_gates, top_k_indices = torch.topk(probs, top_k, dim=-1)
    top_k_gates = (top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)).contiguous()
    top_k_indices = top_k_indices.to(torch.int32).contiguous()

    sorted_hidden, tokens_per_expert, sorted_gates, token_indices = moe_dispatch(
        hidden_states,
        top_k_gates,
        top_k_indices,
    )
    ref_hidden, ref_tokens_per_expert, ref_gates, ref_token_indices = moe_dispatch_ref(
        hidden_states,
        top_k_gates,
        top_k_indices,
    )

    torch.testing.assert_close(tokens_per_expert, ref_tokens_per_expert, atol=0, rtol=0)

    sorted_hidden, sorted_gates, token_indices = _canonicalize_dispatch(
        sorted_hidden,
        tokens_per_expert,
        sorted_gates,
        token_indices,
    )
    ref_hidden, ref_gates, ref_token_indices = _canonicalize_dispatch(
        ref_hidden,
        ref_tokens_per_expert,
        ref_gates,
        ref_token_indices,
    )
    torch.testing.assert_close(token_indices, ref_token_indices, atol=0, rtol=0)
    torch.testing.assert_close(sorted_gates, ref_gates, atol=0, rtol=0)
    torch.testing.assert_close(sorted_hidden, ref_hidden, atol=0, rtol=0)


@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, num_tokens",
    [
        (16, 4, 1024, 64),
        (32, 8, 1024, 128),
        (64, 8, 1024, 256),
        (128, 8, 4096, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_combine(num_experts, top_k, hidden_size, num_tokens, dtype):
    device = get_torch_device()
    torch.manual_seed(0)

    moe_dispatch_ref = MojoMoEDispatch._registry.get("torch")(num_experts=num_experts)
    moe_combine = MojoMoECombine(multiply_by_gates=True)
    moe_combine_ref = MojoMoECombine._registry.get("torch")(multiply_by_gates=True)

    hidden_states = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    expert_outputs = torch.randn(num_tokens * top_k, hidden_size, dtype=dtype, device=device)
    gate_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    probs = torch.softmax(gate_logits, dim=-1)
    top_k_gates, top_k_indices = torch.topk(probs, top_k, dim=-1)
    top_k_gates = (top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)).contiguous()
    top_k_indices = top_k_indices.to(torch.int32).contiguous()

    _, _, sorted_gates, token_indices = moe_dispatch_ref(
        hidden_states,
        top_k_gates,
        top_k_indices,
    )
    output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
    output_buffer_ref = output_buffer.clone()

    out = moe_combine(output_buffer, expert_outputs, sorted_gates, token_indices)
    ref = moe_combine_ref(output_buffer_ref, expert_outputs, sorted_gates, token_indices)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

