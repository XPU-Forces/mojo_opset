from typing import Union

import os
import pytest
import torch

from mojo_opset import MojoQuantExperts
from mojo_opset import MojoQuantMoE
from mojo_opset.experimental import MojoFusedSwiGLUMoEScaleDynamicQuantize
from mojo_opset.experimental import MojoMoEInitRoutingDynamicQuant
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device, get_platform


def _pack_int4_to_int8_along_output(input: torch.Tensor) -> torch.Tensor:
    input_u8 = input.to(torch.uint8)
    packed = ((input_u8[..., 1::2, :] & 0x0F) << 4) | (input_u8[..., 0::2, :] & 0x0F)
    return packed.to(torch.int8)


def _unpack_int4_from_int8_along_output(input: torch.Tensor) -> torch.Tensor:
    input_u8 = input.to(torch.uint8)
    low = (input_u8 & 0x0F).to(torch.int8)
    high = ((input_u8 >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    output = torch.empty(*input.shape[:-2], input.shape[-2] * 2, input.shape[-1], dtype=torch.int8, device=input.device)
    output[..., 0::2, :] = low
    output[..., 1::2, :] = high
    return output


def _quantize_weight_per_group(weight: torch.Tensor, quant_group_size: int, weight_dtype: Union[torch.dtype, str]):
    if weight.shape[-1] % quant_group_size != 0:
        raise ValueError(f"weight input dim {weight.shape[-1]} must be divisible by {quant_group_size}.")
    if quant_group_size > 0:
        weight_groups = weight.float().split(quant_group_size, dim=-1)
    else:
        weight_groups = [weight]
    scales = []
    quantizeds = []
    for weight_group in weight_groups:
        scale = (weight_group.abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-12)
        quantizeds.append(torch.clamp(torch.round(weight_group / scale), -8, 7).to(torch.int8))
        scales.append(scale)

    quantized = torch.cat(quantizeds, dim=-1)
    scale = torch.cat(scales, dim=-1)
    if quant_group_size <= 0:
        scale = scale.squeeze(-1)
    return _pack_int4_to_int8_along_output(quantized) if weight_dtype == "int4" else quantized, scale


def _make_quant_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    up_quant_group_size: int,
    up_weight_dtype: Union[torch.dtype, str],
    down_quant_group_size: int,
    down_weight_dtype: Union[torch.dtype, str],
):
    # Keep int4 packing off NPU because its bitwise ops may fall back to TBE.
    up_weight_fp = (
        torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=torch.float32, device="cpu") * 0.01
    )
    down_weight_fp = (
        torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float32, device="cpu") * 0.01
    )
    up_weight, up_weight_scale = _quantize_weight_per_group(up_weight_fp, up_quant_group_size, up_weight_dtype)
    down_weight, down_weight_scale = _quantize_weight_per_group(down_weight_fp, down_quant_group_size, down_weight_dtype)
    return up_weight, up_weight_scale.bfloat16(), down_weight, down_weight_scale.bfloat16()


quant_moe_backend_cases = [
    (16, 2, 512, 1280, 33),
    (24, 4, 512, 1280, 97),
]


def test_moe_init_routing_dynamic_quant_reference():
    hidden_states = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    top_k_gates = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float32)
    top_k_indices = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
    smooth_scale = torch.ones(2, 8, dtype=torch.float32)

    op = MojoMoEInitRoutingDynamicQuant._registry.get("torch")(num_experts=2, top_k=2, quant_block_size=8)
    quantized, sorted_gates, sorted_token_indices, token_count, scale = op(
        hidden_states,
        top_k_gates,
        top_k_indices,
        smooth_scale,
    )

    sorted_hidden = torch.stack((hidden_states[0], hidden_states[1], hidden_states[0], hidden_states[1])).reshape(
        2, 2, 8
    )
    expected_scale = sorted_hidden.abs().amax(dim=-1, keepdim=True) / 127
    expected_quantized = torch.clamp(torch.round(sorted_hidden / expected_scale), -128, 127).to(torch.int8)

    assert quantized.shape == (2, 2, 8)
    assert quantized.dtype == torch.int8
    torch.testing.assert_close(quantized, expected_quantized, atol=0, rtol=0)
    torch.testing.assert_close(sorted_gates, torch.tensor([[[0.1], [0.8]], [[0.9], [0.2]]]))
    torch.testing.assert_close(sorted_token_indices, torch.tensor([[[0], [1]], [[0], [1]]], dtype=torch.int32))
    torch.testing.assert_close(token_count, torch.tensor([2, 2], dtype=torch.int32))
    torch.testing.assert_close(scale, expected_scale)


def test_fused_swiglu_moe_scale_dynamic_quant_reference():
    input = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]],
            [[2.0, 1.0, 4.0, 2.0], [1.0, 0.5, 2.0, 1.0]],
        ],
        dtype=torch.bfloat16,
    )
    smooth_scale = torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=torch.float32)
    token_count = torch.tensor([2, 2], dtype=torch.int32)

    op = MojoFusedSwiGLUMoEScaleDynamicQuantize._registry.get("torch")()
    quantized, scale = op(input, smooth_scale, token_count, 1.0, 0)

    expanded_scale = torch.tensor(
        [
            [[1.0, 2.0], [1.0, 2.0]],
            [[0.5, 1.5], [0.5, 1.5]],
        ],
        dtype=torch.float32,
    )
    left, right = input.float().chunk(2, dim=-1)
    expected = torch.nn.functional.silu(left) * right
    expected = expected * expanded_scale
    expected_scale = expected.abs().amax(dim=-1).clamp(min=1e-12) / 127
    expected_quantized = torch.clamp(torch.round(expected / expected_scale.unsqueeze(-1)), -128, 127).to(torch.int8)

    torch.testing.assert_close(quantized, expected_quantized, atol=0, rtol=0)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_init_routing_dynamic_quant_backend(dtype):
    device = get_torch_device()
    seq_len = 8
    num_experts = 4
    top_k = 2
    hidden_size = 64

    hidden_states = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)
    gate_logits = torch.randn(seq_len, num_experts, dtype=torch.float32, device=device)
    gate_probs = torch.softmax(gate_logits, dim=-1)
    top_k_logits, top_k_indices = torch.topk(gate_probs, top_k, dim=-1)
    top_k_gates = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True)
    top_k_indices = top_k_indices.to(torch.int32)
    smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device=device)

    op = MojoMoEInitRoutingDynamicQuant(
        num_experts=num_experts,
        top_k=top_k,
        quant_block_size=hidden_size,
    ).to(device)
    op_ref = MojoMoEInitRoutingDynamicQuant._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        quant_block_size=hidden_size,
    ).to(device)
    op.forward_diff_with(
        op_ref,
        hidden_states,
        top_k_gates,
        top_k_indices,
        smooth_scale,
        0,
        atol=(1, 1e-4, 0, 0, 1e-4),
        rtol=(0, 1e-4, 0, 0, 1e-4),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_fused_swiglu_moe_scale_dynamic_quant_backend(dtype):
    device = get_torch_device()
    seq_len = 8
    top_k = 2
    expert_num = 4
    last_dim = 128

    input = torch.randn(seq_len, top_k, last_dim, dtype=dtype, device=device)
    smooth_scale = torch.rand(expert_num, last_dim // 2, dtype=torch.float32, device=device)
    token_count = torch.tensor([4, 5, 3, 4], dtype=torch.int32, device=device)

    op = MojoFusedSwiGLUMoEScaleDynamicQuantize().to(device)
    op_ref = MojoFusedSwiGLUMoEScaleDynamicQuantize._registry.get("torch")().to(device)
    op.forward_diff_with(
        op_ref,
        input,
        smooth_scale,
        token_count,
        1.0,
        0,
        atol=(1, 1e-4),
        rtol=(0, 1e-4),
    )


@pytest.mark.parametrize("num_experts, top_k, hidden_size, intermediate_size, num_tokens", quant_moe_backend_cases)
@pytest.mark.parametrize(
    "up_weight_dtype,up_quant_group_size,down_weight_dtype,down_quant_group_size",
    [("int4", 512, "int4", 320), (torch.int8, -1, torch.int8, -1)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@auto_switch_platform()
@bypass_not_implemented
def test_quant_experts(
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    num_tokens,
    up_weight_dtype,
    up_quant_group_size,
    down_weight_dtype,
    down_quant_group_size,
    dtype,
):
    torch.manual_seed(0)
    device = get_torch_device()

    # Note: use 2 * num_experts to mimic EP scenarios
    expert_indices = torch.randint(0, num_experts * 2, (num_tokens, top_k))

    token_count = torch.bincount(expert_indices.flatten(), minlength=num_experts)[:num_experts].to(torch.int32).to(device)
    total_tokens = int(token_count.sum().item())
    input_fp = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)

    fc1_input_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device=device) + 0.5
    fc2_input_smooth_scale = torch.rand(num_experts, intermediate_size, dtype=torch.float32, device=device) + 0.5
    up_weight, up_weight_scale, down_weight, down_weight_scale = _make_quant_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        up_quant_group_size,
        up_weight_dtype,
        down_quant_group_size,
        down_weight_dtype,
    )

    state_dict = {
        "up_proj_weight": up_weight.to(device),
        "down_proj_weight": down_weight.to(device),
        "up_proj_weight_scale": up_weight_scale.to(device),
        "down_proj_weight_scale": down_weight_scale.to(device),
        "up_proj_quantize.inv_smooth_scale": (1.0 / fc1_input_smooth_scale).to(device),
        "down_proj_quantize.inv_smooth_scale": (1.0 / fc2_input_smooth_scale).to(device),
    }
    op = MojoQuantExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quant_dtype=torch.int8,
        up_quant_group_size=up_quant_group_size,
        up_weight_dtype=up_weight_dtype,
        down_quant_group_size=down_quant_group_size,
        down_weight_dtype=down_weight_dtype,
    ).to(device)
    op_ref = MojoQuantExperts._registry.get("torch")(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quant_dtype=torch.int8,
        up_quant_group_size=up_quant_group_size,
        up_weight_dtype=up_weight_dtype,
        down_quant_group_size=down_quant_group_size,
        down_weight_dtype=down_weight_dtype,
    ).to(device)
    op.load_state_dict(state_dict)
    op_ref.load_state_dict({k: v.clone() for k, v in state_dict.items()})

    op.forward_diff_with(op_ref, input_fp, token_count, mixed_tol=True)

@pytest.mark.parametrize("num_experts, top_k, hidden_size, intermediate_size, num_tokens", quant_moe_backend_cases)
@pytest.mark.parametrize(
    "up_weight_dtype,up_quant_group_size,down_weight_dtype,down_quant_group_size",
    [("int4", 512, "int4", 320), (torch.int8, -1, torch.int8, -1)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@auto_switch_platform()
@bypass_not_implemented
def test_quant_moe(
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    num_tokens,
    up_weight_dtype,
    up_quant_group_size,
    down_weight_dtype,
    down_quant_group_size,
    dtype,
):
    device = get_torch_device()

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    gate_weight = torch.randn(hidden_size, num_experts, dtype=torch.float32, device=device) * 0.2
    fc1_input_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device=device) + 0.5
    fc2_input_smooth_scale = torch.rand(num_experts, intermediate_size, dtype=torch.float32, device=device) + 0.5
    up_weight, up_weight_scale, down_weight, down_weight_scale = _make_quant_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        up_quant_group_size,
        up_weight_dtype,
        down_quant_group_size,
        down_weight_dtype,
    )

    state_dict = {
        "gating.gate_weight": gate_weight.to(device),
        "experts.up_proj_weight": up_weight.to(device),
        "experts.down_proj_weight": down_weight.to(device),
        "experts.up_proj_weight_scale": up_weight_scale.to(device),
        "experts.down_proj_weight_scale": down_weight_scale.to(device),
        "experts.up_proj_quantize.inv_smooth_scale": (1.0 / fc1_input_smooth_scale).to(device),
        "experts.down_proj_quantize.inv_smooth_scale": (1.0 / fc2_input_smooth_scale).to(device),
    }

    op = MojoQuantMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quant_dtype=torch.int8,
        up_quant_group_size=up_quant_group_size,
        up_weight_dtype=up_weight_dtype,
        down_quant_group_size=down_quant_group_size,
        down_weight_dtype=down_weight_dtype,
    ).to(device)
    op_ref = MojoQuantMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quant_dtype=torch.int8,
        up_quant_group_size=up_quant_group_size,
        up_weight_dtype=up_weight_dtype,
        down_quant_group_size=down_quant_group_size,
        down_weight_dtype=down_weight_dtype,
    ).to(device)
    op.load_state_dict(state_dict)
    op_ref.load_state_dict({k: v.clone() for k, v in state_dict.items()})
    op.forward_diff_with(op_ref, hidden_states, mixed_tol=True)
