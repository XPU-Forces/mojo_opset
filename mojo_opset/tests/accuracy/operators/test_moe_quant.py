import pytest
import torch

from mojo_opset import MojoFusedSwiGLUMoEScaleDynamicQuantize
from mojo_opset import MojoMoEInitRoutingDynamicQuant
from mojo_opset import MojoQuantExperts
from mojo_opset import MojoQuantMoE
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


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


def _quantize_w4_per_group(weight: torch.Tensor, quant_group_size: int):
    if weight.shape[-1] % quant_group_size != 0:
        raise ValueError(f"weight input dim {weight.shape[-1]} must be divisible by {quant_group_size}.")
    weight_groups = weight.float().split(quant_group_size, dim=-1)
    scales = []
    quzntizeds = []
    for weight_group in weight_groups:
        scale = (weight_group.abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-12)
        quzntizeds.append(torch.clamp(torch.round(weight_group / scale), -8, 7).to(torch.int8))
        scales.append(scale)
        
    quantized = torch.cat(quzntizeds, dim=-1)
    scale = torch.cat(scales, dim=-1)
    return _pack_int4_to_int8_along_output(quantized), scale


def _manual_quant_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype,
):
    print(f"{input.shape=} {weight.shape=}")
    input_groups = input.split(group_size, dim=-1)
    weight_groups = weight.split(group_size, -1)
    outs = []
    for group_idx, input_group in enumerate(input_groups):
        group_out = torch.mul(input_group.int().unsqueeze(-2), weight_groups[group_idx].int()).float().sum(dim=-1)
        print(f"{input_group.shape=} {weight_groups[group_idx].shape=} {group_out.shape=}")
        outs.append(group_out)
    print(f"{weight_scale.shape=} {input_scale.shape=}")
    out = torch.stack(outs, dim=-1) * weight_scale * input_scale.unsqueeze(-1)
    out = out.sum(dim=-1)
    return out.to(output_dtype)


def _manual_quant_experts(
    inputs: torch.Tensor,
    token_count: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_weight_scale: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_weight_scale: torch.Tensor,
    fc1_input_smooth_scale: torch.Tensor,
    fc2_input_smooth_scale: torch.Tensor,
    group_size: int,
):
    up_proj_weight = torch.repeat_interleave(up_proj_weight, token_count, dim=0)
    up_proj_weight_scale = torch.repeat_interleave(up_proj_weight_scale, token_count, dim=0)
    fc1_input_smooth_scale = torch.repeat_interleave(fc1_input_smooth_scale, token_count, dim=0)
    down_proj_weight = torch.repeat_interleave(down_proj_weight, token_count, dim=0)
    down_proj_weight_scale = torch.repeat_interleave(down_proj_weight_scale, token_count, dim=0)
    fc2_input_smooth_scale = torch.repeat_interleave(fc2_input_smooth_scale, token_count, dim=0)

    fc1_smoothed = inputs / fc1_input_smooth_scale.float()
    fc1_input_scale = fc1_smoothed.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    fc1_input_scale = torch.where(fc1_input_scale < 1e-6, 1.0, fc1_input_scale)
    fc1_input_i8 = torch.clamp(torch.round(fc1_smoothed / fc1_input_scale), -128, 127).to(
        torch.int8
    )
    fc1 = _manual_quant_linear(
        fc1_input_i8,
        fc1_input_scale,
        up_proj_weight,
        up_proj_weight_scale,
        group_size=group_size,
        output_dtype=inputs.dtype,
    )
    gate_proj, up_proj = fc1.float().chunk(2, dim=-1)
    activated = (torch.nn.functional.silu(gate_proj) * up_proj)
    smoothed_activated = activated / fc2_input_smooth_scale.float()
    fc2_input_scale = smoothed_activated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    fc2_input_scale = torch.where(fc2_input_scale < 1e-6, 1.0, fc2_input_scale)
    fc2_input = torch.clamp(torch.round(smoothed_activated / fc2_input_scale), -128, 127).to(
        torch.int8
    )

    fc2 = _manual_quant_linear(
        fc2_input,
        fc2_input_scale,
        down_proj_weight,
        down_proj_weight_scale,
        group_size=group_size,
        output_dtype=inputs.dtype
    )

    return fc2


def _make_quant_weights(num_experts: int, hidden_size: int, intermediate_size: int, quant_group_size: int):
    up_weight_fp = torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=torch.float32)
    down_weight_fp = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float32)
    up_weight, up_weight_scale = _quantize_w4_per_group(up_weight_fp, quant_group_size)
    down_weight, down_weight_scale = _quantize_w4_per_group(down_weight_fp, quant_group_size)
    return up_weight, up_weight_scale.bfloat16(), down_weight, down_weight_scale.bfloat16()


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


def test_quant_experts_reference():
    torch.manual_seed(0)
    num_experts = 3
    hidden_size = 8
    intermediate_size = 12
    quant_group_size = 4
    token_count = torch.tensor([2, 0, 3], dtype=torch.int32)
    total_tokens = int(token_count.sum().item())

    input_fp = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16)
    
    up_weight, up_weight_scale, down_weight, down_weight_scale = _make_quant_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        quant_group_size,
    )
    fc1_input_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32) + 0.5
    fc2_input_smooth_scale = torch.rand(num_experts, intermediate_size, dtype=torch.float32) + 0.5

    op = MojoQuantExperts._registry.get("torch")(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quant_type=torch.int8,
        quant_group_size=quant_group_size,
        weight_bits = 4,
    )
    op.load_state_dict(
        {
            "up_proj_weight": up_weight,
            "down_proj_weight": down_weight,
            "up_proj_weight_scale": up_weight_scale,
            "down_proj_weight_scale": down_weight_scale,
            "up_proj_quantize.smooth_scale": fc1_input_smooth_scale,
            "down_proj_quantize.smooth_scale": fc2_input_smooth_scale,
        }
    )

    out = op(input_fp, token_count)
    
    unpacked_up_weight = _unpack_int4_from_int8_along_output(up_weight)
    unpacked_down_weight = _unpack_int4_from_int8_along_output(down_weight)
    
    ref = _manual_quant_experts(
        input_fp,
        token_count,
        unpacked_up_weight,
        up_weight_scale,
        unpacked_down_weight,
        down_weight_scale,
        fc1_input_smooth_scale,
        fc2_input_smooth_scale,
        quant_group_size,
    )

    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    assert op.up_proj_weight.dtype == torch.int8
    assert op.down_proj_weight.dtype == torch.int8
    assert isinstance(op.up_proj_weight_scale, torch.nn.Parameter)
    assert isinstance(op.down_proj_weight_scale, torch.nn.Parameter)
    assert op.up_proj_weight.shape == (num_experts, intermediate_size, hidden_size)
    assert op.down_proj_weight.shape == (num_experts, hidden_size // 2, intermediate_size)
    assert op.up_proj_weight_scale.shape == (num_experts, intermediate_size * 2, hidden_size // quant_group_size)
    assert op.down_proj_weight_scale.shape == (num_experts, hidden_size, intermediate_size // quant_group_size)
    assert set(op.state_dict()) == {
        "up_proj_weight",
        "down_proj_weight",
        "up_proj_weight_scale",
        "down_proj_weight_scale",
        "up_proj_quantize.smooth_scale",
        "down_proj_quantize.smooth_scale",
    }

def test_quant_moe_reference():
    torch.manual_seed(1)
    num_tokens = 5
    num_experts = 4
    top_k = 2
    hidden_size = 8
    intermediate_size = 12
    quant_group_size = 4

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    gate_weight = torch.randn(hidden_size, num_experts, dtype=torch.float32) * 0.2
    fc1_input_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32) + 0.5
    fc2_input_smooth_scale = torch.rand(num_experts, intermediate_size, dtype=torch.float32) + 0.5
    up_weight, up_weight_scale, down_weight, down_weight_scale = _make_quant_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        quant_group_size,
    )

    op = MojoQuantMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        output_dtype=torch.bfloat16,
        quant_type=torch.int8,
        quant_group_size=quant_group_size,
        weight_bits=4,
    )
    op.load_state_dict(
        {
            "gating.gate_weight": gate_weight,
            "experts.up_proj_weight": up_weight,
            "experts.down_proj_weight": down_weight,
            "experts.up_proj_weight_scale": up_weight_scale,
            "experts.down_proj_weight_scale": down_weight_scale,
            "experts.up_proj_quantize.smooth_scale": fc1_input_smooth_scale,
            "experts.down_proj_quantize.smooth_scale": fc2_input_smooth_scale,
        }
    )

    top_k_indices, top_k_gates = op.gating(hidden_states)
    sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = op.dispatch(
        hidden_states,
        top_k_gates,
        top_k_indices,
    )

    unpacked_up_weight = _unpack_int4_from_int8_along_output(up_weight)
    unpacked_down_weight = _unpack_int4_from_int8_along_output(down_weight)
    

    expert_outputs = _manual_quant_experts(
        sorted_hidden_states,
        tokens_per_expert,
        unpacked_up_weight,
        up_weight_scale,
        unpacked_down_weight,
        down_weight_scale,
        fc1_input_smooth_scale,
        fc2_input_smooth_scale,
        quant_group_size,
    )
    ref = torch.zeros_like(hidden_states, dtype=torch.float32)
    ref.scatter_reduce_(
        0,
        token_indices.to(torch.int64).unsqueeze(-1).expand(-1, hidden_size),
        expert_outputs.float() * sorted_gates.float(),
        reduce="sum",
        include_self=True,
    )

    out = op(hidden_states)
    torch.testing.assert_close(out, ref.to(torch.bfloat16), atol=0, rtol=0)
    assert set(op.state_dict()) == {
        "gating.gate_weight",
        "experts.up_proj_quantize.smooth_scale",
        "experts.up_proj_weight",
        "experts.down_proj_weight",
        "experts.up_proj_weight_scale",
        "experts.down_proj_weight_scale",
        "experts.down_proj_quantize.smooth_scale",
    }
