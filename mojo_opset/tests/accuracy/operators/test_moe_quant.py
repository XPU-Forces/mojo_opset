import torch

from mojo_opset import MojoQuantExperts
from mojo_opset import MojoQuantMoE


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
    group_num = weight.shape[-1] // quant_group_size
    weight_groups = weight.float().reshape(*weight.shape[:-1], group_num, quant_group_size)
    scale = (weight_groups.abs().amax(dim=-1) / 7).clamp(min=1e-12)
    quantized = torch.clamp(torch.round(weight_groups / scale.unsqueeze(-1)), -8, 7).to(torch.int8)
    quantized = quantized.reshape_as(weight)
    return _pack_int4_to_int8_along_output(quantized), scale


def _manual_quant_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
):
    weight = _unpack_int4_from_int8_along_output(packed_weight)
    group_num = weight_scale.shape[1]
    input_groups = input.float().reshape(input.shape[0], group_num, -1)
    weight_groups = weight.float().reshape(weight.shape[0], group_num, -1)
    out = torch.zeros(input.shape[0], weight.shape[0], dtype=torch.float32, device=input.device)
    for group_idx in range(group_num):
        group_out = input_groups[:, group_idx, :] @ weight_groups[:, group_idx, :].transpose(0, 1)
        out = out + group_out * weight_scale[:, group_idx].float().unsqueeze(0)
    return (out * input_scale.reshape(-1, 1).float()).to(output_dtype)


def _manual_quant_experts(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    token_count: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_weight_scale: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_weight_scale: torch.Tensor,
    fc2_input_smooth_scale: torch.Tensor,
    output_dtype: torch.dtype,
):
    expert_inputs = torch.split(input, token_count.tolist(), dim=0)
    expert_input_scales = torch.split(input_scale.reshape(-1), token_count.tolist(), dim=0)
    outputs = []

    for expert_idx, expert_input in enumerate(expert_inputs):
        if expert_input.numel() == 0:
            outputs.append(expert_input.new_empty((0, down_proj_weight.shape[1] * 2), dtype=output_dtype))
            continue

        fc1 = _manual_quant_linear(
            expert_input,
            expert_input_scales[expert_idx],
            up_proj_weight[expert_idx],
            up_proj_weight_scale[expert_idx],
            output_dtype,
        )
        gate_proj, up_proj = fc1.float().chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate_proj) * up_proj).to(output_dtype)
        smoothed_activated = activated * fc2_input_smooth_scale[expert_idx].float().unsqueeze(0)
        fc2_input_scale = smoothed_activated.abs().amax(dim=-1).clamp(min=1e-12) / 127
        fc2_input = torch.clamp(torch.round(smoothed_activated / fc2_input_scale.unsqueeze(-1)), -128, 127).to(
            torch.int8
        )

        fc2 = _manual_quant_linear(
            fc2_input,
            fc2_input_scale,
            down_proj_weight[expert_idx],
            down_proj_weight_scale[expert_idx],
            output_dtype,
        )
        outputs.append(fc2)

    return torch.cat(outputs, dim=0)


def _make_quant_weights(num_experts: int, hidden_size: int, intermediate_size: int, quant_group_size: int):
    up_weight_fp = torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=torch.float32)
    down_weight_fp = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float32)
    up_weight, up_weight_scale = _quantize_w4_per_group(up_weight_fp, quant_group_size)
    down_weight, down_weight_scale = _quantize_w4_per_group(down_weight_fp, quant_group_size)
    return up_weight, up_weight_scale, down_weight, down_weight_scale


def test_quant_experts_reference():
    torch.manual_seed(0)
    num_experts = 3
    hidden_size = 8
    intermediate_size = 12
    quant_group_size = 4
    token_count = torch.tensor([2, 0, 3], dtype=torch.int32)
    total_tokens = int(token_count.sum().item())

    input_fp = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16)
    input_scale = input_fp.float().abs().amax(dim=-1).clamp(min=1e-12) / 127
    input_i8 = torch.clamp(torch.round(input_fp.float() / input_scale.unsqueeze(-1)), -128, 127).to(torch.int8)

    up_weight, up_weight_scale, down_weight, down_weight_scale = _make_quant_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        quant_group_size,
    )
    fc2_input_smooth_scale = torch.rand(num_experts, intermediate_size, dtype=torch.float32) + 0.5

    op = MojoQuantExperts._registry.get("torch")(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        output_dtype=torch.bfloat16,
        quant_type="int4",
        quant_group_size=quant_group_size,
    )
    op.load_state_dict(
        {
            "up_proj_weight": up_weight,
            "down_proj_weight": down_weight,
            "up_proj_weight_scale": up_weight_scale,
            "down_proj_weight_scale": down_weight_scale,
            "fc2_input_quant.smooth_scale": fc2_input_smooth_scale,
        }
    )

    out = op(input_i8, input_scale, token_count)
    ref = _manual_quant_experts(
        input_i8,
        input_scale,
        token_count,
        up_weight,
        up_weight_scale,
        down_weight,
        down_weight_scale,
        fc2_input_smooth_scale,
        torch.bfloat16,
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
        "fc2_input_quant.smooth_scale",
    }


def test_quant_experts_rejects_int8_until_implemented():
    try:
        MojoQuantExperts._registry.get("torch")(
            num_experts=1,
            hidden_size=8,
            intermediate_size=12,
            quant_type="int8",
            quant_group_size=4,
        )
    except NotImplementedError as exc:
        assert "quant_type='int4'" in str(exc)
    else:
        raise AssertionError("quant_type='int8' should be rejected until int8 expert weights are implemented.")


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
    smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32) + 0.5
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
        quant_type="int4",
        quant_group_size=quant_group_size,
    )
    op.load_state_dict(
        {
            "gating.gate_weight": gate_weight,
            "input_quant.smooth_scale": smooth_scale,
            "experts.up_proj_weight": up_weight,
            "experts.down_proj_weight": down_weight,
            "experts.up_proj_weight_scale": up_weight_scale,
            "experts.down_proj_weight_scale": down_weight_scale,
            "experts.fc2_input_quant.smooth_scale": fc2_input_smooth_scale,
        }
    )

    top_k_indices, top_k_gates = op.gating(hidden_states)
    sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = op.dispatch(
        hidden_states,
        top_k_gates,
        top_k_indices,
    )

    expanded_smooth_scale = smooth_scale.repeat_interleave(tokens_per_expert, dim=0)
    quant_input = sorted_hidden_states.float() * expanded_smooth_scale
    input_scale = quant_input.abs().amax(dim=-1).clamp(min=1e-12) / 127
    quantized_hidden_states = torch.clamp(
        torch.round(quant_input / input_scale.unsqueeze(-1)),
        -128,
        127,
    ).to(torch.int8)

    expert_outputs = _manual_quant_experts(
        quantized_hidden_states,
        input_scale,
        tokens_per_expert,
        up_weight,
        up_weight_scale,
        down_weight,
        down_weight_scale,
        fc2_input_smooth_scale,
        torch.bfloat16,
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
        "input_quant.smooth_scale",
        "experts.up_proj_weight",
        "experts.down_proj_weight",
        "experts.up_proj_weight_scale",
        "experts.down_proj_weight_scale",
        "experts.fc2_input_quant.smooth_scale",
    }
