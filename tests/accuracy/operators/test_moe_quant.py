from typing import Optional

import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset import MojoFusedSwiGLUMoEScaleDynamicQuantize
from mojo_opset import MojoGroupQuantGemmCombineMoE
from mojo_opset import MojoGroupQuantGemmMoE
from mojo_opset import MojoMoEInitRoutingDynamicQuant



def _manual_group_quant_gemm_moe(
    input: torch.Tensor,
    weight: torch.Tensor,
    token_count: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    *,
    trans_weight: bool,
) -> torch.Tensor:
    batch_size, top_k, hidden_dim = input.shape
    route_count = batch_size * top_k
    input_fp = input.float()

    if input_scale is not None:
        if input_scale.shape == input.shape[:-1]:
            input_fp = input_fp * input_scale.float().unsqueeze(-1)
        else:
            input_blocks = input_fp.reshape(batch_size, top_k, -1, hidden_dim // input_scale.shape[-1])
            input_fp = (input_blocks * input_scale.float().unsqueeze(-1)).reshape_as(input_fp)

    input_fp = input_fp.reshape(route_count, hidden_dim)
    outputs = []
    route_start = 0
    for expert_idx, expert_token_count in enumerate(token_count.to(dtype=torch.int64).tolist()):
        expert_input = input_fp[route_start : route_start + expert_token_count]
        expert_weight = weight[expert_idx].float()
        if trans_weight:
            expert_weight = expert_weight.transpose(0, 1).contiguous()
        expert_output = expert_input @ expert_weight
        expert_output = expert_output * weight_scale[expert_idx].float().unsqueeze(0)
        outputs.append(expert_output)
        route_start += expert_token_count
    return torch.cat(outputs, dim=0).reshape(batch_size, top_k, -1)


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

    assert quantized.shape == (2, 2, 8)
    assert quantized.dtype == torch.int8
    torch.testing.assert_close(sorted_gates, torch.tensor([[[0.1], [0.8]], [[0.9], [0.2]]]))
    torch.testing.assert_close(sorted_token_indices, torch.tensor([[[0], [1]], [[0], [1]]], dtype=torch.int32))
    torch.testing.assert_close(token_count, torch.tensor([2, 2], dtype=torch.int32))
    assert scale.shape == (2, 2, 1)
    assert scale.dtype == torch.float32


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


def test_group_quant_gemm_moe_reference():
    input = torch.tensor(
        [
            [[1, 0, -1, 2, 1, -2, 0, 1], [2, 1, 0, -1, -2, 1, 2, 0]],
            [[0, 1, 2, 1, -1, 0, 1, 2], [1, -1, 1, -1, 1, -1, 1, -1]],
        ],
        dtype=torch.int8,
    )
    weight = torch.randint(-3, 4, (2, 6, 8), dtype=torch.int8)
    token_count = torch.tensor([2, 2], dtype=torch.int32)
    weight_scale = torch.ones(2, 6, dtype=torch.float32)
    input_scale = torch.ones(2, 2, 1, dtype=torch.float32)

    op = MojoGroupQuantGemmMoE._registry.get("torch")(output_dtype=torch.float32, trans_weight=True)
    out = op(input, weight, token_count, weight_scale, input_scale)
    ref = _manual_group_quant_gemm_moe(
        input,
        weight,
        token_count,
        weight_scale,
        input_scale,
        trans_weight=True,
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_group_quant_gemm_combine_moe_reference():
    input = torch.tensor(
        [
            [[1, 0, -1, 2], [0, 1, 2, 1]],
            [[2, 1, 0, -1], [1, -1, 1, -1]],
        ],
        dtype=torch.int8,
    )
    weight = torch.randint(-2, 3, (2, 4, 3), dtype=torch.int8)
    token_count = torch.tensor([2, 2], dtype=torch.int32)
    top_k_gates = torch.tensor([[[0.2], [0.8]], [[0.6], [0.4]]], dtype=torch.float32)
    token_indices = torch.tensor([[[0], [1]], [[0], [1]]], dtype=torch.int32)
    shared_output = torch.zeros(2, 3, dtype=torch.float32)
    weight_scale = torch.ones(2, 3, dtype=torch.float32)
    input_scale = torch.ones(2, 2, dtype=torch.float32)

    op = MojoGroupQuantGemmCombineMoE._registry.get("torch")(output_dtype=torch.float32, trans_weight=False)
    out = op(input, weight, top_k_gates, token_indices, token_count, shared_output, weight_scale, input_scale)

    routed = _manual_group_quant_gemm_moe(
        input,
        weight,
        token_count,
        weight_scale,
        input_scale,
        trans_weight=False,
    )
    ref = shared_output.clone()
    ref.index_add_(
        0,
        token_indices.reshape(-1).to(dtype=torch.long),
        routed.reshape(-1, routed.shape[-1]) * top_k_gates.reshape(-1, 1),
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@bypass_not_implemented
def test_fused_swiglu_moe_scale_dynamic_quant_backend():
    input = torch.randn(4, 2, 128, dtype=torch.bfloat16)
    smooth_scale = torch.ones(4, 64, dtype=torch.float32)
    token_count = torch.tensor([2, 2, 2, 2], dtype=torch.int32)

    op = MojoFusedSwiGLUMoEScaleDynamicQuantize()
    quantized, scale = op(input, smooth_scale, token_count, 1.0, 0)
    assert quantized.shape == (4, 2, 64)
    assert quantized.dtype == torch.int8
    assert scale.shape == (4, 2)
    assert scale.dtype == torch.float32


@bypass_not_implemented
def test_group_quant_gemm_moe_backend():
    input = torch.randint(-128, 127, (4, 2, 64), dtype=torch.int8)
    weight = torch.randint(-128, 127, (4, 128, 64), dtype=torch.int8)
    token_count = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
    weight_scale = torch.ones(4, 128, dtype=torch.float32)
    input_scale = torch.ones(4, 2, 8, dtype=torch.float32)

    op = MojoGroupQuantGemmMoE(output_dtype=torch.bfloat16, trans_weight=True)
    op_ref = MojoGroupQuantGemmMoE._registry.get("torch")(output_dtype=torch.bfloat16, trans_weight=True)
    op.forward_diff_with(
        op_ref,
        input,
        weight,
        token_count,
        weight_scale,
        input_scale,
        atol=256,
        rtol=1e-2,
    )


@bypass_not_implemented
def test_group_quant_gemm_combine_moe_backend():
    input = torch.randint(-128, 127, (4, 2, 64), dtype=torch.int8)
    weight = torch.randint(-128, 127, (4, 64, 48), dtype=torch.int8)
    token_count = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
    top_k_gates = torch.rand(4, 2, 1, dtype=torch.float32)
    token_indices = torch.tensor(
        [[[0], [1]], [[2], [3]], [[0], [1]], [[2], [3]]],
        dtype=torch.int32,
    )
    shared_output = torch.zeros(4, 48, dtype=torch.bfloat16)
    weight_scale = torch.ones(4, 48, dtype=torch.float32)
    input_scale = torch.ones(4, 2, dtype=torch.float32)

    op = MojoGroupQuantGemmCombineMoE(output_dtype=torch.bfloat16, trans_weight=False)
    op_ref = MojoGroupQuantGemmCombineMoE._registry.get("torch")(output_dtype=torch.bfloat16, trans_weight=False)
    op.forward_diff_with(
        op_ref,
        input,
        weight,
        top_k_gates,
        token_indices,
        token_count,
        shared_output,
        weight_scale,
        input_scale,
        atol=32,
        rtol=5e-3,
    )


