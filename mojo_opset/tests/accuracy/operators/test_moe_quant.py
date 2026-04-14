from typing import Optional

import pytest
import torch
import torch.nn as nn

from mojo_opset.tests.utils import bypass_not_implemented

from mojo_opset import MojoMoE
from mojo_opset import MojoFusedSwiGLUMoEScaleDynamicQuantize
from mojo_opset import MojoGroupQuantGemmCombineMoE
from mojo_opset import MojoGroupQuantGemmMoE
from mojo_opset import MojoMoEInitRoutingDynamicQuant
from mojo_opset.backends.ixformer.operators.moe import IxformerFusedSwiGLUMoEScaleDynamicQuantize
from mojo_opset.backends.ixformer.operators.moe import IxformerGroupQuantGemmCombineMoE
from mojo_opset.backends.ixformer.operators.moe import IxformerGroupQuantGemmMoE
from mojo_opset.backends.ixformer.operators.moe import IxformerMoEInitRoutingDynamicQuant
from mojo_opset.backends.ixformer.operators.moe import IxformerQuantMoe
from mojo_opset.utils.platform import get_torch_device



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
    top_k_indices = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
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
    weight = torch.randint(-3, 4, (2, 6, 8)).to(dtype=torch.int8)
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
    weight = torch.randint(-2, 3, (2, 4, 3)).to(dtype=torch.int8)
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
    input = torch.randint(-128, 127, (4, 2, 64)).to(dtype=torch.int8)
    weight = torch.randint(-128, 127, (4, 128, 64)).to(dtype=torch.int8)
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
    input = torch.randint(-128, 127, (4, 2, 64)).to(dtype=torch.int8)
    weight = torch.randint(-128, 127, (4, 64, 48)).to(dtype=torch.int8)
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

@pytest.mark.ci
@pytest.mark.parametrize("seqlen", [2 , 11, 16, 128, 256, 311, 1024, 1025, 3072, 3071, 8192, 16384])
@pytest.mark.parametrize("num_experts, hidden_size", [(128, 4096), (128, 5120)])
@pytest.mark.parametrize("top_k", [2, 4, 8])
@bypass_not_implemented
def test_moe_init_routing_dynamic_quant_backend(seqlen: int, num_experts: int, hidden_size: int, top_k: int):
    hidden_states = torch.randn(seqlen, hidden_size, dtype=torch.bfloat16)
    # top_k_gates = torch.softmax(torch.randn(seqlen, top_k, dtype=torch.float32), dim=-1)
    # top_k_indices = torch.stack([torch.randperm(4)[:2] for _ in range(8)])
    smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32)

    top_k_gates = torch.randn([seqlen, top_k], dtype=torch.float32)
    top_k_indices = torch.randint(0, num_experts, (seqlen, top_k,), dtype=torch.int32)
    quant_mode = 0

    op = MojoMoEInitRoutingDynamicQuant(num_experts=num_experts, top_k=top_k, quant_block_size=hidden_size)
    op_ref = MojoMoEInitRoutingDynamicQuant._registry.get("torch")(num_experts=num_experts, top_k=top_k, quant_block_size=hidden_size)
    op.forward_diff_with(
        op_ref,
        hidden_states,
        top_k_gates,
        top_k_indices,
        smooth_scale,
        quant_mode,
        atol=(1, 1e-4, 0, 0, 1e-4),
        rtol=(0, 1e-4, 0, 0, 1e-4),
    )


def generate_random_list(M, N):
    """
    生成一个长度为M，总和为N，所有元素>=0的随机列表
    使用均匀分布方法
    """
    points = torch.cat([torch.tensor([0, N]), torch.randint(0, N + 1, (M - 1,))])
    points, _ = torch.sort(points)
    result = (points[1:] - points[:-1]).tolist()

    return result

@pytest.mark.ci
@pytest.mark.parametrize("seq_len", [2, 64, 128, 1024, 4096])
@pytest.mark.parametrize("last_dim", [1280, 3584, 4096])
@pytest.mark.parametrize("EXPERT_NUM", [8, 32, 48, 64])
@pytest.mark.parametrize("TOPK", [2, 4, 8])
@bypass_not_implemented
def test_fused_swiglu_moe_scale_dynamic_quant_backend(seq_len, last_dim, EXPERT_NUM, TOPK):
    input = torch.randn(seq_len, TOPK, last_dim, dtype=torch.bfloat16)
    smooth_scale = torch.rand(EXPERT_NUM, last_dim//2, dtype=torch.float32)
    token_count = torch.tensor(generate_random_list(EXPERT_NUM, seq_len * TOPK), dtype=torch.int32)

    op = MojoFusedSwiGLUMoEScaleDynamicQuantize()
    op_ref = MojoFusedSwiGLUMoEScaleDynamicQuantize._registry.get("torch")()
    op.forward_diff_with(op_ref, input, smooth_scale, token_count, 1.0, 0, atol=(1, 1e-4), rtol=(0, 1e-4))

def _quantize_per_output_channel(weight_fp: torch.Tensor):
    """
    Quantize per output channel along the input-K axis.
    weight_fp: [E, O, I] float
    returns:
      q_weight: [E, O, I] int8
      scale:    [E, O] float32
      dequant:  [E, O, I] float32
    """
    scale = weight_fp.abs().amax(dim=-1).clamp(min=1e-6) / 127.0
    q_weight = torch.clamp(
        torch.round(weight_fp / scale.unsqueeze(-1)),
        -128,
        127,
    ).to(torch.int8)
    dequant = q_weight.float() * scale.unsqueeze(-1)
    return q_weight, scale.float(), dequant


@torch.no_grad()
def _mojo_moe_forward_with_fixed_routes(
    moe_ref: MojoMoE,
    hidden_states: torch.Tensor,
    top_k_indices: torch.Tensor,
    top_k_gates: torch.Tensor,
) -> torch.Tensor:
    sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = moe_ref.dispatch(
        hidden_states, top_k_gates, top_k_indices
    )
    expert_outputs = moe_ref.experts(sorted_hidden_states, tokens_per_expert)
    output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
    return moe_ref.combine(output_buffer, expert_outputs, sorted_gates, token_indices)


COMMON_IXFORMER_QUANT_MOE_CASES = [
    # (num_experts, top_k, hidden_size, intermediate_size, num_tokens)
    (8, 2, 256, 512, 64),
    (8, 4, 256, 512, 96),
    (16, 4, 512, 1024, 128),
    (8, 8, 512, 1024, 64),
]


@bypass_not_implemented
@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, intermediate_size, num_tokens",
    COMMON_IXFORMER_QUANT_MOE_CASES,
)
@torch.no_grad()
def test_ixformer_quant_moe_vs_mojo_moe_accuracy(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    num_tokens: int,
):
    device = get_torch_device()
    dtype = torch.bfloat16

    torch.manual_seed(0)

    # torch reference path
    mojo_ref = MojoMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    for p in mojo_ref.parameters():
        nn.init.normal_(p, std=0.02)
    mojo_ref = mojo_ref.to(dtype).to(device)
    mojo_ref.gating.gate_weight.data = mojo_ref.gating.gate_weight.data.float()

    # ixformer quant module under test
    ixf_moe = IxformerQuantMoe(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        output_dtype=dtype,
        routed_scaling_factor=1.0,
    ).to(device)

    # Align gating weights
    ixf_moe.gating.gate_weight.data.copy_(mojo_ref.gating.gate_weight.data)

    # Quantize the reference expert weights, then:
    # - use quantized weights in ixformer path
    # - use dequantized weights in torch reference path
    # so the diff mostly reflects implementation path rather than raw quant loss.
    w13_fp = mojo_ref.experts.up_proj_weight.data.float()
    w2_fp = mojo_ref.experts.down_proj_weight.data.float()

    w13_q, w13_s, w13_deq = _quantize_per_output_channel(w13_fp)
    w2_q, w2_s, w2_deq = _quantize_per_output_channel(w2_fp)

    ixf_moe.w13_weight.data.copy_(w13_q.to(device))
    ixf_moe.w13_weight_scale.data.copy_(w13_s.to(device))
    ixf_moe.w2_weight.data.copy_(w2_q.to(device))
    ixf_moe.w2_weight_scale.data.copy_(w2_s.to(device))

    mojo_ref.experts.up_proj_weight.data.copy_(
        w13_deq.to(device=device, dtype=mojo_ref.experts.up_proj_weight.dtype)
    )
    mojo_ref.experts.down_proj_weight.data.copy_(
        w2_deq.to(device=device, dtype=mojo_ref.experts.down_proj_weight.dtype)
    )

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # A) fixed-route compare: isolate expert + combine path
    fixed_top_k_indices, fixed_top_k_gates = mojo_ref.gating(x)
    out_ixf_fixed = ixf_moe(
        x,
        top_k_indices=fixed_top_k_indices,
        top_k_gates=fixed_top_k_gates,
    )
    out_ref_fixed = _mojo_moe_forward_with_fixed_routes(
        mojo_ref,
        x,
        fixed_top_k_indices,
        fixed_top_k_gates,
    )
    torch.testing.assert_close(
        out_ixf_fixed.float(),
        out_ref_fixed.float(),
        atol=0.20,
        rtol=0.10,
    )

    # B) end-to-end compare
    out_ixf_e2e = ixf_moe(x)
    out_ref_e2e = mojo_ref(x)
    torch.testing.assert_close(
        out_ixf_e2e.float(),
        out_ref_e2e.float(),
        atol=0.35,
        rtol=0.15,
    )


@pytest.mark.ci
@pytest.mark.parametrize("seq_len", [2, 64, 128, 1024, 4096])
@pytest.mark.parametrize("last_dim", [1280, 3584, 4096])
@pytest.mark.parametrize("EXPERT_NUM", [8, 32, 48, 64])
@pytest.mark.parametrize("TOPK", [2, 4, 8])
@bypass_not_implemented
def test_ixformer_fused_swiglu_moe_scale_dynamic_quant(seq_len, last_dim, EXPERT_NUM, TOPK):
    device = get_torch_device()
    input = torch.randn(seq_len, TOPK, last_dim, dtype=torch.bfloat16, device=device)
    smooth_scale = torch.rand(EXPERT_NUM, last_dim // 2, dtype=torch.float32, device=device)
    token_count = torch.tensor(
        generate_random_list(EXPERT_NUM, seq_len * TOPK),
        dtype=torch.int32,
        device=device,
    )

    op = IxformerFusedSwiGLUMoEScaleDynamicQuantize()
    op_ref = MojoFusedSwiGLUMoEScaleDynamicQuantize._registry.get("torch")()
    route_count = seq_len * TOPK
    sorted_token_ids = torch.arange(route_count, dtype=torch.int32, device=device)
    topk_indices = torch.repeat_interleave(
        torch.arange(EXPERT_NUM, dtype=torch.int32, device=device),
        token_count.to(dtype=torch.int64, device=device),
    )
    assert topk_indices.numel() == route_count

    quantized, scale = op(
        input=input,
        smooth_scale=smooth_scale,
        sorted_token_ids=sorted_token_ids,
        topk_indices=topk_indices,
        fc1_intermediate_size=last_dim,
        beta=1.0,
        quant_mode=0,
    )
    quantized = quantized.view(seq_len, TOPK, -1)
    scale = scale.view(seq_len, TOPK)

    quantized_ref, scale_ref = op_ref(input, smooth_scale, token_count, 1.0, 0)
    torch.testing.assert_close(quantized.float(), quantized_ref.float(), atol=1, rtol=0)
    torch.testing.assert_close(scale.float(), scale_ref.float(), atol=1e-4, rtol=1e-4)


@bypass_not_implemented
def test_ixformer_group_quant_gemm_moe():
    device = get_torch_device()
    input = torch.randint(-128, 127, (4, 2, 64), dtype=torch.int8, device=device)
    weight = torch.randint(-128, 127, (4, 128, 64), dtype=torch.int8, device=device)
    token_count = torch.tensor([2, 2, 2, 2], dtype=torch.int32, device="cpu")
    weight_scale = torch.ones(4, 128, dtype=torch.float32, device=device)
    input_scale = torch.ones(4, 2, 8, dtype=torch.float32, device=device)

    op = IxformerGroupQuantGemmMoE(output_dtype=torch.bfloat16, trans_weight=True)
    op_ref = MojoGroupQuantGemmMoE._registry.get("torch")(output_dtype=torch.bfloat16, trans_weight=True)

    input_ixf = input.reshape(-1, input.shape[-1]).contiguous()
    input_scale_ixf = torch.ones(input_ixf.shape[0], dtype=torch.float32, device=device)
    out = op(input_ixf, weight, token_count, weight_scale, input_scale_ixf)
    out_ref = op_ref(input, weight, token_count, weight_scale, input_scale)
    torch.testing.assert_close(out.view_as(out_ref).float(), out_ref.float(), atol=256, rtol=1e-2)


@bypass_not_implemented
def test_ixformer_group_quant_gemm_combine_moe():
    device = get_torch_device()
    input = torch.randint(-128, 127, (4, 2, 64), dtype=torch.int8, device=device)
    weight = torch.randint(-128, 127, (4, 64, 48), dtype=torch.int8, device=device)
    token_count = torch.tensor([2, 2, 2, 2], dtype=torch.int32, device="cpu")
    top_k_gates = torch.rand(4, 2, 1, dtype=torch.float32, device=device)
    token_indices = torch.tensor(
        [[[0], [1]], [[2], [3]], [[0], [1]], [[2], [3]]],
        dtype=torch.int32,
        device=device,
    )
    shared_output = torch.zeros(4, 48, dtype=torch.bfloat16, device=device)
    weight_scale = torch.ones(4, 48, dtype=torch.float32, device=device)
    input_scale = torch.ones(4, 2, dtype=torch.float32, device=device)

    op = IxformerGroupQuantGemmCombineMoE(output_dtype=torch.bfloat16, trans_weight=False, top_k=2)
    op_ref = MojoGroupQuantGemmCombineMoE._registry.get("torch")(output_dtype=torch.bfloat16, trans_weight=False)

    input_ixf = input.reshape(-1, input.shape[-1]).contiguous()
    top_k_gates_flat = top_k_gates.squeeze(-1).reshape(-1).contiguous()
    route_to_token = token_indices.reshape(-1).to(dtype=torch.long)
    grouped_routes = torch.argsort(route_to_token, stable=True)
    top_k_gates_ixf = top_k_gates_flat[grouped_routes].view(top_k_gates.shape[0], top_k_gates.shape[1]).contiguous()
    token_indices_ixf = torch.empty_like(grouped_routes, dtype=torch.int32)
    token_indices_ixf[grouped_routes] = torch.arange(grouped_routes.numel(), dtype=torch.int32, device=device)
    src_to_dst = torch.zeros_like(token_indices_ixf)
    input_scale_ixf = input_scale.reshape(-1).contiguous()

    weight_ixf = weight.transpose(-1, -2).contiguous()
    out = op(
        input=input_ixf,
        weight=weight_ixf,
        top_k_gates=top_k_gates_ixf,
        token_indices=token_indices_ixf,
        src_to_dst=src_to_dst,
        token_count=token_count,
        shared_output=shared_output,
        weight_scale=weight_scale,
        input_scale=input_scale_ixf,
    )
    out_ref = op_ref(
        input=input,
        weight=weight,
        top_k_gates=top_k_gates,
        token_indices=token_indices,
        token_count=token_count,
        shared_output=shared_output,
        weight_scale=weight_scale,
        input_scale=input_scale,
    )
    # Ixformer combine path performs bf16 accumulation in kernel, while torch
    # reference keeps more float32 accumulation steps before final cast.
    # With full-range int8 inputs this introduces stable quantization steps
    # (typically 256/512, occasionally 1024), so use a realistic absolute tol.
    torch.testing.assert_close(out.float(), out_ref.float(), atol=1024, rtol=5e-3)


@pytest.mark.ci
@pytest.mark.parametrize("seqlen", [2 , 11, 16, 128, 256, 311, 1024, 1025, 3072, 3071, 8192, 16384])
@pytest.mark.parametrize("num_experts, hidden_size", [(128, 4096), (128, 5120)])
@pytest.mark.parametrize("top_k", [2, 4, 8])
@bypass_not_implemented
def test_ixformer_moe_init_routing_dynamic_quant(
    seqlen: int,
    num_experts: int,
    hidden_size: int,
    top_k: int,
):
    # Torch reference path allocates several large route buffers in float32.
    # Skip extreme cases that are prone to OOM and do not add precision signal.
    if seqlen * top_k * hidden_size > 220_000_000:
        pytest.skip("Case too large for stable reference memory usage.")

    device = get_torch_device()
    hidden_states = torch.randn(seqlen, hidden_size, dtype=torch.bfloat16, device=device)
    smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device=device)

    top_k_gates = torch.randn([seqlen, top_k], dtype=torch.float32, device=device)
    top_k_indices = torch.randint(0, num_experts, (seqlen, top_k), dtype=torch.int32, device=device)
    quant_mode = 0

    op = IxformerMoEInitRoutingDynamicQuant(num_experts=num_experts, top_k=top_k, quant_block_size=hidden_size)
    op_ref = MojoMoEInitRoutingDynamicQuant._registry.get("torch")(num_experts=num_experts, top_k=top_k, quant_block_size=hidden_size)

    out = op(
        hidden_states=hidden_states.unsqueeze(0),
        top_k_gates=top_k_gates,
        top_k_indices=top_k_indices,
        smooth_scale=smooth_scale,
        quant_mode=quant_mode,
    )
    out_ref = op_ref(
        hidden_states=hidden_states,
        top_k_gates=top_k_gates,
        top_k_indices=top_k_indices,
        smooth_scale=smooth_scale,
        quant_mode=quant_mode,
    )

    i8_hidden_states, _, sorted_token_ids, _, expert_sizes_cpu, quant_scale = out
    i8_ref, _, _sorted_token_indices_ref, token_count_ref, scale_ref = out_ref

    # Align both outputs to source-route order before comparing.
    # Ixformer returns dst_to_src directly, while torch reference is ordered by
    # stable sort over flat top_k_indices.
    flat_experts = top_k_indices.reshape(-1).to(torch.int64)
    _, sort_indices = flat_experts.sort(stable=True)

    i8_ixf_flat = i8_hidden_states.reshape(-1, hidden_size)
    i8_ref_flat = i8_ref.reshape(-1, hidden_size)

    i8_ixf_src = torch.empty_like(i8_ixf_flat)
    i8_ixf_src[sorted_token_ids.to(torch.long)] = i8_ixf_flat
    i8_ref_src = torch.empty_like(i8_ref_flat)
    i8_ref_src[sort_indices.to(torch.long)] = i8_ref_flat

    torch.testing.assert_close(i8_ixf_src.float(), i8_ref_src.float(), atol=2, rtol=0)
    torch.testing.assert_close(expert_sizes_cpu.to(torch.int32), token_count_ref.to(torch.int32).cpu(), atol=0, rtol=0)

    scale_ixf_flat = quant_scale.reshape(-1)
    scale_ref_flat = scale_ref.reshape(-1)
    scale_ixf_src = torch.empty_like(scale_ixf_flat)
    scale_ixf_src[sorted_token_ids.to(torch.long)] = scale_ixf_flat
    scale_ref_src = torch.empty_like(scale_ref_flat)
    scale_ref_src[sort_indices.to(torch.long)] = scale_ref_flat

    torch.testing.assert_close(
        scale_ixf_src.float(),
        scale_ref_src.float(),
        atol=3e-4,
        rtol=1e-3,
    )