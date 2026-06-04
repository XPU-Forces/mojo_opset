import math

import pytest
import torch
import torch.nn as nn
import torch_npu

from mojo_opset import MojoExperts
from mojo_opset import MojoGemm
from mojo_opset.backends.torch_npu.operators.mxfp8 import mx_dequant_weight
from mojo_opset.backends.torch_npu.operators.mxfp8 import prepare_mx_expert_scale_for_grouped_matmul
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.platform import get_torch_device


def _require_npu_mxfp8():
    if get_platform() != "npu":
        pytest.skip("MXFP8 torch_npu API cases require NPU.")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("Current torch build does not expose torch.float8_e4m3fn.")
    if not hasattr(torch_npu, "float8_e8m0fnu"):
        pytest.skip("Current torch_npu build does not expose torch_npu.float8_e8m0fnu.")


def _ones_e8m0(shape, device):
    # E8M0 uses exponent-only encoding; 127 represents scale 1.0.
    return torch.full(shape, 127, dtype=torch.uint8, device=device)


def _assert_close_to_reference(actual, expected, *, atol=0.15, rtol=0.15):
    torch.testing.assert_close(
        actual.detach().cpu().to(torch.float32),
        expected.detach().cpu().to(torch.float32),
        atol=atol,
        rtol=rtol,
    )


def _error_stats(y: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    y32 = y.float()
    ref32 = ref.float()
    diff = (y32 - ref32).abs()
    rmse = torch.mean((y32 - ref32) ** 2).sqrt()
    ref_rms = torch.mean(ref32 ** 2).sqrt().clamp(min=1e-6)
    cos = torch.nn.functional.cosine_similarity(
        y32.reshape(-1),
        ref32.reshape(-1),
        dim=0,
        eps=1e-8,
    )
    return {
        "p99_abs": float(torch.quantile(diff.reshape(-1), 0.99).item()),
        "nrms": float((rmse / ref_rms).item()),
        "cosine": float(cos.item()),
    }


def _normalize_mx_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() >= 3 and scale.shape[-2] == 1:
        return scale.squeeze(-2)
    if scale.dim() >= 3 and scale.shape[-1] == 2 and scale.shape[-2] != 1:
        return scale.reshape(*scale.shape[:-2], -1)
    return scale


def _moe_golden_torch(
    x_sorted: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    counts = tokens_per_expert.reshape(-1).to("cpu", dtype=torch.int64)
    outs = []
    row = 0
    for expert_idx, count in enumerate(counts.tolist()):
        if count <= 0:
            continue
        x_e = x_sorted[row : row + count].float()
        row += count
        gate_up = torch.nn.functional.linear(x_e, up_weight[expert_idx].float())
        gate, up = gate_up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        outs.append(torch.nn.functional.linear(activated, down_weight[expert_idx].float()))
    if not outs:
        return x_sorted
    return torch.cat(outs, dim=0).to(dtype=x_sorted.dtype)


def test_prepare_mx_expert_scale_pads_odd_block_count():
    scale = torch.full((128, 1536, 15), 127, dtype=torch.uint8)
    prepared = prepare_mx_expert_scale_for_grouped_matmul(scale, in_features=512)

    assert prepared.shape == (128, 8, 1536, 2)
    assert torch.all(prepared[:, :-1, :, :] == 127)
    assert torch.all(prepared[:, -1, :, 0] == 127)
    assert torch.all(prepared[:, -1, :, 1] == 0)


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_quant_matmul_function_and_accuracy():
    _require_npu_mxfp8()
    device = get_torch_device()
    m, k, n = 128, 128, 64

    x1_fp32 = torch.randint(-2, 3, (m, k), dtype=torch.int8).to(torch.float32)
    x2_fp32 = torch.randint(-2, 3, (k, n), dtype=torch.int8).to(torch.float32)
    x1 = x1_fp32.to(torch.float16).to(torch.float8_e4m3fn).to(device)
    x2 = x2_fp32.to(torch.float16).to(torch.float8_e4m3fn).to(device)
    scale = _ones_e8m0((math.ceil(k / 64), n, 2), device)
    pertoken_scale = _ones_e8m0((m, math.ceil(k / 64), 2), device)

    out = torch_npu.npu_quant_matmul(
        x1,
        x2,
        scale,
        pertoken_scale=pertoken_scale,
        output_dtype=torch.bfloat16,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32],
    )
    ref = (x1_fp32 @ x2_fp32).to(torch.bfloat16)

    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16
    _assert_close_to_reference(out, ref, atol=0.5, rtol=0.05)


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_mojo_gemm_uses_quant_matmul(monkeypatch):
    _require_npu_mxfp8()
    device = get_torch_device()
    torch.manual_seed(0)
    m, k, n = 64, 256, 128

    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    weight = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    weight_fp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
        weight,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        scale_alg=0,
        dst_type_max=0.0,
    )
    weight_scale = _normalize_mx_scale(weight_scale)

    op = MojoGemm(k, n, bias=False, dtype=torch.bfloat16).to(device)
    op.weight = nn.Parameter(weight_fp8)
    op.register_buffer("per_group_scales", weight_scale)

    original = torch_npu.npu_quant_matmul
    calls = {"count": 0}

    def _spy_quant_matmul(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_quant_matmul", _spy_quant_matmul)

    y = op(x)
    ref = (x.float() @ weight.float().t()).to(torch.bfloat16)
    stats = _error_stats(y, ref)
    assert calls["count"] >= 1
    assert stats["cosine"] > 0.98, stats
    assert stats["nrms"] < 0.12, stats


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_mojo_gemm_applies_input_smooth_inv(monkeypatch):
    _require_npu_mxfp8()
    device = get_torch_device()
    torch.manual_seed(4)
    m, k, n = 64, 256, 128

    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    base_weight = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    smooth = torch.linspace(0.5, 1.5, k, dtype=torch.bfloat16, device=device)
    folded_weight = (base_weight * smooth.reshape(1, -1)).contiguous()
    weight_fp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
        folded_weight,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        scale_alg=0,
        dst_type_max=0.0,
    )
    weight_scale = _normalize_mx_scale(weight_scale)

    op = MojoGemm(k, n, bias=False, dtype=torch.bfloat16).to(device)
    op.weight = nn.Parameter(weight_fp8)
    op.register_buffer("per_group_scales", weight_scale)
    op.register_buffer("input_smooth_inv", (1.0 / smooth.float()).to(torch.bfloat16))

    original = torch_npu.npu_quant_matmul
    calls = {"count": 0}

    def _spy_quant_matmul(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_quant_matmul", _spy_quant_matmul)

    y = op(x)
    ref = (x.float() @ base_weight.float().t()).to(torch.bfloat16)
    stats = _error_stats(y, ref)
    assert calls["count"] >= 1
    assert stats["cosine"] > 0.98, stats
    assert stats["nrms"] < 0.12, stats


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_mojo_gemm_dequant_fallback_when_quant_matmul_fails(monkeypatch):
    _require_npu_mxfp8()
    device = get_torch_device()
    torch.manual_seed(1)
    m, k, n = 16, 128, 64

    x = (torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    weight = (torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    weight_fp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
        weight,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        scale_alg=0,
        dst_type_max=0.0,
    )
    weight_scale = _normalize_mx_scale(weight_scale)

    op = MojoGemm(k, n, bias=False, dtype=torch.bfloat16).to(device)
    op.weight = nn.Parameter(weight_fp8)
    op.register_buffer("per_group_scales", weight_scale)
    calls = {"count": 0}

    def _fail_quant_matmul(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        raise RuntimeError("mock MXFP8 quant_matmul unsupported")

    monkeypatch.setattr(torch_npu, "npu_quant_matmul", _fail_quant_matmul)

    y = op(x)
    ref_weight = mx_dequant_weight(weight_fp8, weight_scale, out_dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, ref_weight)
    stats = _error_stats(y, ref)
    assert calls["count"] >= 1
    assert stats["cosine"] > 0.98, stats
    assert stats["nrms"] < 0.12, stats


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_grouped_matmul_function_and_accuracy():
    _require_npu_mxfp8()
    device = get_torch_device()
    num_groups, m, k, n = 2, 128, 128, 64
    group_rows = [64, 64]

    x_fp32 = torch.randint(-2, 3, (m, k), dtype=torch.int8).to(torch.float32)
    weight_fp32 = torch.randint(-2, 3, (num_groups, n, k), dtype=torch.int8).to(torch.float32)
    x = x_fp32.to(torch.float16).to(torch.float8_e4m3fn).to(device)
    weight = weight_fp32.to(torch.float16).to(torch.float8_e4m3fn).to(device).transpose(1, 2)
    scale = _ones_e8m0((num_groups, n, math.ceil(k / 64), 2), device).transpose(1, 2)
    per_token_scale = _ones_e8m0((m, math.ceil(k / 64), 2), device)
    group_list = torch.tensor([sum(group_rows[: idx + 1]) for idx in range(num_groups)], dtype=torch.int64, device=device)

    out = torch_npu.npu_grouped_matmul(
        [x],
        [weight],
        scale=[scale],
        per_token_scale=[per_token_scale],
        group_list=group_list,
        split_item=2,
        group_type=0,
        output_dtype=torch.bfloat16,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        group_list_type=0,
    )[0]

    ref_chunks = []
    start = 0
    for group_idx, rows in enumerate(group_rows):
        end = start + rows
        ref_chunks.append(x_fp32[start:end] @ weight_fp32[group_idx].transpose(0, 1))
        start = end
    ref = torch.cat(ref_chunks, dim=0).to(torch.bfloat16)

    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16
    _assert_close_to_reference(out, ref, atol=0.5, rtol=0.05)


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_mojo_experts_use_grouped_matmul(monkeypatch):
    _require_npu_mxfp8()
    device = get_torch_device()
    torch.manual_seed(2)
    num_experts, hidden, inter = 2, 128, 64
    tokens_per_expert = torch.tensor([16, 16], dtype=torch.int32, device=device)
    x = (torch.randn(int(tokens_per_expert.sum().item()), hidden, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    up = (torch.randn(num_experts, inter * 2, hidden, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    down = (torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    up_fp8, up_scale = torch_npu.npu_dynamic_mx_quant(
        up,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        scale_alg=0,
        dst_type_max=0.0,
    )
    down_fp8, down_scale = torch_npu.npu_dynamic_mx_quant(
        down,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        scale_alg=0,
        dst_type_max=0.0,
    )
    up_scale = _normalize_mx_scale(up_scale)
    down_scale = _normalize_mx_scale(down_scale)

    op = MojoExperts(num_experts=num_experts, hidden_size=hidden, intermediate_size=inter).to(device)
    op.up_proj_weight = nn.Parameter(up_fp8)
    op.down_proj_weight = nn.Parameter(down_fp8)
    op.register_buffer("up_proj_weight_mx_scale", up_scale)
    op.register_buffer("down_proj_weight_mx_scale", down_scale)

    original_gmm = torch_npu.npu_grouped_matmul
    calls = {"count": 0}

    def _spy_grouped_matmul(*args, **kwargs):
        calls["count"] += 1
        return original_gmm(*args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_grouped_matmul", _spy_grouped_matmul)

    y = op(x, tokens_per_expert)
    ref = _moe_golden_torch(x, tokens_per_expert, up, down)
    stats = _error_stats(y, ref)
    assert calls["count"] == 2
    assert stats["cosine"] > 0.97, stats
    assert stats["nrms"] < 0.15, stats


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_mojo_experts_dequant_fallback_when_grouped_matmul_fails(monkeypatch):
    _require_npu_mxfp8()
    device = get_torch_device()
    torch.manual_seed(3)
    num_experts, hidden, inter = 2, 64, 32
    tokens_per_expert = torch.tensor([4, 4], dtype=torch.int32, device=device)
    x = (torch.randn(int(tokens_per_expert.sum().item()), hidden, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    up = (torch.randn(num_experts, inter * 2, hidden, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    down = (torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    up_fp8, up_scale = torch_npu.npu_dynamic_mx_quant(
        up,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        scale_alg=0,
        dst_type_max=0.0,
    )
    down_fp8, down_scale = torch_npu.npu_dynamic_mx_quant(
        down,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        scale_alg=0,
        dst_type_max=0.0,
    )
    up_scale = _normalize_mx_scale(up_scale)
    down_scale = _normalize_mx_scale(down_scale)

    op = MojoExperts(num_experts=num_experts, hidden_size=hidden, intermediate_size=inter).to(device)
    op.up_proj_weight = nn.Parameter(up_fp8)
    op.down_proj_weight = nn.Parameter(down_fp8)
    op.register_buffer("up_proj_weight_mx_scale", up_scale)
    op.register_buffer("down_proj_weight_mx_scale", down_scale)
    calls = {"count": 0}

    def _fail_grouped_matmul(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        raise RuntimeError("mock MXFP8 grouped_matmul unsupported")

    monkeypatch.setattr(torch_npu, "npu_grouped_matmul", _fail_grouped_matmul)

    y = op(x, tokens_per_expert)
    up_deq = mx_dequant_weight(up_fp8, up_scale, out_dtype=torch.bfloat16)
    down_deq = mx_dequant_weight(down_fp8, down_scale, out_dtype=torch.bfloat16)
    ref = _moe_golden_torch(x, tokens_per_expert, up_deq, down_deq)
    stats = _error_stats(y, ref)
    assert calls["count"] >= 1
    assert stats["cosine"] > 0.98, stats
    assert stats["nrms"] < 0.12, stats


def _mxfp8_paged_attention_case(query_len, kv_len):
    _require_npu_mxfp8()
    device = get_torch_device()
    num_q_heads, num_kv_heads, head_dim = 2, 1, 128
    block_size, num_blocks = 512, 1

    query_fp32 = torch.randint(-2, 3, (query_len, num_q_heads, head_dim), dtype=torch.int8).to(torch.float32)
    key_fp32 = torch.zeros((num_blocks, num_kv_heads, block_size, head_dim), dtype=torch.float32)
    value_fp32 = torch.zeros_like(key_fp32)
    key_fp32[:, :, :kv_len, :] = torch.randint(
        -2,
        3,
        (num_blocks, num_kv_heads, kv_len, head_dim),
        dtype=torch.int8,
    ).to(torch.float32)
    value_fp32[:, :, :kv_len, :] = torch.randint(
        -2,
        3,
        (num_blocks, num_kv_heads, kv_len, head_dim),
        dtype=torch.int8,
    ).to(torch.float32)

    query = query_fp32.to(torch.float16).to(torch.float8_e4m3fn).to(device)
    key = key_fp32.to(torch.float16).to(torch.float8_e4m3fn).to(device)
    value = value_fp32.to(torch.float16).to(torch.float8_e4m3fn).to(device)
    q_scale = _ones_e8m0((query_len, num_q_heads, head_dim // 64, 2), device)
    k_scale = _ones_e8m0((num_blocks, num_kv_heads, block_size, head_dim // 64, 2), device)
    v_scale = _ones_e8m0((num_blocks, num_kv_heads, block_size // 64, head_dim, 2), device)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        key,
        value,
        block_table=block_table,
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        actual_seq_qlen=[query_len],
        actual_seq_kvlen=[kv_len],
        num_query_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        softmax_scale=softmax_scale,
        input_layout="TND",
        block_size=block_size,
        query_quant_mode=6,
        key_quant_mode=6,
        value_quant_mode=8,
        dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
        dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
        dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
        out_dtype=torch.bfloat16,
    )

    key_active = key_fp32[0, :, :kv_len, :].repeat_interleave(num_q_heads // num_kv_heads, dim=0)
    value_active = value_fp32[0, :, :kv_len, :].repeat_interleave(num_q_heads // num_kv_heads, dim=0)
    scores = torch.einsum("tnd,nsd->tns", query_fp32, key_active) * softmax_scale
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("tns,nsd->tnd", probs, value_active).to(torch.bfloat16)

    assert out.shape == (query_len, num_q_heads, head_dim)
    assert out.dtype == torch.bfloat16
    _assert_close_to_reference(out, ref, atol=0.25, rtol=0.2)


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_paged_prefill_gqa_fia_v2_function_and_accuracy():
    # MXFP8 fullquant requires actualSeqLengthsQ to be a multiple of 64.
    _mxfp8_paged_attention_case(query_len=64, kv_len=64)


@auto_switch_platform()
@bypass_not_implemented
def test_mxfp8_paged_decode_gqa_fia_v2_function_and_accuracy():
    # Single-token/small-KV decode is rejected by the v2 MXFP8 fullquant tiler;
    # validate the PageAttention/GQA route with the minimum legal Q/KV block.
    _mxfp8_paged_attention_case(query_len=64, kv_len=64)
