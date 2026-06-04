import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from mojo_opset import MojoExperts
from mojo_opset import MojoGemm
from mojo_opset import MojoGroupRMSNorm
from mojo_opset import MojoApplyRoPE
from mojo_opset import MojoMoEGating
from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoRotaryEmbedding
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.platform import get_torch_device


def _require_npu_kernel_spy():
    if get_platform() != "npu":
        pytest.skip("torch_npu kernel spy tests require NPU.")


def _spy_success(monkeypatch, owner, name, calls):
    original = getattr(owner, name)

    def _wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        calls[name] = calls.get(name, 0) + 1
        return result

    monkeypatch.setattr(owner, name, _wrapped)


@auto_switch_platform()
@bypass_not_implemented
def test_gemm_npu_linear_kernel_called(monkeypatch):
    _require_npu_kernel_spy()
    device = get_torch_device()
    dtype = torch.bfloat16
    m, k, n = 16, 128, 64

    op = MojoGemm(k, n, bias=True, dtype=dtype).to(device)
    x = torch.randn((m, k), dtype=dtype, device=device)
    calls = {}
    _spy_success(monkeypatch, torch_npu, "npu_linear", calls)

    out = op(x)
    ref = F.linear(x, op.weight, op.bias)

    assert calls["npu_linear"] == 1
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=2e-2, rtol=2e-2)


@auto_switch_platform()
@bypass_not_implemented
def test_rmsnorm_npu_rms_norm_kernel_called(monkeypatch):
    _require_npu_kernel_spy()
    device = get_torch_device()
    dtype = torch.bfloat16
    shape = (16, 128)

    op = MojoRMSNorm(norm_size=shape[-1], eps=1e-5, dtype=dtype).to(device)
    op_ref = MojoRMSNorm._registry.get("torch")(norm_size=shape[-1], eps=1e-5).to(dtype).to(device)
    with torch.no_grad():
        op.weight.normal_(mean=1.0, std=0.02)
        op_ref.weight.copy_(op.weight)
    x = torch.randn(shape, dtype=dtype, device=device)
    calls = {}
    _spy_success(monkeypatch, torch_npu, "npu_rms_norm", calls)

    out = op(x)
    ref = op_ref(x)

    assert calls["npu_rms_norm"] == 1
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=3e-2, rtol=6e-3)


@auto_switch_platform()
@bypass_not_implemented
def test_group_rmsnorm_npu_rms_norm_kernel_called(monkeypatch):
    _require_npu_kernel_spy()
    device = get_torch_device()
    dtype = torch.bfloat16
    num_groups, hidden_size = 2, 128

    op = MojoGroupRMSNorm(num_groups=num_groups, norm_size=hidden_size, eps=1e-5, dtype=dtype).to(device)
    op_ref = MojoGroupRMSNorm._registry.get("torch")(
        num_groups=num_groups,
        norm_size=hidden_size,
        eps=1e-5,
    ).to(dtype).to(device)
    with torch.no_grad():
        op.weight.normal_(mean=1.0, std=0.02)
        op_ref.weight.copy_(op.weight)
    x_groups = [
        torch.randn((8, group_dim, hidden_size), dtype=dtype, device=device)
        for group_dim in (3, 5)
    ]
    calls = {}
    _spy_success(monkeypatch, torch_npu, "npu_rms_norm", calls)

    out = op(x_groups)
    ref = op_ref(x_groups)

    assert calls["npu_rms_norm"] == num_groups
    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual.to(torch.float32), expected.to(torch.float32), atol=3e-2, rtol=6e-3)


@auto_switch_platform()
@bypass_not_implemented
def test_apply_rope_npu_rotary_mul_kernel_called(monkeypatch):
    _require_npu_kernel_spy()
    device = get_torch_device()
    dtype = torch.bfloat16
    batch_size, seq_len, q_heads, k_heads, head_dim = 1, 16, 2, 1, 128

    rotary_embedding = MojoRotaryEmbedding._registry.get("torch")(
        rope_theta=10000.0,
        rope_dim=head_dim,
        init_max_length=seq_len,
    ).to(device)
    x = torch.randn((batch_size, seq_len, q_heads * head_dim), dtype=dtype, device=device)
    cos, sin = rotary_embedding(x)
    q = torch.randn((batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=device)

    op = MojoApplyRoPE()
    op_ref = MojoApplyRoPE._registry.get("torch")()
    calls = {}
    _spy_success(monkeypatch, torch_npu, "npu_rotary_mul", calls)

    q_out, k_out = op(q, k, cos, sin, False)
    q_ref, k_ref = op_ref(q, k, cos, sin, False)

    assert calls["npu_rotary_mul"] == 2
    torch.testing.assert_close(q_out.to(torch.float32), q_ref.to(torch.float32), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_out.to(torch.float32), k_ref.to(torch.float32), atol=5e-2, rtol=5e-2)


@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_gqa_fia_kernel_called(monkeypatch):
    _require_npu_kernel_spy()
    device = get_torch_device()
    dtype = torch.bfloat16
    q_len, num_q_heads, num_kv_heads, head_dim, block_size = 128, 2, 1, 128, 128

    query = torch.randn((q_len, num_q_heads, head_dim), dtype=dtype, device=device)
    k_cache = torch.randn((1, num_kv_heads, block_size, head_dim), dtype=dtype, device=device)
    v_cache = torch.randn_like(k_cache)
    cu_q_lens = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    op = MojoPagedPrefillGQA(is_causal=True)
    op_ref = MojoPagedPrefillGQA._registry.get("torch")(is_causal=True)
    calls = {}
    _spy_success(monkeypatch, torch_npu, "npu_fused_infer_attention_score", calls)

    out = op(query, k_cache, v_cache, cu_q_lens, block_tables, softmax_scale=softmax_scale)
    ref = op_ref(query, k_cache, v_cache, cu_q_lens, block_tables, softmax_scale=softmax_scale)

    assert calls["npu_fused_infer_attention_score"] == 1
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=2e-2, rtol=2e-2)


@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_gqa_fia_kernel_called(monkeypatch):
    _require_npu_kernel_spy()
    device = get_torch_device()
    dtype = torch.bfloat16
    batch_size, num_q_heads, num_kv_heads, head_dim, block_size = 1, 2, 1, 128, 128

    query = torch.randn((batch_size, num_q_heads, head_dim), dtype=dtype, device=device)
    k_cache = torch.randn((1, num_kv_heads, block_size, head_dim), dtype=dtype, device=device)
    v_cache = torch.randn_like(k_cache)
    total_seq_lens = torch.tensor([block_size], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    op = MojoPagedDecodeGQA(is_causal=True)
    op_ref = MojoPagedDecodeGQA._registry.get("torch")(is_causal=True)
    calls = {}
    _spy_success(monkeypatch, torch_npu, "npu_fused_infer_attention_score", calls)

    out = op(query, k_cache, v_cache, total_seq_lens, block_tables, softmax_scale=softmax_scale)
    ref = op_ref(query, k_cache, v_cache, total_seq_lens, block_tables, softmax_scale=softmax_scale)

    assert calls["npu_fused_infer_attention_score"] == 1
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=2e-2, rtol=2e-2)


@auto_switch_platform()
@bypass_not_implemented
def test_moe_gating_topk_softmax_kernel_called(monkeypatch):
    _require_npu_kernel_spy()
    device = get_torch_device()
    dtype = torch.bfloat16
    num_experts, top_k, hidden_size, num_tokens = 8, 2, 128, 16

    op = MojoMoEGating(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k).to(device)
    op_ref = MojoMoEGating._registry.get("torch")(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    with torch.no_grad():
        op.gate_weight.normal_(std=0.02)
        op_ref.gate_weight.copy_(op.gate_weight)
    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
    calls = {}
    _spy_success(monkeypatch, torch_npu, "npu_moe_gating_top_k_softmax", calls)

    indices, gates = op(x)
    indices_ref, gates_ref = op_ref(x)

    assert calls["npu_moe_gating_top_k_softmax"] == 1
    torch.testing.assert_close(indices.cpu(), indices_ref.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(gates.to(torch.float32), gates_ref.to(torch.float32), atol=2e-2, rtol=2e-2)


@auto_switch_platform()
@bypass_not_implemented
def test_experts_grouped_matmul_and_swiglu_kernels_called(monkeypatch):
    _require_npu_kernel_spy()
    device = get_torch_device()
    dtype = torch.bfloat16
    num_experts, hidden_size, intermediate_size = 2, 128, 256

    op = MojoExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    op_ref = MojoExperts._registry.get("torch")(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    for param in op.parameters():
        nn.init.normal_(param, std=0.02)
    op = op.to(dtype).to(device)
    op_ref = op_ref.to(dtype).to(device)
    op_ref.load_state_dict(op.state_dict())

    tokens_per_expert = torch.tensor([8, 8], dtype=torch.int32, device=device)
    x = torch.randn((int(tokens_per_expert.sum().item()), hidden_size), dtype=dtype, device=device)
    calls = {}
    _spy_success(monkeypatch, torch_npu, "npu_grouped_matmul", calls)
    _spy_success(monkeypatch, torch_npu, "npu_swiglu", calls)

    out = op(x, tokens_per_expert)
    ref = op_ref(x, tokens_per_expert)

    assert calls["npu_grouped_matmul"] == 2
    assert calls["npu_swiglu"] == 1
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=3e-2, rtol=3e-2)
