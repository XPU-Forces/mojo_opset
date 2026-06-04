import math

import torch
import torch.nn as nn
import torch_npu

from mojo_opset import MojoApplyRoPE
from mojo_opset import MojoExperts
from mojo_opset import MojoGemm
from mojo_opset import MojoGroupRMSNorm
from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoRMSNorm
from mojo_opset.tests.utils import assert_close
from mojo_opset.utils.platform import get_torch_device


def _unsupported_kernel_counter():
    calls = {"count": 0}

    def _raise_unsupported(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        raise RuntimeError("mock mxfp8 is not supported")

    return calls, _raise_unsupported


def test_gemm_fallback_when_npu_kernel_fails(monkeypatch):
    device = get_torch_device()
    dtype = torch.bfloat16
    x = torch.randn(4, 128, device=device, dtype=dtype)

    op = MojoGemm(128, 64, bias=True, dtype=dtype).to(device)
    ref = MojoGemm._registry.get("torch")(128, 64, bias=True, dtype=dtype).to(device)
    ref.load_state_dict(op.state_dict())

    calls, fail_kernel = _unsupported_kernel_counter()
    monkeypatch.setattr(torch_npu, "npu_linear", fail_kernel)

    assert_close(op(x), ref(x))
    assert calls["count"] == 1


def test_rmsnorm_fallback_when_npu_kernel_fails(monkeypatch):
    device = get_torch_device()
    dtype = torch.bfloat16
    x = torch.randn(4, 128, device=device, dtype=dtype)

    op = MojoRMSNorm(norm_size=x.shape[-1], eps=1e-5, device=device, dtype=dtype)
    ref = MojoRMSNorm._registry.get("torch")(norm_size=x.shape[-1], eps=1e-5).to(device).to(dtype)
    ref.load_state_dict(op.state_dict())

    calls, fail_kernel = _unsupported_kernel_counter()
    monkeypatch.setattr(torch_npu, "npu_rms_norm", fail_kernel)

    assert_close(op(x), ref(x))
    assert calls["count"] == 1


def test_group_rmsnorm_fallback_when_npu_kernel_fails(monkeypatch):
    device = get_torch_device()
    dtype = torch.bfloat16
    group_dims = (2, 3, 1)
    hidden_size = 128
    x = torch.randn(4, sum(group_dims), hidden_size, device=device, dtype=dtype)
    x_groups = list(torch.split(x, group_dims, dim=1))

    op = MojoGroupRMSNorm(
        num_groups=len(group_dims),
        norm_size=hidden_size,
        eps=1e-5,
        device=device,
        dtype=dtype,
    )
    ref = (
        MojoGroupRMSNorm._registry.get("torch")(
            num_groups=len(group_dims),
            norm_size=hidden_size,
            eps=1e-5,
        )
        .to(device)
        .to(dtype)
    )
    ref.load_state_dict(op.state_dict())
    with torch.no_grad():
        op.weight.normal_(mean=1.0, std=0.02)
        ref.weight.copy_(op.weight)

    calls, fail_kernel = _unsupported_kernel_counter()
    monkeypatch.setattr(torch_npu, "npu_rms_norm", fail_kernel)

    assert_close(tuple(op(x_groups)), tuple(ref(x_groups)))
    assert calls["count"] == 1


def test_apply_rope_fallback_when_npu_kernel_fails(monkeypatch):
    device = get_torch_device()
    dtype = torch.bfloat16
    batch_size, seq_len, q_heads, kv_heads, head_dim = 2, 4, 4, 2, 128
    q = torch.randn(batch_size, seq_len, q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, kv_heads, head_dim, device=device, dtype=dtype)
    cos = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(seq_len, head_dim, device=device, dtype=dtype)

    op = MojoApplyRoPE()
    ref = MojoApplyRoPE._registry.get("torch")()

    calls, fail_kernel = _unsupported_kernel_counter()
    monkeypatch.setattr(torch_npu, "npu_rotary_mul", fail_kernel)

    assert_close(op(q, k, cos, sin, head_first=False), ref(q, k, cos, sin, head_first=False))
    assert calls["count"] == 1


def test_experts_fallback_when_npu_kernel_fails(monkeypatch):
    device = get_torch_device()
    dtype = torch.bfloat16
    num_experts, hidden_size, intermediate_size = 4, 128, 256
    tokens_per_expert = torch.tensor([2, 1, 0, 3], dtype=torch.int32, device=device)
    x = torch.randn(int(tokens_per_expert.sum().item()), hidden_size, device=device, dtype=dtype)

    op = MojoExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device).to(dtype)
    ref = MojoExperts._registry.get("torch")(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device).to(dtype)
    for p in op.parameters():
        nn.init.normal_(p, std=0.02)
    ref.load_state_dict(op.state_dict())

    calls, fail_kernel = _unsupported_kernel_counter()
    monkeypatch.setattr(torch_npu, "npu_grouped_matmul", fail_kernel)

    assert_close(op(x, tokens_per_expert), ref(x, tokens_per_expert))
    assert calls["count"] == 1


def test_experts_fallback_when_npu_swiglu_fails(monkeypatch):
    device = get_torch_device()
    dtype = torch.bfloat16
    tokens_per_expert = torch.tensor([3], dtype=torch.int32, device=device)
    x = torch.randn(int(tokens_per_expert.sum().item()), 128, device=device, dtype=dtype)

    op = MojoExperts(num_experts=1, hidden_size=128, intermediate_size=256).to(device).to(dtype)
    ref = MojoExperts._registry.get("torch")(num_experts=1, hidden_size=128, intermediate_size=256).to(device).to(dtype)
    for p in op.parameters():
        nn.init.normal_(p, std=0.02)
    ref.load_state_dict(op.state_dict())

    calls, fail_kernel = _unsupported_kernel_counter()
    monkeypatch.setattr(torch_npu, "npu_swiglu", fail_kernel)

    assert_close(op(x, tokens_per_expert), ref(x, tokens_per_expert))
    assert calls["count"] == 1


def _paged_prefill_data(device, dtype):
    q_lens = torch.tensor([2, 3], device=device, dtype=torch.int32)
    cu_q_lens = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(q_lens, 0, dtype=torch.int32),
        ]
    )
    total_tokens = int(cu_q_lens[-1].item())
    q_heads, kv_heads, head_dim, block_size = 4, 2, 128, 128

    query = torch.randn(total_tokens, q_heads, head_dim, device=device, dtype=dtype)
    key_cache = torch.zeros(2, kv_heads, block_size, head_dim, device=device, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    block_tables = torch.tensor([[0], [1]], device=device, dtype=torch.int32)

    for batch_id, q_len in enumerate(q_lens.tolist()):
        key_cache[batch_id, :, :q_len, :] = torch.randn(kv_heads, q_len, head_dim, device=device, dtype=dtype)
        value_cache[batch_id, :, :q_len, :] = torch.randn(kv_heads, q_len, head_dim, device=device, dtype=dtype)

    return query, key_cache, value_cache, cu_q_lens, block_tables


def test_paged_prefill_gqa_fallback_when_npu_kernel_fails(monkeypatch):
    device = get_torch_device()
    dtype = torch.bfloat16
    query, key_cache, value_cache, cu_q_lens, block_tables = _paged_prefill_data(device, dtype)
    softmax_scale = 1.0 / math.sqrt(query.shape[-1])

    op = MojoPagedPrefillGQA(is_causal=True, gqa_layout="AABB")
    ref = MojoPagedPrefillGQA._registry.get("torch")(is_causal=True, gqa_layout="AABB")

    calls, fail_kernel = _unsupported_kernel_counter()
    monkeypatch.setattr(torch_npu, "npu_fused_infer_attention_score", fail_kernel)

    assert_close(
        op(query, key_cache, value_cache, cu_q_lens, block_tables, softmax_scale=softmax_scale),
        ref(query, key_cache, value_cache, cu_q_lens, block_tables, softmax_scale=softmax_scale),
    )
    assert calls["count"] == 1


def _paged_decode_data(device, dtype):
    batch_size, q_heads, kv_heads, head_dim, block_size = 2, 4, 2, 128, 128
    query = torch.randn(batch_size, q_heads, head_dim, device=device, dtype=dtype)
    key_cache = torch.zeros(batch_size, kv_heads, block_size, head_dim, device=device, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    total_seq_lens = torch.tensor([3, 4], device=device, dtype=torch.int32)
    block_tables = torch.tensor([[0], [1]], device=device, dtype=torch.int32)

    for batch_id, seq_len in enumerate(total_seq_lens.tolist()):
        key_cache[batch_id, :, :seq_len, :] = torch.randn(kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        value_cache[batch_id, :, :seq_len, :] = torch.randn(kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    return query, key_cache, value_cache, total_seq_lens, block_tables


def test_paged_decode_gqa_fallback_when_npu_kernel_fails(monkeypatch):
    device = get_torch_device()
    dtype = torch.bfloat16
    query, key_cache, value_cache, total_seq_lens, block_tables = _paged_decode_data(device, dtype)
    softmax_scale = 1.0 / math.sqrt(query.shape[-1])

    op = MojoPagedDecodeGQA(is_causal=True, gqa_layout="AABB")
    ref = MojoPagedDecodeGQA._registry.get("torch")(is_causal=True, gqa_layout="AABB")

    calls, fail_kernel = _unsupported_kernel_counter()
    monkeypatch.setattr(torch_npu, "npu_fused_infer_attention_score", fail_kernel)

    assert_close(
        op(
            query,
            key_cache,
            value_cache,
            total_seq_lens,
            block_tables,
            softmax_scale=softmax_scale,
            max_total_seq_len=int(total_seq_lens.max().item()),
        ),
        ref(
            query,
            key_cache,
            value_cache,
            total_seq_lens,
            block_tables,
            softmax_scale=softmax_scale,
            max_total_seq_len=int(total_seq_lens.max().item()),
        ),
    )
    assert calls["count"] == 1
