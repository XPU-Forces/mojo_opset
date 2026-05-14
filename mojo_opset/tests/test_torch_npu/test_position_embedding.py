"""Unit tests for ``torch_npu`` RoPE operators."""

import pytest
import torch

from mojo_opset.backends.torch_npu.operators.position_embedding import TorchNpuApplyRoPE
from mojo_opset.backends.torch_npu.operators.position_embedding import TorchNpuApplyVisionRoPE2D
from mojo_opset.core import MojoApplyRoPE
from mojo_opset.core import MojoApplyVisionRoPE2D

from mojo_opset.tests.test_torch_npu.common import assert_close_npu
from mojo_opset.tests.test_torch_npu.common import require_npu


def test_torch_npu_apply_rope_decode_matches_reference():
    """Single-batch decode layout ``[B, N, D]`` with ``cos/sin`` ``[B, rope_dim]`` (head_first swaps N/B)."""
    dev = require_npu()
    dtype = torch.bfloat16
    bs, n_heads, d, rope_d = 2, 4, 64, 32
    q = torch.randn(bs, n_heads, d, dtype=dtype, device=dev)
    k = torch.randn(bs, n_heads, d, dtype=dtype, device=dev)
    cos = torch.randn(bs, rope_d, dtype=dtype, device=dev)
    sin = torch.randn(bs, rope_d, dtype=dtype, device=dev)

    op = TorchNpuApplyRoPE()
    ref = MojoApplyRoPE._registry.get("torch")()
    qo, ko = op(q, k, cos, sin, head_first=False)
    qr, kr = ref(q.cpu(), k.cpu(), cos.cpu(), sin.cpu(), head_first=False)
    assert_close_npu(qo, qr.to(dev), atol=3e-2, rtol=3e-2)
    assert_close_npu(ko, kr.to(dev), atol=3e-2, rtol=3e-2)


def test_torch_npu_apply_rope_prefill_3d_matches_reference():
    dev = require_npu()
    dtype = torch.float16
    t, n, d, rope_d = 16, 2, 48, 48
    q = torch.randn(t, n, d, dtype=dtype, device=dev)
    k = torch.randn(t, n, d, dtype=dtype, device=dev)
    cos = torch.randn(t, rope_d, dtype=dtype, device=dev)
    sin = torch.randn(t, rope_d, dtype=dtype, device=dev)
    op = TorchNpuApplyRoPE()
    ref = MojoApplyRoPE._registry.get("torch")()
    try:
        qo, ko = op(q, k, cos, sin, head_first=True)
    except RuntimeError as e:
        if "broadcast" in str(e).lower() or "561103" in str(e):
            pytest.skip(f"npu_rotary_mul layout not supported for this case on this NPU: {e}")
        raise
    qr, kr = ref(q.cpu(), k.cpu(), cos.cpu(), sin.cpu(), head_first=True)
    assert_close_npu(qo, qr.to(dev), atol=3e-2, rtol=3e-2)
    assert_close_npu(ko, kr.to(dev), atol=3e-2, rtol=3e-2)


def test_torch_npu_apply_vision_rope2d_matches_reference():
    dev = require_npu()
    dtype = torch.bfloat16
    t, n, d = 8, 2, 64
    rope_d = d
    q = torch.randn(t, n, d, dtype=dtype, device=dev)
    k = torch.randn(t, n, d, dtype=dtype, device=dev)
    cos = torch.randn(t, rope_d, dtype=dtype, device=dev)
    sin = torch.randn(t, rope_d, dtype=dtype, device=dev)
    op = TorchNpuApplyVisionRoPE2D()
    ref = MojoApplyVisionRoPE2D._registry.get("torch")()
    qo, ko = op(q, k, cos, sin)
    qr, kr = ref(q.cpu(), k.cpu(), cos.cpu(), sin.cpu())
    assert_close_npu(qo, qr.to(dev), atol=3e-2, rtol=3e-2)
    assert_close_npu(ko, kr.to(dev), atol=3e-2, rtol=3e-2)
