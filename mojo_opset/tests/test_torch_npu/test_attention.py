"""Unit tests for ``torch_npu`` attention operators.

Uses shapes that hit the Python ``super().forward`` fallback inside the NPU
classes (``block_size`` / sequence length not aligned to 128), so results match
``Mojo*._registry.get("torch")`` exactly on A2 and A5 without depending on
``npu_fused_infer_attention_score`` tiling quirks.
"""

import math

import pytest
import torch

from mojo_opset.backends.torch_npu.operators.attention import TorchNpuPagedDecodeGQA
from mojo_opset.backends.torch_npu.operators.attention import TorchNpuPagedPrefillGQA
from mojo_opset.backends.torch_npu.operators.attention import TorchNpuPrefillGQA
from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPrefillGQA

from mojo_opset.tests.test_torch_npu.common import assert_close_npu
from mojo_opset.tests.test_torch_npu.common import require_npu


def test_torch_npu_prefill_gqa_fallback_matches_torch_reference():
    dev = require_npu()
    # Third dim of KV (here: seq length) not multiple of 128 → fused NPU kernel skipped.
    B, Hq, Hkv, D, S = 1, 8, 2, 128, 64
    dtype = torch.bfloat16
    q = torch.randn(B, Hq, S, D, dtype=dtype, device=dev)
    k = torch.randn(B, Hkv, S, D, dtype=dtype, device=dev)
    v = torch.randn(B, Hkv, S, D, dtype=dtype, device=dev)
    cu = torch.tensor([0, S], dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(D)
    op = TorchNpuPrefillGQA(is_causal=True, gqa_layout="ABAB")
    ref = MojoPrefillGQA._registry.get("torch")(is_causal=True, gqa_layout="ABAB")
    out = op(q, k, v, cu, softmax_scale=scale)
    ref_out = ref(q.cpu(), k.cpu(), v.cpu(), cu.cpu(), softmax_scale=scale)
    assert_close_npu(out, ref_out.to(dev), atol=1e-2, rtol=1e-2)


def test_torch_npu_paged_decode_gqa_fallback_matches_torch_reference():
    dev = require_npu()
    B, Hq, Hkv, D, blk, seq = 1, 8, 4, 128, 64, 64
    dtype = torch.bfloat16
    nb = 8
    q = torch.randn(B, Hq, D, dtype=dtype, device=dev)
    kc = torch.randn(nb, Hkv, blk, D, dtype=dtype, device=dev)
    vc = torch.randn(nb, Hkv, blk, D, dtype=dtype, device=dev)
    ts = torch.tensor([seq], dtype=torch.int32, device=dev)
    bt = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(D)
    op = TorchNpuPagedDecodeGQA(is_causal=True, gqa_layout="ABAB")
    ref = MojoPagedDecodeGQA._registry.get("torch")(is_causal=True, gqa_layout="ABAB")
    out = op(q, kc, vc, ts, bt, softmax_scale=scale)
    ref_out = ref(q.cpu(), kc.cpu(), vc.cpu(), ts.cpu(), bt.cpu(), softmax_scale=scale)
    assert_close_npu(out, ref_out.to(dev), atol=1e-2, rtol=1e-2)


def test_torch_npu_paged_prefill_gqa_fallback_matches_torch_reference():
    dev = require_npu()
    T, Hq, Hkv, D, blk = 64, 8, 2, 128, 64
    dtype = torch.bfloat16
    query = torch.randn(T, Hq, D, dtype=dtype, device=dev)
    k_cache = torch.randn(2, Hkv, blk, D, dtype=dtype, device=dev)
    v_cache = torch.randn(2, Hkv, blk, D, dtype=dtype, device=dev)
    cu_q_lens = torch.tensor([0, T], dtype=torch.int32, device=dev)
    block_tables = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(D)
    op = TorchNpuPagedPrefillGQA(is_causal=True, gqa_layout="ABAB")
    ref = MojoPagedPrefillGQA._registry.get("torch")(is_causal=True, gqa_layout="ABAB")
    out = op(
        query,
        k_cache,
        v_cache,
        cu_q_lens,
        block_tables,
        softmax_scale=scale,
        cu_total_seq_lens=None,
    )
    ref_out = ref(
        query.cpu(),
        k_cache.cpu(),
        v_cache.cpu(),
        cu_q_lens.cpu(),
        block_tables.cpu(),
        softmax_scale=scale,
        cu_total_seq_lens=None,
    )
    assert_close_npu(out, ref_out.to(dev), atol=1.5e-2, rtol=1.5e-2)
