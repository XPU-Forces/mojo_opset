"""Unit tests for ``torch_npu`` activation operators."""

import pytest
import torch
import torch.nn.functional as F

from mojo_opset.backends.torch_npu.operators.activation import TorchNpuGelu
from mojo_opset.backends.torch_npu.operators.activation import TorchNpuSilu
from mojo_opset.backends.torch_npu.operators.activation import TorchNpuSwiGLU

from mojo_opset.tests.test_torch_npu.common import assert_close_npu
from mojo_opset.tests.test_torch_npu.common import require_npu


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_npu_gelu_matches_torch(approximate, dtype):
    dev = require_npu()
    x = torch.randn(4, 256, dtype=dtype, device=dev)
    op = TorchNpuGelu()
    out = op(x, approximate=approximate)
    ref = F.gelu(x.cpu(), approximate=approximate).to(dtype).to(dev)
    assert_close_npu(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_npu_silu_matches_torch(dtype):
    dev = require_npu()
    x = torch.randn(4, 256, dtype=dtype, device=dev)
    op = TorchNpuSilu()
    try:
        out = op(x)
    except RuntimeError as e:
        if "npu_silu" in str(e) or "aclnn" in str(e).lower():
            pytest.skip(f"npu_silu not supported on this stack (use F.silu on NPU): {e}")
        raise
    ref = torch.nn.functional.silu(x.cpu()).to(dtype).to(dev)
    assert_close_npu(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_npu_swiglu_matches_torch(dtype):
    dev = require_npu()
    gate = torch.randn(2, 64, dtype=dtype, device=dev)
    up = torch.randn(2, 64, dtype=dtype, device=dev)
    op = TorchNpuSwiGLU(swiglu_limit=0.0)
    out = op(gate, up)
    ref = F.silu(gate.cpu()) * up.cpu()
    ref = ref.to(dtype).to(dev)
    assert_close_npu(out, ref, atol=5e-3, rtol=5e-3)
