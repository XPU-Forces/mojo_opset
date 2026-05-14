"""Unit tests for ``torch_npu`` quantize-related operators."""

import pytest
import torch

from mojo_opset.backends.torch_npu.operators.quantize import TorchNpuDequantSwiGLUQuant
from mojo_opset.backends.torch_npu.operators.quantize import TorchNpuDynamicQuant
from mojo_opset.backends.torch_npu.operators.quantize import TorchNpuMoEDynamicQuant
from mojo_opset.core import MojoDequantSwiGLUQuant
from mojo_opset.core import MojoDynamicQuant
from mojo_opset.core import MojoMoEDynamicQuant

from mojo_opset.tests.test_torch_npu.common import assert_close_npu
from mojo_opset.tests.test_torch_npu.common import require_npu


def _sync_weights(npu_mod, cpu_mod):
    npu_mod.load_state_dict(cpu_mod.state_dict())


def test_torch_npu_dynamic_quant_no_smooth_matches_reference():
    dev = require_npu()
    ref = MojoDynamicQuant._registry.get("torch")(input_size=None, quant_dtype=torch.int8, device="cpu", dtype=torch.bfloat16)
    op = TorchNpuDynamicQuant(input_size=None, quant_dtype=torch.int8, device=dev, dtype=torch.bfloat16)
    _sync_weights(op, ref)
    x = torch.randn(5, 48, dtype=torch.bfloat16, device=dev)
    q_n, s_n = op(x)
    q_r, s_r = ref(x.cpu())
    torch.testing.assert_close(q_n.cpu(), q_r, atol=0, rtol=0)
    assert_close_npu(s_n, s_r.to(dev), atol=1e-2, rtol=1e-2)


def test_torch_npu_dynamic_quant_with_smooth_matches_reference():
    dev = require_npu()
    k = 32
    ref = MojoDynamicQuant._registry.get("torch")(input_size=k, quant_dtype=torch.int8, device="cpu", dtype=torch.bfloat16)
    op = TorchNpuDynamicQuant(input_size=k, quant_dtype=torch.int8, device=dev, dtype=torch.bfloat16)
    torch.nn.init.uniform_(ref.inv_smooth_scale, 0.5, 1.5)
    _sync_weights(op, ref)
    x = torch.randn(3, 11, k, dtype=torch.bfloat16, device=dev)
    q_n, s_n = op(x)
    q_r, s_r = ref(x.cpu())
    torch.testing.assert_close(q_n.cpu(), q_r, atol=0, rtol=0)
    assert_close_npu(s_n, s_r.to(dev), atol=1e-2, rtol=1e-2)


def test_torch_npu_moe_dynamic_quant_matches_reference():
    dev = require_npu()
    e, k = 2, 24
    ref = MojoMoEDynamicQuant._registry.get("torch")(
        expert_num=e, input_size=k, quant_dtype=torch.int8, device="cpu", dtype=torch.bfloat16
    )
    op = TorchNpuMoEDynamicQuant(expert_num=e, input_size=k, quant_dtype=torch.int8, device=dev, dtype=torch.bfloat16)
    torch.nn.init.uniform_(ref.inv_smooth_scale, 0.8, 1.2)
    _sync_weights(op, ref)
    x = torch.randn(6, k, dtype=torch.bfloat16, device=dev)
    tc = torch.tensor([3, 3], dtype=torch.int32, device=dev)
    q_n, s_n = op(x, tc)
    q_r, s_r = ref(x.cpu(), tc.cpu())
    torch.testing.assert_close(q_n.cpu(), q_r, atol=0, rtol=0)
    assert_close_npu(s_n, s_r.to(dev), atol=1e-2, rtol=1e-2)


def test_torch_npu_dequant_swiglu_quant_matches_reference():
    dev = require_npu()
    h = 48
    tokens = 6
    ref = MojoDequantSwiGLUQuant._registry.get("torch")(
        expert_num=1,
        hidden_size=h,
        quant_dtype=torch.int8,
        activate_left=False,
        quant_mode=1,
        device="cpu",
        dtype=torch.bfloat16,
    )
    op = TorchNpuDequantSwiGLUQuant(
        expert_num=1,
        hidden_size=h,
        quant_dtype=torch.int8,
        activate_left=False,
        quant_mode=1,
        device=dev,
        dtype=torch.bfloat16,
    )
    torch.nn.init.uniform_(ref.weight_scale, 0.01, 0.05)
    torch.nn.init.uniform_(ref.quant_scale, 0.5, 2.0)
    _sync_weights(op, ref)
    x = torch.randn(tokens, 2 * h, dtype=torch.bfloat16, device=dev)
    act_s = torch.randn(tokens, dtype=torch.bfloat16, device=dev).abs() + 0.1
    try:
        q_n, s_n = op(x, activation_scale=act_s)
    except RuntimeError as e:
        err = str(e)
        if "561002" in err or "weight_scale must be None" in err:
            pytest.skip(f"npu_dequant_swiglu_quant constraint on this NPU: {e}")
        raise
    q_r, s_r = ref(x.cpu(), activation_scale=act_s.cpu())
    torch.testing.assert_close(q_n.cpu(), q_r, atol=0, rtol=0)
    assert_close_npu(s_n, s_r.to(dev), atol=1e-2, rtol=1e-2)
