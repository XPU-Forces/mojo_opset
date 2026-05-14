"""Unit tests for ``torch_npu`` normalization operators."""

import pytest
import torch

from mojo_opset.backends.torch_npu.operators.norm import TorchNpuLayerNormQuant
from mojo_opset.backends.torch_npu.operators.norm import TorchNpuResidualAddLayerNormQuant
from mojo_opset.backends.torch_npu.operators.norm import TorchNpuResidualAddRMSNorm
from mojo_opset.backends.torch_npu.operators.norm import TorchNpuResidualAddRMSNormQuant
from mojo_opset.backends.torch_npu.operators.norm import TorchNpuRMSNorm
from mojo_opset.backends.torch_npu.operators.norm import TorchNpuRMSNormQuant
from mojo_opset.core import MojoLayerNormQuant
from mojo_opset.core import MojoResidualAddLayerNormQuant
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoResidualAddRMSNormQuant
from mojo_opset.core import MojoRMSNorm
from mojo_opset.core import MojoRMSNormQuant

from mojo_opset.tests.test_torch_npu.common import assert_close_npu
from mojo_opset.tests.test_torch_npu.common import require_npu


def _init_learnable_params(mod: torch.nn.Module) -> None:
    """``Mojo*`` modules allocate ``empty`` parameters; fill before ``load_state_dict`` to NPU."""
    with torch.no_grad():
        for name, p in mod.named_parameters(recurse=False):
            if p is None:
                continue
            if p.ndim >= 1:
                torch.nn.init.uniform_(p, -0.1, 0.1)


def _copy_ref_to_npu(op: torch.nn.Module, ref: torch.nn.Module) -> None:
    _init_learnable_params(ref)
    op.load_state_dict(ref.state_dict())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_npu_rms_norm_matches_torch(dtype):
    dev = require_npu()
    d = 128
    ref = MojoRMSNorm._registry.get("torch")(d, eps=1e-5, device="cpu", dtype=dtype)
    op = TorchNpuRMSNorm(d, eps=1e-5, device=dev, dtype=dtype)
    _copy_ref_to_npu(op, ref)
    x = (torch.randn(2, 16, d, dtype=dtype, device=dev) * 0.1).contiguous()
    assert_close_npu(op(x), ref(x.cpu()).to(dev), atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_npu_residual_add_rms_norm_matches_torch(norm_pos, dtype):
    dev = require_npu()
    d = 64
    ref = MojoResidualAddRMSNorm._registry.get("torch")(
        d, eps=1e-5, norm_pos=norm_pos, device="cpu", dtype=dtype
    )
    op = TorchNpuResidualAddRMSNorm(d, eps=1e-5, norm_pos=norm_pos, device=dev, dtype=dtype)
    _copy_ref_to_npu(op, ref)
    h = (torch.randn(2, 10, d, dtype=dtype, device=dev) * 0.1).contiguous()
    r = (torch.randn(2, 10, d, dtype=dtype, device=dev) * 0.1).contiguous()
    o1, o2 = op(h, r)
    r1, r2 = ref(h.cpu(), r.cpu())
    assert_close_npu(o1, r1.to(dev), atol=2e-2, rtol=2e-2)
    assert_close_npu(o2, r2.to(dev), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_torch_npu_rms_norm_quant_matches_reference(dtype):
    dev = require_npu()
    d = 96
    ref = MojoRMSNormQuant._registry.get("torch")(d, eps=1e-5, quant_dtype=torch.int8, device="cpu", dtype=dtype)
    op = TorchNpuRMSNormQuant(d, eps=1e-5, quant_dtype=torch.int8, device=dev, dtype=dtype)
    _copy_ref_to_npu(op, ref)
    x = (torch.randn(3, 7, d, dtype=dtype, device=dev) * 0.1).contiguous()
    q_n, s_n = op(x)
    q_r, s_r = ref(x.cpu())
    assert_close_npu(s_n, s_r.to(dev), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q_n.cpu(), q_r, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_torch_npu_layer_norm_quant_matches_reference(dtype):
    """FP32 activations/weights: CANN ``layer_norm`` rejects BF16 weight with FP32 input."""
    dev = require_npu()
    d = 80
    ref = MojoLayerNormQuant._registry.get("torch")(
        d, eps=1e-5, quant_dtype=torch.int8, device="cpu", dtype=dtype
    )
    op = TorchNpuLayerNormQuant(d, eps=1e-5, quant_dtype=torch.int8, device=dev, dtype=dtype)
    _copy_ref_to_npu(op, ref)
    x = (torch.randn(2, 5, d, dtype=dtype, device=dev) * 0.1).contiguous()
    q_n, s_n = op(x)
    q_r, s_r = ref(x.cpu())
    assert_close_npu(s_n, s_r.to(dev), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q_n.cpu(), q_r, atol=0, rtol=0)


@pytest.mark.parametrize("norm_pos", ["pre", "post"])
def test_torch_npu_residual_add_rms_norm_quant_matches_reference(norm_pos):
    dev = require_npu()
    dtype = torch.bfloat16
    d = 72
    ref = MojoResidualAddRMSNormQuant._registry.get("torch")(
        d, eps=1e-5, norm_pos=norm_pos, quant_dtype=torch.int8, device="cpu", dtype=dtype
    )
    op = TorchNpuResidualAddRMSNormQuant(
        d, eps=1e-5, norm_pos=norm_pos, quant_dtype=torch.int8, device=dev, dtype=dtype
    )
    _copy_ref_to_npu(op, ref)
    h = (torch.randn(2, 4, d, dtype=dtype, device=dev) * 0.1).contiguous()
    r = (torch.randn(2, 4, d, dtype=dtype, device=dev) * 0.1).contiguous()
    q_n, res_n, s_n = op(h, r)
    q_r, res_r, s_r = ref(h.cpu(), r.cpu())
    torch.testing.assert_close(q_n.cpu(), q_r, atol=0, rtol=0)
    assert_close_npu(res_n, res_r.to(dev), atol=5e-3, rtol=5e-3)
    assert_close_npu(s_n, s_r.to(dev), atol=1e-2, rtol=1e-2)


def test_torch_npu_residual_add_layer_norm_quant_matches_reference():
    dev = require_npu()
    dtype = torch.float32
    d = 88
    ref = MojoResidualAddLayerNormQuant._registry.get("torch")(
        d, eps=1e-5, norm_pos="pre", quant_dtype=torch.int8, device="cpu", dtype=dtype
    )
    op = TorchNpuResidualAddLayerNormQuant(
        d, eps=1e-5, norm_pos="pre", quant_dtype=torch.int8, device=dev, dtype=dtype
    )
    _copy_ref_to_npu(op, ref)
    h = (torch.randn(2, 3, d, dtype=dtype, device=dev) * 0.1).contiguous()
    r = (torch.randn(2, 3, d, dtype=dtype, device=dev) * 0.1).contiguous()
    q_n, res_n, s_n = op(h, r)
    q_r, res_r, s_r = ref(h.cpu(), r.cpu())
    torch.testing.assert_close(q_n.cpu(), q_r, atol=0, rtol=0)
    assert_close_npu(res_n, res_r.to(dev), atol=5e-3, rtol=5e-3)
    assert_close_npu(s_n, s_r.to(dev), atol=1e-2, rtol=1e-2)
