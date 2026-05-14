"""Unit tests for ``torch_npu`` GEMM-related operators."""

import pytest
import torch

from mojo_opset.backends.torch_npu.operators.gemm import TorchNpuGroupGemm
from mojo_opset.backends.torch_npu.operators.gemm import TorchNpuQuantBatchGemmReduceSum
from mojo_opset.backends.torch_npu.operators.gemm import TorchNpuQuantGemm
from mojo_opset.core import MojoGroupGemm
from mojo_opset.experimental import MojoQuantBatchGemmReduceSum
from mojo_opset.tests.accuracy.operators.test_gemm import _load_gemm_dequant_module
from mojo_opset.tests.accuracy.operators.test_gemm import _make_int8_gemm_data
from mojo_opset.tests.accuracy.operators.test_gemm import generate_quant_group_gemm_data

from mojo_opset.tests.test_torch_npu.common import assert_close_npu
from mojo_opset.tests.test_torch_npu.common import require_npu

torch.manual_seed(42)


@pytest.mark.parametrize(
    "m, k, n",
    [(1, 256, 128), (8, 512, 256)],
)
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trans_weight", [False, True])
def test_torch_npu_quant_gemm_matches_float_reference(m, k, n, output_dtype, trans_weight):
    dev = require_npu()
    x_i8, w_i8, x_scale, w_scale = _make_int8_gemm_data(m, k, n, trans_weight)

    w_for_mm = w_i8.t().contiguous() if trans_weight else w_i8
    ref = (x_i8.float() @ w_for_mm.float()) * x_scale.unsqueeze(-1) * w_scale.float().unsqueeze(0)
    ref = ref.to(output_dtype)

    op = TorchNpuQuantGemm(
        in_features=k,
        out_features=n,
        output_dtype=output_dtype,
        trans_weight=trans_weight,
        device=dev,
    )
    op = _load_gemm_dequant_module(op, w_i8.to(dev), w_scale.to(dev))
    out = op(x_i8.to(dev), x_scale.to(dev)).cpu()

    if output_dtype == torch.float32:
        torch.testing.assert_close(out, ref, atol=2.5e-1, rtol=5e-3)
    else:
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trans_weight", [False, True])
def test_torch_npu_group_gemm_matches_torch_reference(dtype, trans_weight):
    dev = require_npu()
    # Two groups of 16 rows; K=64, N=48. NPU grouped matmul expects each weight[g] to be (K, N)
    # when trans_weight=False, else logical (N, K) storage transposed to (K, N).
    g, rows, k, n = 2, 16, 64, 48
    input = torch.randn(g * rows, k, dtype=dtype, device=dev)
    if trans_weight:
        weight = torch.randn(g, n, k, dtype=dtype, device=dev)
    else:
        weight = torch.randn(g, k, n, dtype=dtype, device=dev)
    group_list = torch.tensor([rows, rows], dtype=torch.int32, device=dev)

    op = TorchNpuGroupGemm(trans_weight=trans_weight, weight=weight)
    ref_mod = MojoGroupGemm._registry.get("torch")(trans_weight=trans_weight, weight=weight.cpu())
    out = op(input, group_list)
    ref = ref_mod(input.cpu(), group_list.cpu()).to(dev)
    assert_close_npu(out, ref, atol=5e-3, rtol=5e-3)


def test_torch_npu_quant_batch_gemm_reduce_sum_matches_reference():
    dev = require_npu()
    b, m, k, n = 2, 4, 64, 128
    x1, weight, x1_scale, x2_scale = generate_quant_group_gemm_data(
        b=b, m=m, k=k, n=n, trans_weight=False
    )
    ref_mod = MojoQuantBatchGemmReduceSum._registry.get("torch")(trans_weight=False, weight=weight.cpu())
    op = TorchNpuQuantBatchGemmReduceSum(trans_weight=False, weight=weight.to(dev))

    x1_d = x1.to(dev)
    x1s_d = x1_scale.to(dev)
    x2s_d = x2_scale.to(dev)
    try:
        out = op(x1_d, x1s_d, x2s_d)
    except RuntimeError as e:
        pytest.skip(f"npu_quant_matmul_reduce_sum not supported on this CANN/NPU: {e}")

    ref = ref_mod(x1.cpu(), x1_scale.cpu(), x2_scale.cpu())
    assert_close_npu(out, ref, atol=1.5e-1, rtol=1.5e-1)
