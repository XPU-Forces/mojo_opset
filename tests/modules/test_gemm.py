from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules as M
from tests._checks import assert_close


def make_quant_batch(case, device):
    b, m, k, n, transposed, scale_dtype = case
    return (
        torch.randint(-128, 128, (b, m, k), dtype=torch.int8, device=device),
        torch.randint(-128, 128, (b, n, k) if transposed else (b, k, n), dtype=torch.int8, device=device),
        torch.randn(b, m, dtype=torch.float32, device=device) / 127,
        torch.randn(n, device=device).to(scale_dtype) / 127,
    )


def batch_case(case, device, impl, module=False):
    x, w, s1, s2 = make_quant_batch(case, device)
    call = (
        M.QuantBatchGemmReduceSum(w, case[4], implementation=impl)
        if module
        else lambda a, b, c: F.quant_batch_gemm_reduce_sum(a, w, b, c, trans_weight=case[4], implementation=impl)
    )
    ref = lambda a, b, c: F.quant_batch_gemm_reduce_sum(
        a, w, b, c, trans_weight=case[4], implementation="torch_reference"
    )
    return call, ref, (x, s1, s2)


def check_group(actual, expected, case, dtype):
    if case[0] > 1 and case[1] >= 3072:
        # The original grouped-GEMM matrix uses atol=1, rtol=2^-6, ptol=.90.
        close = torch.isclose(actual, expected, atol=1, rtol=2**-6)
        assert torch.isfinite(actual).all() and close.float().mean().item() >= 0.90
    else:
        assert_close(actual, expected, dtype, atol=1e-3, rtol=1e-3)


def make_group(case, dtype, device):
    groups, rows, k, n, transposed, counts = case
    if counts is None:
        values = torch.randint(1, max(2, 2 * rows // groups), (groups,))
        values = (values / values.sum() * rows).to(torch.int32)
        values[-1] += rows - int(values.sum())
    else:
        values = torch.tensor(counts, dtype=torch.int32)
    x = torch.randn(rows, k, dtype=dtype, device=device)
    weight = torch.randn((groups, n, k) if transposed else (groups, k, n), dtype=dtype, device=device)
    return x, weight, values.to(device)


def group_case(case, dtype, device, impl, module=False):
    if impl == "torch_npu" and dtype == torch.float32:
        pytest.skip("Original torch_npu grouped matmul does not support float32")
    x, weight, counts = make_group(case, dtype, device)
    call = (
        M.GroupGemm(weight, case[4], implementation=impl)
        if module
        else partial(F.group_gemm, weight=weight, trans_weight=case[4], implementation=impl)
    )
    ref = partial(F.group_gemm, weight=weight, trans_weight=case[4], implementation="torch_reference")
    return (
        (lambda a, b: call(a, b)) if module else (lambda a, b: call(a, group_list=b)),
        lambda a, b: ref(a, group_list=b),
        (x, counts),
    )


def _make_int8_gemm_data(m, k, n, trans_weight):
    """Create quantised input/weight pairs with corresponding scales.

    Returns weight in (N, K) layout when trans_weight=True, (K, N) otherwise.
    All tensors are contiguous.
    """
    x_fp = torch.randn(m, k)
    x_scale = (x_fp.abs().amax(dim=-1) / 127).clamp(min=1e-12)
    x_i8 = torch.clamp(torch.round(x_fp / x_scale.unsqueeze(-1)), -128, 127).to(torch.int8)
    w_fp_nk = torch.randn(n, k)
    w_scale = (w_fp_nk.abs().amax(dim=-1) / 127).clamp(min=1e-12).to(torch.bfloat16)
    w_i8_nk = torch.clamp(torch.round(w_fp_nk / w_scale.unsqueeze(-1)), -128, 127).to(torch.int8)
    if trans_weight:
        w_i8 = w_i8_nk
    else:
        w_i8 = w_i8_nk.t().contiguous()
    return (x_i8, w_i8, x_scale, w_scale)


def move_tensors(values, device):
    return tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in values)


def quant_case(case, dtype, transposed, device, impl, module=False):
    x, w, xs, ws = move_tensors(_make_int8_gemm_data(*case, transposed), device)
    if module:
        call = M.QuantGemm(case[1], case[2], dtype, transposed, implementation=impl, device=device)
        call.load_state_dict({"weight": w, "weight_scale": ws})
    else:
        call = lambda a, scale: F.quant_gemm(
            a, w, scale, ws, output_dtype=dtype, trans_weight=transposed, implementation=impl
        )
    ref = lambda a, scale: F.quant_gemm(
        a, w, scale, ws, output_dtype=dtype, trans_weight=transposed, implementation="torch_reference"
    )
    return call, ref, (x, xs)


def skip_batch(impl, device):
    if impl not in (None, "torch_npu", "torch_reference"):
        pytest.skip("quantized batch reduction has no Triton provider")
    if device == "npu" and impl != "torch_reference":
        pytest.skip("Inherited master exclusion: CANN 8.2 quantized batch reduction issue")


GROUP_GEMM_CASES = [
    (8, 8 * 2560, 4096, 4096, False, None),
    (4, 4 * 1024, 2048, 1024, False, None),
    (6, 6 * 512, 1024, 2048, True, None),
    (1, 256, 128, 64, False, [256]),
    (4, 192, 64, 96, False, [16, 64, 32, 80]),
    (4, 256, 128, 96, True, [48, 80, 64, 64]),
    (2, 192, 128, 96, False, [64, 128]),
    (2, 192, 128, 96, True, [64, 128]),
    (1, 3, 4, 6, False, [3]),
    (1, 5, 4, 6, False, [5]),
    (1, 10, 4, 6, False, [10]),
]


GROUP_GEMM_TYPED_CASES = [(case, dtype) for case in GROUP_GEMM_CASES for dtype in (torch.float16, torch.bfloat16)] + [
    ((1, 16, 32, 64, False, [16]), torch.float32),
    ((1, 8, 16, 32, False, [8]), torch.float32),
]


QUANT_BATCH_CASES = [
    (4, 7, 128, 256, False, torch.bfloat16),
    (1, 16, 64, 128, False, torch.bfloat16),
    (2, 9, 256, 512, False, torch.bfloat16),
    (4, 31, 128, 256, True, torch.bfloat16),
    (8, 1, 128, 128, False, torch.float16),
]


QUANT_GEMM_CASES = [(1, 4096, 4096), (32, 4096, 11008), (128, 2048, 4096), (64, 4096, 4096)]


@pytest.mark.api("modules.GroupGemm")
@pytest.mark.accuracy
@pytest.mark.parametrize("case,dtype", GROUP_GEMM_TYPED_CASES)
def test_group(accuracy_backend, case, dtype):
    impl, _, device = accuracy_backend
    call, ref, inputs = group_case(case, dtype, device, impl, True)
    check_group(call(*inputs), ref(*inputs), case, dtype)


@pytest.mark.api("modules.QuantGemm")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", QUANT_GEMM_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transposed", [False, True])
def test_quant(accuracy_backend, case, dtype, transposed):
    impl, _, device = accuracy_backend
    call, ref, inputs = quant_case(case, dtype, transposed, device, impl, True)
    assert_close(call(*inputs), ref(*inputs), dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.api("modules.QuantBatchGemmReduceSum")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", QUANT_BATCH_CASES)
def test_batch(accuracy_backend, case):
    impl, _, device = accuracy_backend
    skip_batch(impl, device)
    call, ref, inputs = batch_case(case, device, impl, True)
    assert_close(call(*inputs), ref(*inputs), torch.bfloat16, rtol=1e-2, atol=1e-1)


@pytest.mark.accuracy
def test_registered_weight(accuracy_backend):
    from mojo_opset.modules import QuantGemm

    _, _, device = accuracy_backend
    op = QuantGemm(16, 8, device=device)
    assert set(op.state_dict()) == {"weight", "weight_scale"}
    assert op.weight.dtype == torch.int8 and op.weight_scale.dtype == torch.bfloat16
    assert op.weight.shape == (16, 8) and op.weight_scale.shape == (8,)
