from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset.functions._dispatch import load_impl
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._checks import check_forward_backward
from tests._compile import compile_fullgraph


def make_quant_batch(case, device):
    b, m, k, n, transposed, scale_dtype = case
    return (
        torch.randint(-128, 128, (b, m, k), dtype=torch.int8, device=device),
        torch.randint(-128, 128, (b, n, k) if transposed else (b, k, n), dtype=torch.int8, device=device),
        torch.randn(b, m, dtype=torch.float32, device=device) / 127,
        torch.randn(n, device=device).to(scale_dtype) / 127,
    )


def batch_case(case, device, impl):
    x, w, s1, s2 = make_quant_batch(case, device)
    call = lambda a, b, c: F.quant_batch_gemm_reduce_sum(a, w, b, c, trans_weight=case[4], implementation=impl)
    ref = lambda a, b, c: F.quant_batch_gemm_reduce_sum(a, w, b, c, trans_weight=case[4], implementation="torch")
    return (call, ref, (x, s1, s2))


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


def skip_batch(impl, device):
    if device == "npu" and impl != "torch":
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
    *[
        (groups, groups - 1, 32, 48, transposed, [0] + [1] * (groups - 1))
        for groups in (127, 128, 129)
        for transposed in (False, True)
    ],
    (129, 1, 32, 48, False, [0] * 128 + [1]),
    (129, 1, 32, 48, True, [1] + [0] * 128),
    (3, 7, 32, 48, False, [2, 0, 5]),
    (3, 7, 32, 48, True, [2, 0, 5]),
    (3, 0, 32, 48, False, [0, 0, 0]),
    (0, 0, 32, 48, True, []),
]


GROUP_GEMM_TYPED_CASES = [(case, dtype) for case in GROUP_GEMM_CASES for dtype in (torch.float16, torch.bfloat16)] + [
    ((1, 16, 32, 64, False, [16]), torch.float32),
    ((1, 8, 16, 32, False, [8]), torch.float32),
    # FP32 wgrad must fit CBUF across multiple reduction tiles and empty groups.
    *[
        pytest.param((3, 273, 33, 65, transposed, [0, 17, 256]), torch.float32, id=f"fp32-multitile-{transposed}")
        for transposed in (False, True)
    ],
    # 384 experts / representative EP8 = 48; 1024 tokens * top-8 = 8192 rows.
    *[
        pytest.param((48, 8192, k, n, True, [0] + [174] * 46 + [188]), torch.bfloat16, id=f"g48-r8192-k{k}-n{n}")
        for k, n in [(5120, 3072), (1536, 5120), (7168, 4096), (2048, 7168)]
    ],
]


QUANT_BATCH_CASES = [
    (4, 7, 128, 256, False, torch.bfloat16),
    (1, 16, 64, 128, False, torch.bfloat16),
    (2, 9, 256, 512, False, torch.bfloat16),
    (4, 31, 128, 256, True, torch.bfloat16),
    (8, 1, 128, 128, False, torch.float16),
]


QUANT_GEMM_CASES = [(1, 4096, 4096), (32, 4096, 11008), (128, 2048, 4096), (64, 4096, 4096)]


@pytest.mark.api("functions.group_gemm")
@pytest.mark.parametrize("case,dtype", GROUP_GEMM_TYPED_CASES)
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
@pytest.mark.parametrize(
    "check", [pytest.param("accuracy", marks=pytest.mark.accuracy), pytest.param("bitwise", marks=pytest.mark.bitwise)]
)
def test_group(accuracy_backend, case, dtype, backward, check):
    impl, target, device = accuracy_backend
    if impl in ("torch_npu", "native") and dtype == torch.float32:
        pytest.skip("torch_npu/native grouped matmul supports FP16/BF16")
    if backward and load_impl("group_gemm", impl, target=target)[1] is None:
        pytest.skip("group_gemm backward is not implemented by this backend")
    x, weight, counts = make_group(case, dtype, device)
    inputs = (x.requires_grad_(backward), weight.requires_grad_(backward))
    call = partial(F.group_gemm, group_list=counts, trans_weight=case[4])
    if check == "bitwise":
        assert_repeatable(partial(call, implementation=impl), inputs, backward=backward)
    elif backward:
        check_forward_backward(call, inputs, implementation=impl)
    else:
        check_group(call(*inputs, implementation=impl), call(*inputs, implementation="torch"), case, dtype)


@pytest.mark.api("functions.group_gemm")
@pytest.mark.accuracy
@pytest.mark.parametrize("layout", ["transposed", "row_strided", "column_strided"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_group_input_layout(accuracy_backend, layout, transposed, dtype, training, compiled):
    impl, target, device = accuracy_backend
    if compiled and (impl == "torch" or not target.startswith("npu.")):
        pytest.skip("NPU backend custom-op compilation contract")
    case = (2, 17, 64, 32, transposed, [6, 11])
    _, rows, k, _, _, _ = case
    _, weight, counts = make_group(case, dtype, device)
    if layout == "transposed":
        x = torch.randn(k, rows, dtype=dtype, device=device).t()
    elif layout == "row_strided":
        x = torch.randn(2 * rows, k, dtype=dtype, device=device)[::2]
        counts = counts.cpu()
    else:
        x = torch.randn(rows, 2 * k, dtype=dtype, device=device)[:, 1::2]
        counts = torch.tensor([6, 0, 11, 0], device=device, dtype=torch.int64)[::2]
    assert not x.is_contiguous()
    if training and load_impl("group_gemm", impl)[1] is None:
        pytest.skip("group_gemm backward is not implemented by this backend")
    # Exercise noncontiguous weights as well as inputs through custom-op fakes.
    weight = weight.transpose(1, 2).contiguous().transpose(1, 2)
    x.requires_grad_(training)
    weight.requires_grad_(training)
    call = partial(F.group_gemm, group_list=counts, trans_weight=transposed, implementation=impl)
    if compiled:
        from mojo_opset import preload

        preload("group_gemm", implementation=impl)
        call = compile_fullgraph(call)
    try:
        actual = call(x, weight)
        expected = F.group_gemm(x, weight, counts, trans_weight=transposed, implementation="torch")
        check_group(actual, expected, case, dtype)
        assert actual.is_contiguous()
        if training:
            upstream = torch.randn(actual.shape[1], actual.shape[0], device=device, dtype=dtype).T
            actual_grads = torch.autograd.grad(actual, (x, weight), upstream)
            expected_grads = torch.autograd.grad(expected, (x, weight), upstream)
            assert_close(actual_grads, expected_grads, dtype)
    finally:
        if compiled:
            torch._dynamo.reset()


@pytest.mark.api("functions.quant_gemm")
@pytest.mark.accuracy
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    "case,dtype,reference_only",
    [(case, dtype, False) for case in QUANT_GEMM_CASES for dtype in (torch.float16, torch.bfloat16)]
    + [
        pytest.param(case, dtype, True, marks=pytest.mark.reference)
        for case in QUANT_GEMM_CASES[:3] + [(64, 512, 256)]
        for dtype in (torch.float32, torch.float16, torch.bfloat16)
    ],
)
def test_quant_gemm_accuracy(accuracy_backend, case, dtype, transposed, reference_only):
    impl, _, device = accuracy_backend
    x, weight, xscale, wscale = move_tensors(_make_int8_gemm_data(*case, transposed), device)
    options = dict(output_dtype=dtype, trans_weight=transposed)
    actual = F.quant_gemm(x, weight, xscale, wscale, implementation="torch" if reference_only else impl, **options)
    if reference_only:
        w = weight.t() if transposed else weight
        expected = ((x.float() @ w.float()) * xscale[:, None] * wscale.float()[None, :]).to(dtype)
    else:
        expected = F.quant_gemm(x, weight, xscale, wscale, implementation="torch", **options)
    tolerance = 0 if reference_only else 1e-2
    assert_close(actual, expected, dtype, rtol=tolerance, atol=tolerance)


@pytest.mark.api("functions.quant_batch_gemm_reduce_sum")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", QUANT_BATCH_CASES)
def test_quant_batch_gemm_reduce_sum_accuracy(accuracy_backend, case):
    impl, _, device = accuracy_backend
    skip_batch(impl, device)
    call, ref, inputs = batch_case(case, device, impl)
    assert_close(call(*inputs), ref(*inputs), torch.bfloat16, rtol=1e-2, atol=1e-1)
