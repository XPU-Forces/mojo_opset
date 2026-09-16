from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from tests._attention_reference import _chunked_swa_torch_backward
from tests._attention_reference import _chunked_swa_torch_forward
from tests._checks import assert_accuracy
from tests._checks import assert_repeatable

from .._checks import assert_close
from .._checks import assert_mojo_close
from .._checks import clone_with_grad

NATIVE_SWA_INFER_LAYOUTS = [(16, False, 1023), (16, True, 255), (8, False, 1023), (8, True, 255)]


NATIVE_SWA_INFER_LENGTHS = [([129, 257], [513, 1025]), ([129, 257], [2049, 4097]), ([1024], [9216])]


def make_native_swa_infer_case(device, heads, interleave, window, lengths):
    q_lens, k_lens = lengths
    cu_q = torch.tensor([0, *q_lens], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    cu_k = torch.tensor([0, *k_lens], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    q = torch.randn(sum(q_lens), heads, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(sum(k_lens), 4 if heads == 16 else 1, 128, dtype=q.dtype, device=device)
    v = torch.randn_like(k)
    kwargs = dict(local_window_size=window, global_window_size=4, gqa_interleave=interleave)
    return (q, k, v, cu_q, cu_k), kwargs


SWA_INFER_CASES = [
    pytest.param(2, 16, 4, 128, 1024, 0, torch.bfloat16, id="bf16"),
    pytest.param(2, 16, 4, 96, 1024, 0, torch.bfloat16, id="padded-dim"),
    pytest.param(2, 8, 1, 128, 1024, 2048, torch.bfloat16, id="cached-kv"),
]


SWA_INFER_WINDOWS = [("ABAB", 4, 255), ("AABB", 4, 1023)]


def assert_swa_infer_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    # Original check_tol_diff compares in FP32 using these operator-specific limits.
    rtol, atol = (1e-6, 1e-5) if actual.dtype == torch.float32 else (2e-2, 2e-2)
    assert_close(actual.float(), expected.float(), torch.float32, rtol=rtol, atol=atol)


def make_swa_infer_case(batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, device):
    # generate_sdpa_data in Mojo's operator test creates lengths and tensors on CPU.
    q_lens = torch.randint(max_q // 2, max_q, (batch,), dtype=torch.int32).clamp(min=1)
    kv_lens = q_lens
    if max_cache:
        kv_lens = q_lens + torch.randint(max_cache // 2, max_cache, (batch,), dtype=torch.int32)
    cq, ck = (
        torch.cat((torch.zeros(1, dtype=torch.int32), lens.cumsum(0, dtype=torch.int32))) for lens in (q_lens, kv_lens)
    )
    q = torch.randn(int(cq[-1]), q_heads, dim, dtype=dtype)
    k = torch.randn(int(ck[-1]), kv_heads, dim, dtype=dtype)
    v = torch.randn_like(k)
    return tuple(x.to(device) for x in (q, k, v, cq, ck))


SWA_CASES = [
    pytest.param(*case, id=name)
    for *case, name in [
        (2, 16, 4, 128, 1024, 0, torch.float32, "fp32"),
        (2, 16, 4, 96, 1024, 0, torch.bfloat16, "padded-dim"),
        (2, 16, 4, 128, 4096, 0, torch.bfloat16, "long-batch"),
        (1, 12, 4, 128, 1024, 0, torch.bfloat16, "1024"),
        (1, 12, 4, 128, 2048, 0, torch.bfloat16, "2048"),
        (1, 12, 4, 128, 4096, 0, torch.bfloat16, "4096"),
        (1, 12, 4, 128, 8192, 0, torch.bfloat16, "8192"),
        (1, 12, 4, 128, 16384, 0, torch.bfloat16, "16384"),
        (1, 12, 4, 128, 32768, 0, torch.bfloat16, "32768"),
    ]
]


def make_attention_case(device, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype):
    generator = torch.Generator().manual_seed(43)
    q_lens = torch.randint(max_q // 2, max_q, (batch,), generator=generator, dtype=torch.int32).clamp(min=1)
    kv_lens = q_lens.clone()
    if max_cache:
        kv_lens += torch.randint(max_cache // 2, max_cache, (batch,), generator=generator, dtype=torch.int32)
    cq, ck = (
        torch.cat((torch.zeros(1, dtype=torch.int32), lens.cumsum(0, dtype=torch.int32))).to(device)
        for lens in (q_lens, kv_lens)
    )
    tensors = tuple(
        torch.randn(tokens, heads, dim, device=device, dtype=dtype, requires_grad=True)
        for tokens, heads in [
            (int(q_lens.sum()), q_heads),
            (int(kv_lens.sum()), kv_heads),
            (int(kv_lens.sum()), kv_heads),
        ]
    )
    return tensors, cq, ck


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("windows", [(None, None), (63, 4), (None, 16)])
@pytest.mark.accuracy
def test_swa_windows(accuracy_backend, interleave, windows):
    device = accuracy_backend[2]
    cu = torch.tensor([0, 97, 256], dtype=torch.int32, device=device)

    def run(q, k, v, **selection):
        return F.swa(
            q,
            k,
            v,
            cu,
            cu,
            local_window_size=windows[0],
            global_window_size=windows[1],
            gqa_interleave=interleave,
            output_f32=True,
            **selection,
        )

    assert_accuracy(
        run,
        tuple(torch.randn(256, heads, 128, device=device, dtype=torch.bfloat16) for heads in (4, 2, 2)),
        implementation=accuracy_backend[0],
        grad_tolerances={i: (1e-2, 4e-2) for i in range(3)},
    )


def _make_attention_inputs(device, dtype):
    return tuple(
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        for shape in ((256, 4, 128), (256, 2, 128), (256, 2, 128))
    )


@pytest.mark.api("functions.flash_attention", "functions.swa")
@pytest.mark.parametrize("dtype", (torch.bfloat16,))
@pytest.mark.parametrize("function_name,output_f32", [("flash_attention", None), ("swa", False), ("swa", True)])
@pytest.mark.accuracy
def test_attention(accuracy_backend, dtype, function_name, output_f32):
    implementation, target, device = accuracy_backend
    reference_inputs = _make_attention_inputs(device, dtype)
    actual_inputs = clone_with_grad(reference_inputs)
    cu_lens = torch.tensor([0, 256], device=device, dtype=torch.int32)
    function = getattr(F, function_name)
    kwargs = (
        {"max_q_len": 256, "max_k_len": 256, "causal": True}
        if function_name == "flash_attention"
        else {"is_causal": True, "local_window_size": 128, "global_window_size": 16, "output_f32": output_f32}
    )

    expected = function(
        *reference_inputs,
        cu_lens,
        cu_lens,
        implementation="torch_reference",
        **kwargs,
    )
    actual = function(
        *actual_inputs,
        cu_lens,
        cu_lens,
        implementation=implementation,
        **kwargs,
    )
    grad_output = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)

    assert_close(actual, expected, dtype)
    grad_atol = 4e-2 if function_name == "swa" and dtype == torch.bfloat16 else None
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual_grad, expected_grad, dtype, atol=grad_atol)


@pytest.mark.api("functions.flash_attention_infer")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.accuracy
def test_flash_attention_infer(accuracy_backend, causal):
    implementation, _, device = accuracy_backend
    torch.manual_seed(1)
    cq = torch.tensor([0, 97, 226], dtype=torch.int32, device=device)
    ck = torch.tensor([0, 193, 450], dtype=torch.int32, device=device)
    q = torch.randn(226, 4, 128, device=device, dtype=torch.bfloat16)
    k, v = [torch.randn(450, 2, 128, device=device, dtype=q.dtype) for _ in range(2)]
    actual = F.flash_attention_infer(q, k, v, cq, ck, 129, 257, causal=causal, implementation=implementation)
    expected = F.flash_attention_infer(q, k, v, cq, ck, 129, 257, causal=causal, implementation="torch_reference")
    assert_close(actual, expected, q.dtype)


@pytest.mark.api("functions.native_swa_infer")
@pytest.mark.parametrize(
    "heads,interleave,window,lengths,case",
    [
        pytest.param(heads, interleave, window, lengths, "default", id=f"{heads}-{interleave}-{window}-{lengths}")
        for lengths in NATIVE_SWA_INFER_LENGTHS
        for heads, interleave, window in NATIVE_SWA_INFER_LAYOUTS
    ]
    + [
        pytest.param(8, False, 1023, ([4], [4]), case, id=case)
        for case in ("empty_cu", "cpu_cu", "zero_length", "short_kv", "scale", "window")
    ],
)
@pytest.mark.accuracy
def test_native_swa_infer(native_swa_backend, heads, interleave, window, lengths, case):
    implementation, device = native_swa_backend
    if case != "default" and implementation == "torch_reference":
        pytest.skip("native launch contract")
    torch.manual_seed(42)
    (q, k, v, cq, ck), kwargs = make_native_swa_infer_case(device, heads, interleave, window, lengths)
    if case == "empty_cu":
        cq = ck = torch.empty(0, device=device, dtype=torch.int32)
    elif case == "cpu_cu":
        cq, ck = cq.cpu(), ck.cpu()
    elif case == "zero_length":
        cq = ck = torch.tensor([0, 0, 4], dtype=torch.int32, device=device)
    elif case == "short_kv":
        k, v = k[:3], v[:3]
        ck = torch.tensor([0, 3], dtype=torch.int32, device=device)
    elif case == "scale":
        kwargs["softmax_scale"] = 1.0
    elif case == "window":
        kwargs["local_window_size"] = 127
    inputs = (q, k, v, cq, ck)
    if case != "default":
        with pytest.raises((RuntimeError, ValueError)):
            F.native_swa_infer(*inputs, implementation=implementation, **kwargs)
        return
    actual = F.native_swa_infer(*inputs, implementation=implementation, **kwargs)
    expected = F.native_swa_infer(*inputs, implementation="torch_reference", **kwargs)
    assert not actual.requires_grad
    assert_close(actual, expected, inputs[0].dtype)


@pytest.mark.api("functions.flash_attention", "functions.swa")
@pytest.mark.parametrize("op,output_f32", [("flash_attention", None), ("swa", False), ("swa", True)])
@pytest.mark.bitwise
def test_attention_bitwise(accuracy_backend, op, output_f32):
    implementation, _, device = accuracy_backend
    inputs = tuple(
        torch.randn(tokens, heads, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        for tokens, heads in ((129, 8), (257, 1), (257, 1))
    )
    cq, ck = (torch.tensor([0, n], device=device, dtype=torch.int32) for n in (129, 257))
    kwargs = (
        dict(max_q_len=129, max_k_len=257, causal=True)
        if op.startswith("flash_attention")
        else dict(is_causal=True, local_window_size=1023, global_window_size=4, output_f32=output_f32)
    )
    assert_repeatable(partial(getattr(F, op), implementation=implementation, **kwargs), (*inputs, cq, ck))


@pytest.mark.api("functions.swa_infer")
@pytest.mark.parametrize("batch,q_heads,kv_heads,dim,max_q,max_cache,dtype", SWA_INFER_CASES)
@pytest.mark.parametrize("layout,global_window,local_window", SWA_INFER_WINDOWS)
@pytest.mark.accuracy
def test_swa_infer(
    accuracy_backend, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, layout, global_window, local_window
):
    implementation, _, device = accuracy_backend
    inputs = make_swa_infer_case(batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, device)
    options = dict(
        local_window_size=local_window,
        global_window_size=global_window,
        gqa_interleave=layout == "ABAB",
        softmax_scale=dim**-0.5,
    )
    actual = F.swa_infer(*inputs, implementation=implementation, **options)
    expected = F.swa_infer(*inputs, implementation="torch_reference", **options)
    assert not actual.requires_grad
    assert_swa_infer_close(actual, expected)


@pytest.mark.api("functions.swa")
@pytest.mark.parametrize("batch,q_heads,kv_heads,dim,max_q,max_cache,dtype", SWA_CASES)
@pytest.mark.parametrize("interleave,global_window,local_window", [(True, 4, 255), (False, 4, 1023)])
@pytest.mark.accuracy
def test_swa(
    accuracy_backend, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, interleave, global_window, local_window
):
    implementation, _, device = accuracy_backend
    inputs, cq, ck = make_attention_case(device, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype)
    options = dict(
        is_causal=True,
        local_window_size=local_window,
        global_window_size=global_window,
        softmax_scale=dim**-0.5,
        gqa_interleave=interleave,
    )
    if max_q > 8192 and (device != "npu" or implementation == "torch_reference"):
        pytest.skip("the full-attention reference is quadratic; original long cases require an optimized NPU provider")
    actual = F.swa(*inputs, cq, ck, implementation=implementation, output_f32=True, **options)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    reference = tuple(x.detach() for x in inputs)
    if max_q > 8192:
        # Preserve the original long-sequence oracle without materializing an S x S matrix.
        expected, lse, output_f32 = _chunked_swa_torch_forward(*reference, cq, ck, **options, output_f32=True)
        expected_grads = _chunked_swa_torch_backward(upstream, *reference, output_f32, lse, cq, ck, **options)
    else:
        reference = clone_with_grad(reference)
        expected = F.swa(*reference, cq, ck, implementation="torch_reference", output_f32=True, **options)
        expected_grads = torch.autograd.grad(expected, reference, upstream)
    assert_mojo_close(actual, expected)
    for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads)):
        assert_mojo_close(actual_grad, expected_grad, name=f"grad[{index}]")


@pytest.mark.api("functions.swa")
@pytest.mark.parametrize("batch,q_heads,kv_heads,dim,max_q,max_cache,dtype", SWA_CASES)
@pytest.mark.parametrize("interleave,global_window,local_window", [(True, 4, 255), (False, 4, 1023)])
@pytest.mark.bitwise
def test_swa_shapes_bitwise(
    accuracy_backend,
    batch,
    q_heads,
    kv_heads,
    dim,
    max_q,
    max_cache,
    dtype,
    interleave,
    global_window,
    local_window,
):
    implementation, _, device = accuracy_backend
    if max_q > 8192 and (device != "npu" or implementation == "torch_reference"):
        pytest.skip("original long cases require an optimized NPU provider")
    inputs, cq, ck = make_attention_case(device, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype)
    assert_repeatable(
        partial(
            F.swa,
            local_window_size=local_window,
            global_window_size=global_window,
            gqa_interleave=interleave,
            output_f32=True,
            implementation=implementation,
        ),
        (*inputs, cq, ck),
    )
