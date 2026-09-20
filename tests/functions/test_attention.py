from functools import partial

import pytest
import torch

from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_accuracy
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

from .._checks import assert_close
from .._checks import assert_mojo_close
from .._checks import clone_with_grad

NATIVE_SWA_INFER_LAYOUTS = [(16, False, 1023), (16, True, 255), (8, False, 1023), (8, True, 255)]


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_swa_reference(dtype):
    torch.manual_seed(0)
    q, k, v = (torch.randn(n, 2, 8).to(dtype) for n in (17, 25, 25))
    cq, ck = (torch.tensor([0, n], dtype=torch.int32) for n in (17, 25))
    # Original core rounds QK and unnormalized PV to the input dtype, then
    # normalizes in FP32. Casting the operands before bmm changes this contract.
    scores = torch.bmm(q.transpose(0, 1), k.permute(1, 2, 0)).float() * 0.25
    mask = torch.arange(17)[:, None] + 8 >= torch.arange(25)[None, :]
    scores = scores.masked_fill(~mask, float("-inf"))
    p = (scores - scores.amax(-1, keepdim=True)).exp()
    expected = torch.bmm(p.to(dtype), v.transpose(0, 1)).float() / p.sum(-1, keepdim=True)
    expected = expected.transpose(0, 1).to(dtype)
    actual = F.swa(q, k, v, cq, ck, softmax_scale=0.25, implementation="torch_reference")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.reference
@pytest.mark.parametrize("dtype,large,small", [(torch.bfloat16, 1000.0, 1.0), (torch.float16, 64.0, 1 / 64)])
@pytest.mark.parametrize("pattern", ["incremental", "cancellation"])
@pytest.mark.parametrize("autocast_enabled", [False, True])
@pytest.mark.accuracy
def test_swa_reference_chunked_gradients(dtype, large, small, pattern, autocast_enabled):
    tokens = 4096 if pattern == "incremental" else 2048
    q = torch.ones(tokens, 1, 1, dtype=dtype, requires_grad=True)
    k = torch.zeros(2, 1, 1, dtype=dtype, requires_grad=True)
    v = torch.tensor([1.0, -1.0], dtype=dtype).reshape(2, 1, 1).requires_grad_()
    grad_output = torch.full_like(q, small)
    if pattern == "incremental":
        # Later 1024-query chunks must not disappear into the large first one.
        grad_output[:1024] = large
    else:
        # Each chunk contains a small term which disappears if its bmm result
        # is rounded before the large opposite-signed terms cancel.
        grad_output[:512] = large
        grad_output[1024:1536] = -large
    with torch.autocast("cpu", dtype=dtype, enabled=autocast_enabled):
        output = F.swa(
            q, k, v,
            torch.tensor([0, tokens], dtype=torch.int32),
            torch.tensor([0, 2], dtype=torch.int32),
            is_causal=False,
            softmax_scale=1.0,
            implementation="torch_reference",
        )
        dq, dk, dv = torch.autograd.grad(output, (q, k, v), grad_output)
    # Both attention probabilities are exactly 1/2 and output is exactly zero:
    # dK = (+sum(dO)/2, -sum(dO)/2), dV = (sum(dO)/2, sum(dO)/2).
    expected = (grad_output.float().sum() / 2).to(dtype)
    torch.testing.assert_close(dq, torch.zeros_like(q), rtol=0, atol=0)
    torch.testing.assert_close(dk, torch.stack((expected, -expected)).reshape_as(k), rtol=0, atol=0)
    torch.testing.assert_close(dv, expected.expand_as(v), rtol=0, atol=0)


@pytest.mark.reference
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.accuracy
def test_reference_flash_infer_matches_forward(causal):
    torch.manual_seed(1)
    q, k, v = torch.randn(7, 4, 8), torch.randn(11, 2, 8), torch.randn(11, 2, 8)
    cq, ck = torch.tensor([0, 3, 7], dtype=torch.int32), torch.tensor([0, 5, 11], dtype=torch.int32)
    kwargs = dict(causal=causal, implementation="torch_reference")
    actual = F.flash_attention_infer(q, k, v, cq, ck, 4, 6, **kwargs)
    expected = F.flash_attention(q, k, v, cq, ck, 4, 6, **kwargs)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        F.flash_attention_infer(q.requires_grad_(), k, v, cq, ck, 4, 6, **kwargs)
    with torch.no_grad():
        assert not F.flash_attention_infer(q, k, v, cq, ck, 4, 6, **kwargs).requires_grad


@pytest.mark.reference
@pytest.mark.parametrize(
    "kwargs,error",
    [({"dropout_p": 0.1}, NotImplementedError), ({"attention_mask": torch.ones(1)}, NotImplementedError)],
)
@pytest.mark.accuracy
def test_flash_infer_rejects_unsupported_options(kwargs, error):
    q = torch.randn(3, 1, 8)
    cu = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(error):
        F.flash_attention_infer(q, q, q, cu, cu, 3, 3, implementation="torch_reference", **kwargs)


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


@pytest.mark.api("functions.swa")
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


@pytest.mark.parametrize("dtype", (torch.bfloat16,))
@pytest.mark.parametrize(
    "function_name,output_f32",
    [
        pytest.param(op, output_f32, marks=pytest.mark.api("functions." + op))
        for op, output_f32 in [("flash_attention", None), ("swa", False), ("swa", True)]
    ],
)
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
def test_native_swa_infer(accuracy_backend, heads, interleave, window, lengths, case):
    implementation, _, device = accuracy_backend
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


@pytest.mark.parametrize(
    "op,output_f32",
    [
        pytest.param(op, output_f32, marks=pytest.mark.api("functions." + op))
        for op, output_f32 in [("flash_attention", None), ("swa", False), ("swa", True)]
    ],
)
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
    actual = F.swa(*inputs, cq, ck, implementation=implementation, output_f32=True, **options)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    reference = clone_with_grad(inputs)
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


@pytest.mark.accuracy
@pytest.mark.api("functions.native_swa_infer")
def test_native_swa_compile(accuracy_backend):
    implementation, _, device = accuracy_backend
    q = torch.randn(128, 8, 128, dtype=torch.bfloat16, device=device)
    k, v = torch.randn_like(q[:, :1]).contiguous(), torch.randn_like(q[:, :1]).contiguous()
    cu = torch.tensor([0, 128], dtype=torch.int32, device=device)

    def run(q, k, v):
        return F.native_swa_infer(
            q, k, v, cu, cu, local_window_size=1023, global_window_size=4, implementation=implementation
        )

    try:
        expected = run(q, k, v)  # Resolve wrappers and load extension before capture.
        actual = compile_fullgraph(run, backend="eager")(q, k, v)
        torch.testing.assert_close(actual, expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.accuracy
@pytest.mark.api("functions.swa_infer")
def test_swa_infer_compile(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("NPU Triton inference leaf contract")
    preload("swa_infer", implementation=implementation, target=target)
    q = torch.randn(128, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(256, 2, 128, dtype=q.dtype, device=device)
    v = torch.randn_like(k)
    cq = torch.tensor([0, 128], dtype=torch.int32, device=device)
    ck = torch.tensor([0, 256], dtype=torch.int32, device=device)
    nodes = []

    def compiler(graph, _inputs):
        nodes.extend(str(node.target) for node in graph.graph.nodes if node.op == "call_function")
        return graph.forward

    def run(q, k, v):
        return F.swa_infer(q, k, v, cq, ck, local_window_size=255, global_window_size=4, implementation=implementation)

    try:
        expected = run(q, k, v)
        actual = compile_fullgraph(run, backend=compiler)(q, k, v)
        torch.testing.assert_close(actual, expected)
        leaves = [name for name in nodes if name.startswith("mojo_")]
        assert len(leaves) == 1 and "swa_infer" in leaves[0], leaves
    finally:
        torch._dynamo.reset()


@pytest.mark.accuracy
@pytest.mark.api("functions.swa")
@pytest.mark.parametrize("op,output_f32", [("swa", False), ("swa", True)])
def test_compiled_leaves(accuracy_backend, op, output_f32):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("NPU Triton leaf decomposition contract")
    preload(op, implementation=implementation, target=target)
    nodes = {"forward": [], "backward": []}

    def compiler(phase):
        def capture(graph, inputs):
            nodes[phase].extend(str(node.target) for node in graph.graph.nodes if node.op == "call_function")
            return make_boxed_func(graph.forward)

        return capture

    backend = aot_autograd(fw_compiler=compiler("forward"), bw_compiler=compiler("backward"))
    cu = torch.tensor([0, 128], dtype=torch.int32, device=device)
    inputs = tuple(torch.randn(128, h, 128, dtype=torch.bfloat16, device=device, requires_grad=True) for h in (4, 2, 2))

    def run(q, k, v):
        return F.swa(q, k, v, cu, cu, output_f32=output_f32, implementation=implementation)

    expected_forward = {"swa_fwd"}
    expected_backward = {"swa_preprocess", "swa_dkdv", "swa_dq"}
    eager_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    try:
        compiled = compile_fullgraph(run, backend=backend)
        output, expected = compiled(*inputs), run(*eager_inputs)
        output = (output,) if isinstance(output, torch.Tensor) else output
        expected = (expected,) if isinstance(expected, torch.Tensor) else expected
        upstream = tuple(torch.randn_like(t) for t in output)
        for a, b in zip(output, expected):
            torch.testing.assert_close(a, b)
        actual_grads = torch.autograd.grad(output, inputs, upstream)
        expected_grads = torch.autograd.grad(expected, eager_inputs, upstream)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b)
        for phase, leaves in (("forward", expected_forward), ("backward", expected_backward)):
            names = {name.split(".")[-2] for name in nodes[phase] if name.startswith("mojo_")}
            assert leaves <= names, nodes[phase]
    finally:
        torch._dynamo.reset()
