"""Tests for functions.attention: Flash Attention and SWA."""

import importlib

from functools import partial

import pytest
import torch

from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._checks import clone_with_grad
from tests._compile import compile_fullgraph
from tests._fixtures import check_swa_case


@pytest.mark.reference
@pytest.mark.parametrize(
    "kwargs,error",
    [({"dropout_p": 0.1}, NotImplementedError), ({"attention_mask": torch.ones(1)}, NotImplementedError)],
)
def test_flash_attention_infer_rejects_unsupported_options(kwargs, error):
    q = torch.randn(3, 1, 8)
    cu = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(error):
        F.flash_attention_infer(q, q, q, cu, cu, 3, 3, implementation="torch", **kwargs)


@pytest.mark.api("functions.flash_attention_infer")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.accuracy
def test_flash_attention_infer_accuracy(accuracy_backend, causal):
    implementation, _, device = accuracy_backend
    torch.manual_seed(1)
    cq = torch.tensor([0, 97, 226], dtype=torch.int32, device=device)
    ck = torch.tensor([0, 193, 450], dtype=torch.int32, device=device)
    q = torch.randn(226, 4, 128, device=device, dtype=torch.bfloat16)
    k, v = [torch.randn(450, 2, 128, device=device, dtype=q.dtype) for _ in range(2)]
    actual = F.flash_attention_infer(q, k, v, cq, ck, 129, 257, causal=causal, implementation=implementation)
    expected = F.flash_attention_infer(q, k, v, cq, ck, 129, 257, causal=causal, implementation="torch")
    assert_close(actual, expected, q.dtype)


NATIVE_SWA_INFER_LAYOUTS = [(16, False, 1023), (16, True, 255), (8, False, 1023), (8, True, 255)]


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_swa_reference_rounding_accuracy(dtype):
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
    actual = F.swa(q, k, v, cq, ck, softmax_scale=0.25, implementation="torch")
    assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.reference
@pytest.mark.parametrize("dtype,large,small", [(torch.bfloat16, 1000.0, 1.0), (torch.float16, 64.0, 1 / 64)])
@pytest.mark.parametrize("pattern", ["incremental", "cancellation"])
@pytest.mark.parametrize("autocast_enabled", [False, True])
@pytest.mark.accuracy
def test_swa_reference_chunked_gradient_accuracy(dtype, large, small, pattern, autocast_enabled):
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
            q,
            k,
            v,
            torch.tensor([0, tokens], dtype=torch.int32),
            torch.tensor([0, 2], dtype=torch.int32),
            is_causal=False,
            softmax_scale=1.0,
            implementation="torch",
        )
        dq, dk, dv = torch.autograd.grad(output, (q, k, v), grad_output)
    # Both attention probabilities are exactly 1/2 and output is exactly zero:
    # dK = (+sum(dO)/2, -sum(dO)/2), dV = (sum(dO)/2, sum(dO)/2).
    expected = (grad_output.float().sum() / 2).to(dtype)
    assert_close(dq, torch.zeros_like(q), rtol=0, atol=0)
    assert_close(dk, torch.stack((expected, -expected)).reshape_as(k), rtol=0, atol=0)
    assert_close(dv, expected.expand_as(v), rtol=0, atol=0)


@pytest.mark.reference
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.accuracy
def test_reference_flash_infer_matches_forward(causal):
    torch.manual_seed(1)
    q, k, v = torch.randn(7, 4, 8), torch.randn(11, 2, 8), torch.randn(11, 2, 8)
    cq, ck = torch.tensor([0, 3, 7], dtype=torch.int32), torch.tensor([0, 5, 11], dtype=torch.int32)
    kwargs = dict(causal=causal, implementation="torch")
    actual = F.flash_attention_infer(q, k, v, cq, ck, 4, 6, **kwargs)
    expected = F.flash_attention(q, k, v, cq, ck, 4, 6, **kwargs)
    assert_close(actual, expected)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        F.flash_attention_infer(q.requires_grad_(), k, v, cq, ck, 4, 6, **kwargs)
    with torch.no_grad():
        assert not F.flash_attention_infer(q, k, v, cq, ck, 4, 6, **kwargs).requires_grad


NATIVE_SWA_INFER_LENGTHS = [([129, 257], [513, 1025]), ([129, 257], [2049, 4097]), ([1024], [9216])]

SWA_PACKED_CASES = [
    pytest.param(16, False, 1023, ([129, 257], [513, 1025]), id="h16-aabb-ragged"),
    pytest.param(16, True, 255, ([129, 257], [2049, 4097]), id="h16-abab-cached-prefix"),
    pytest.param(8, False, 1023, ([1024], [4096]), id="h8-aabb-single"),
    pytest.param(8, True, 255, ([257, 511], [1025, 2047]), id="h8-abab-ragged"),
]


def _make_native_swa_infer_case(device, heads, interleave, window, lengths):
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


def _assert_swa_infer_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    # Original check_tol_diff compares in FP32 using these operator-specific limits.
    rtol, atol = (1e-6, 1e-5) if actual.dtype == torch.float32 else (2e-2, 2e-2)
    assert_close(actual.float(), expected.float(), torch.float32, rtol=rtol, atol=atol)


def _make_swa_infer_case(batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, device):
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


def _make_swa_case(device, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype):
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


def _make_attention_inputs(device, dtype, tokens=256, q_heads=4, kv_heads=2):
    return tuple(
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        for shape in ((tokens, q_heads, 128), (tokens, kv_heads, 128), (tokens, kv_heads, 128))
    )


@pytest.mark.parametrize(
    "tokens,q_heads,kv_heads,dtype,causal,packed",
    [
        pytest.param(
            tokens,
            hq,
            hk,
            dtype,
            causal,
            packed,
            marks=pytest.mark.api("functions.flash_attention"),
        )
        for tokens, hq, hk in [
            (256, 4, 2),
            # Real head counts; T=1024 is a single-card CI truncation.
            # TODO: restore after BF16 GQA dK/dV tolerance failures in Triton, torch_npu and native are resolved.
            # (1024, 48, 8),
            # (1024, 56, 8),
        ]
        for dtype in (torch.float16, torch.bfloat16)
        for causal in (False, True)
        for packed in (False, True)
    ],
)
@pytest.mark.accuracy
def test_flash_attention_accuracy(accuracy_backend, dtype, tokens, q_heads, kv_heads, causal, packed):
    implementation, _, device = accuracy_backend
    reference_inputs = _make_attention_inputs(device, dtype, tokens, q_heads, kv_heads)
    if packed:
        reference_inputs = (
            reference_inputs[0],
            *(torch.randn(tokens + 2, kv_heads, 128, device=device, dtype=dtype, requires_grad=True) for _ in range(2)),
        )
    actual_inputs = clone_with_grad(reference_inputs)
    cq = torch.tensor([0, tokens // 2 - 1, tokens] if packed else [0, tokens], device=device, dtype=torch.int32)
    ck = torch.tensor([0, tokens // 2, tokens + 2] if packed else [0, tokens], device=device, dtype=torch.int32)
    function = F.flash_attention
    kwargs = {"max_q_len": tokens, "max_k_len": tokens + 2 if packed else tokens, "causal": causal}

    expected = function(
        *reference_inputs,
        cq,
        ck,
        implementation="torch",
        **kwargs,
    )
    actual = function(
        *actual_inputs,
        cq,
        ck,
        implementation=implementation,
        **kwargs,
    )
    grad_output = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)

    assert_close(actual, expected, dtype)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual_grad, expected_grad, dtype)


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
def test_native_swa_infer_accuracy(accuracy_backend, heads, interleave, window, lengths, case):
    implementation, _, device = accuracy_backend
    rejected_case = case != "default"
    if rejected_case and implementation == "torch":
        pytest.skip("native launch contract")
    torch.manual_seed(42)
    (q, k, v, cq, ck), kwargs = _make_native_swa_infer_case(device, heads, interleave, window, lengths)
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
    if rejected_case:
        with pytest.raises((RuntimeError, ValueError)):
            F.native_swa_infer(*inputs, implementation=implementation, **kwargs)
        return
    actual = F.native_swa_infer(*inputs, implementation=implementation, **kwargs)
    expected = F.native_swa_infer(*inputs, implementation="torch", **kwargs)
    assert not actual.requires_grad
    assert_close(actual, expected, inputs[0].dtype)


@pytest.mark.parametrize(
    "q_tokens,k_tokens,q_heads,kv_heads,dtype,causal",
    [
        pytest.param(tq, tk, hq, hk, dtype, causal, marks=pytest.mark.api("functions.flash_attention"))
        for tq, tk, hq, hk, dtype, causal in [
            (129, 257, 8, 1, torch.float16, False),
            (129, 257, 8, 1, torch.float16, True),
            (129, 257, 8, 1, torch.bfloat16, False),
            (129, 257, 8, 1, torch.bfloat16, True),
            (1024, 1024, 48, 8, torch.bfloat16, True),
            (1024, 1024, 56, 8, torch.bfloat16, True),
        ]
    ],
)
@pytest.mark.bitwise
def test_flash_attention_bitwise(accuracy_backend, q_tokens, k_tokens, q_heads, kv_heads, dtype, causal):
    implementation, _, device = accuracy_backend
    inputs = tuple(
        torch.randn(tokens, heads, 128, device=device, dtype=dtype, requires_grad=True)
        for tokens, heads in ((q_tokens, q_heads), (k_tokens, kv_heads), (k_tokens, kv_heads))
    )
    cq, ck = (torch.tensor([0, n], device=device, dtype=torch.int32) for n in (q_tokens, k_tokens))
    kwargs = dict(max_q_len=q_tokens, max_k_len=k_tokens, causal=causal)
    assert_repeatable(partial(F.flash_attention, implementation=implementation, **kwargs), (*inputs, cq, ck))


@pytest.mark.api("functions.swa_infer")
@pytest.mark.parametrize("batch,q_heads,kv_heads,dim,max_q,max_cache,dtype", SWA_INFER_CASES)
@pytest.mark.parametrize("layout,global_window,local_window", SWA_INFER_WINDOWS)
@pytest.mark.accuracy
def test_swa_infer_accuracy(
    accuracy_backend, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, layout, global_window, local_window
):
    implementation, _, device = accuracy_backend
    inputs = _make_swa_infer_case(batch, q_heads, kv_heads, dim, max_q, max_cache, dtype, device)
    options = dict(
        local_window_size=local_window,
        global_window_size=global_window,
        gqa_interleave=layout == "ABAB",
        softmax_scale=dim**-0.5,
    )
    actual = F.swa_infer(*inputs, implementation=implementation, **options)
    expected = F.swa_infer(*inputs, implementation="torch", **options)
    assert not actual.requires_grad
    _assert_swa_infer_close(actual, expected)


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
    selected = check_swa_case(accuracy_backend, dtype=dtype, dim=dim, interleave=interleave)
    if max_q > 8192 and (device != "npu" or implementation == "torch"):
        pytest.skip("original long cases require an optimized NPU provider")
    inputs, cq, ck = _make_swa_case(device, batch, q_heads, kv_heads, dim, max_q, max_cache, dtype)
    assert_repeatable(
        partial(
            F.swa,
            local_window_size=local_window,
            global_window_size=global_window,
            gqa_interleave=interleave,
            output_f32=selected != "torch_npu",
            implementation=implementation,
        ),
        (*inputs, cq, ck),
    )


@pytest.mark.api("functions.native_swa_infer")
def test_native_swa_infer_compile_matches_eager(accuracy_backend):
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
        assert_close(actual, expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.swa_infer")
def test_swa_infer_compiles_as_single_leaf(accuracy_backend):
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
        assert_close(actual, expected)
        leaves = [name for name in nodes if name.startswith("mojo_")]
        assert len(leaves) == 1 and "swa_infer" in leaves[0], leaves
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize(
    "op,output_f32",
    [
        pytest.param(op, f32, marks=pytest.mark.api("functions." + op))
        for op, f32 in [("swa", False), ("swa", True), ("flash_attention", False)]
    ],
)
def test_attention_compiles_forward_backward_leaves(accuracy_backend, op, output_f32):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton", "torch_npu") or not target.startswith("npu."):
        pytest.skip("NPU leaf decomposition contract")
    if op == "swa":
        implementation = check_swa_case(accuracy_backend, output_f32=output_f32)
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
        # Torch 2.13 Dynamo rejects duplicate Tensor inputs to Function.apply.
        # A zero-copy view keeps Q/K offsets equal without duplicating storage.
        if op == "flash_attention":
            return F.flash_attention(q, k, v, cu, cu.view_as(cu), 128, 128, causal=True, implementation=implementation)
        return F.swa(
            q,
            k,
            v,
            cu,
            cu.view_as(cu),
            local_window_size=63,
            global_window_size=4,
            output_f32=output_f32,
            implementation=implementation,
        )

    expected_forward = {"swa_fwd"}
    expected_backward = {"swa_preprocess", "swa_dkdv", "swa_dq"}
    if op == "swa" and implementation == "torch_npu":
        expected_backward = {"swa_bwd"}
    if op == "flash_attention" and implementation == "torch_npu":
        expected_forward = {"flash_attention_fwd"}
        expected_backward = {"flash_attention_bwd"}
    eager_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    try:
        compiled = compile_fullgraph(run, backend=backend)
        output, expected = compiled(*inputs), run(*eager_inputs)
        upstream = torch.randn_like(output)
        assert_close(output, expected)
        actual_grads = torch.autograd.grad(output, inputs, upstream)
        expected_grads = torch.autograd.grad(expected, eager_inputs, upstream)
        assert_close(actual_grads, expected_grads)
        for phase, leaves in (("forward", expected_forward), ("backward", expected_backward)):
            names = {name.split(".")[-2] for name in nodes[phase] if name.startswith("mojo_")}
            assert leaves <= names, nodes[phase]
    finally:
        torch._dynamo.reset()


def _make_swa_inputs(device, dtype):
    return tuple(
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        for shape in ((256, 4, 128), (256, 2, 128), (256, 2, 128))
    )


@pytest.mark.api("functions.swa")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "case,lengths,heads,interleave,local_window,global_window,output_f32",
    [(None, [0, 256], 4, False, 128, 16, output_f32) for output_f32 in (False, True)]
    + [
        (None, [0, 97, 256], 4, interleave, local, global_, True)
        for interleave in (False, True)
        for local, global_ in ((None, None), (63, 4), (None, 16))
    ]
    + [
        pytest.param(
            None,
            lengths,
            heads,
            interleave,
            local,
            4,
            output_f32,
            id=f"{case.id}-cache-fp32{output_f32}",
        )
        for case in SWA_PACKED_CASES
        for heads, interleave, local, lengths in [case.values]
        for output_f32 in (False, True)
    ]
    + [
        pytest.param(
            case.values,
            None,
            None,
            interleave,
            local,
            global_,
            True,
            id=f"{case.id}-interleave{interleave}-lw{local}-gw{global_}",
        )
        for case in SWA_CASES
        for interleave, global_, local in ((True, 4, 255), (False, 4, 1023))
    ],
)
def test_swa_accuracy(accuracy_backend, case, lengths, heads, interleave, local_window, global_window, output_f32):
    implementation, _, device = accuracy_backend
    dtype = torch.bfloat16 if case is None else case[-1]
    dim = 128 if case is None else case[3]
    selected = check_swa_case(
        accuracy_backend,
        dtype=dtype,
        dim=dim,
        interleave=interleave,
        output_f32=output_f32 if case is None else False,
        windows=(local_window, global_window),
    )
    if case is not None:
        output_f32 = selected != "torch_npu"
    packed_case = case is None and isinstance(lengths[0], list)
    if case is None:
        if packed_case:
            torch.manual_seed(43)
            (q, k, v, cq, ck), _ = _make_native_swa_infer_case(device, heads, interleave, local_window, lengths)
            inputs = tuple(t.requires_grad_() for t in (q, k, v))
        else:
            inputs = _make_swa_inputs(device, torch.bfloat16)
            cq = ck = torch.tensor(lengths, dtype=torch.int32, device=device)
        output_tolerance, grad_tolerance = (1e-2, 1e-2), (1e-2, 4e-2)
    else:
        inputs, cq, ck = _make_swa_case(device, *case)
        # Preserve the original shape-suite pointwise budgets.
        output_tolerance = grad_tolerance = (6e-3, 6e-3) if case[-1] == torch.float32 else (5e-2, 1e-1)
    options = dict(
        is_causal=True,
        local_window_size=local_window,
        global_window_size=global_window,
        gqa_interleave=interleave,
        output_f32=output_f32,
    )
    reference_inputs = clone_with_grad(inputs)
    actual = F.swa(*inputs, cq, ck, implementation=implementation, **options)
    expected = F.swa(*reference_inputs, cq, ck, implementation="torch", **options)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream)
    assert_close(actual, expected, rtol=output_tolerance[0], atol=output_tolerance[1])
    for index, (a, b) in enumerate(zip(actual_grads, expected_grads)):
        assert_close(a, b, rtol=grad_tolerance[0], atol=grad_tolerance[1], name=f"grad[{index}]")


@pytest.mark.api("functions.swa")
@pytest.mark.parametrize("output_f32", [False, True])
@pytest.mark.bitwise
def test_swa_bitwise(accuracy_backend, output_f32):
    implementation, _, device = accuracy_backend
    check_swa_case(accuracy_backend, output_f32=output_f32)
    inputs = tuple(
        torch.randn(tokens, heads, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        for tokens, heads in ((129, 8), (257, 1), (257, 1))
    )
    cq, ck = (torch.tensor([0, n], device=device, dtype=torch.int32) for n in (129, 257))
    kwargs = dict(is_causal=True, local_window_size=1023, global_window_size=4, output_f32=output_f32)
    assert_repeatable(partial(F.swa, implementation=implementation, **kwargs), (*inputs, cq, ck))


# Independent CPU FP64 oracle: also checks repeated calls and untouched KV gradients.
@pytest.mark.api("functions.swa")
@pytest.mark.accuracy
@pytest.mark.parametrize("lengths", [((257, 257),), ((128, 384), (129, 386)), ((256, 2048),)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleave", [False, True])
def test_swa_backward_varlen_matches_fp64(accuracy_backend, lengths, dtype, interleave):
    implementation, _, device = accuracy_backend
    selected = check_swa_case(accuracy_backend, dtype=dtype, interleave=interleave)
    generator = torch.Generator().manual_seed(17)
    q_heads, kv_heads, dim = 4, 2, 128
    local_window, global_window = 128, 32
    total_q = sum(q_len for q_len, _ in lengths)
    total_kv = sum(kv_len for _, kv_len in lengths)
    inputs = [
        torch.randn(tokens, heads, dim, generator=generator, dtype=dtype, device="cpu")
        for tokens, heads in [(total_q, q_heads), (total_kv, kv_heads), (total_kv, kv_heads)]
    ]
    grad_out = torch.randn(total_q, q_heads, dim, generator=generator, dtype=dtype, device="cpu")
    refs = [tensor.double().requires_grad_() for tensor in inputs]
    cu_q, cu_kv = [0], [0]
    outputs, unattended = [], []
    head_indices = torch.arange(q_heads, device="cpu")
    head_indices = head_indices % kv_heads if interleave else head_indices // (q_heads // kv_heads)
    for q_len, kv_len in lengths:
        q = refs[0][cu_q[-1] : cu_q[-1] + q_len].transpose(0, 1)
        k = refs[1][cu_kv[-1] : cu_kv[-1] + kv_len][:, head_indices].transpose(0, 1)
        v = refs[2][cu_kv[-1] : cu_kv[-1] + kv_len][:, head_indices].transpose(0, 1)
        q_pos = torch.arange(q_len, device="cpu")[:, None] + kv_len - q_len
        k_pos = torch.arange(kv_len, device="cpu")[None, :]
        mask = (k_pos <= q_pos) & ((k_pos < global_window) | (k_pos >= q_pos - local_window))
        scores = (q @ k.transpose(-1, -2)) * dim**-0.5
        probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        outputs.append((probabilities @ v).transpose(0, 1))
        unattended.append(~mask.any(dim=0))
        cu_q.append(cu_q[-1] + q_len)
        cu_kv.append(cu_kv[-1] + kv_len)
    expected = torch.cat(outputs)
    expected.backward(grad_out.double())
    unattended = torch.cat(unattended)
    cu_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_kv = torch.tensor(cu_kv, dtype=torch.int32, device=device)
    grad_out = grad_out.to(device)
    # Repeat to expose stale accumulator contents left by earlier invocations.
    for _ in range(2):
        actual_inputs = [tensor.to(device).requires_grad_() for tensor in inputs]
        actual = F.swa(
            *actual_inputs,
            cu_q,
            cu_kv,
            is_causal=True,
            local_window_size=local_window,
            global_window_size=global_window,
            softmax_scale=dim**-0.5,
            gqa_interleave=interleave,
            output_f32=selected != "torch_npu",
            implementation=implementation,
        )
        actual.backward(grad_out)
        pairs = [(actual, expected)] + [(got.grad, ref.grad) for got, ref in zip(actual_inputs, refs)]
        for got, ref in pairs:
            got = got.detach().cpu().double()
            ref = ref.detach()
            tolerance = 0.02 if dtype == torch.bfloat16 else 0.006
            assert_close(got, ref, atol=tolerance, rtol=tolerance)
            assert (got - ref).norm() / ref.norm() < 0.01
        for got in actual_inputs[1:]:
            assert torch.count_nonzero(got.grad.cpu()[unattended]).item() == 0


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_generate_mask = None

if triton is not None:

    @triton.jit
    def read_tiles(
        template,
        output,
        positions,
        rows,
        cols,
        seq_len,
        BM: tl.constexpr,
        BN: tl.constexpr,
        GW: tl.constexpr,
        LW: tl.constexpr,
    ):
        tile = tl.program_id(0)
        m = tl.load(positions + 2 * tile)
        n = tl.load(positions + 2 * tile + 1)
        mask = _generate_mask(template, rows, cols, BM, BN, m, n, GW, LW, seq_len, seq_len)
        offsets = tile * BM * BN + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        tl.store(output + offsets, mask)


@pytest.mark.api("functions.swa")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "block_m,block_n,window,global_window",
    [
        # No local window versus the diagonal-only local window.
        (128, 128, None, 4),
        (128, 128, 0, 4),
        # Compression threshold and sink alignment boundaries.
        (128, 128, 511, 127),
        (128, 128, 512, 128),
        (128, 128, 513, 0),
        (128, 128, 513, 127),
        (128, 128, 513, 128),
        (128, 128, 513, 129),
        (128, 128, 262143, 4),
        # Other tile shapes need only one compressed, unaligned-sink case.
        (64, 64, 513, 129),
        (256, 64, 513, 129),
    ],
)
def test_swa_mask_template_tiles(accuracy_backend, block_m, block_n, window, global_window):
    if triton is None:
        pytest.skip("Triton is required for mask tile checks")
    _, target, device = accuracy_backend
    if not (target.startswith("npu.a2") or target.startswith("npu.a5")):
        pytest.skip("NPU Triton mask template")
    arch = "a2" if target.startswith("npu.a2") else "a5"
    if arch == "a5" and block_m > 128:
        pytest.skip("A5 uses tiles up to 128")

    global _generate_mask
    backend = importlib.import_module(f"mojo_opset.kernels.npu_{arch}_triton.swa")

    _generate_mask = backend.gen_mask_causal_with_window

    lw = 0 if window is None else window
    gw = 0 if global_window is None else global_window

    # Probe both sides of each boundary, including cached-prefix offsets that
    # are not aligned to the Q tile. Only visit K tiles reachable by causal Q.
    starts = {0, 1, 127, 128, 255, 256, 511, 512, 1024}
    starts.update(max(0, lw + gw + offset) for offset in (-block_m, -1, 0, 1, block_n, 2 * block_m))
    positions = set()
    for m in starts:
        columns = {
            0,
            gw // block_n * block_n,
            (gw + block_n - 1) // block_n * block_n,
            m // block_n * block_n,
            max(0, (m - lw) // block_n * block_n),
        }
        for n in columns:
            if n < m + block_m:
                positions.add((m, n))
    positions = sorted(positions)
    coordinates = torch.tensor(positions, dtype=torch.int32, device=device)
    template = backend.get_mask_causal_with_window(block_m, block_n, window, global_window, device)
    output = torch.empty((len(positions), block_m, block_n), dtype=torch.bool, device=device)
    read_tiles[(len(positions),)](
        template,
        output,
        coordinates,
        *template.shape,
        max(starts) + block_m,
        block_m,
        block_n,
        global_window,
        window,
    )
    cpu = torch.tensor(positions, device="cpu")
    q = cpu[:, 0, None, None] + torch.arange(block_m, device="cpu")[None, :, None]
    k = cpu[:, 1, None, None] + torch.arange(block_n, device="cpu")[None, None, :]
    local = (k >= q - lw) if window is not None else torch.zeros_like(k <= q)
    expected = (k <= q) & ((k < gw) | local)
    assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "op", [pytest.param(op, marks=pytest.mark.api("functions." + op)) for op in ("swa", "flash_attention")]
)
@pytest.mark.accuracy
def test_external_context(accuracy_backend, op):
    """Fused callers own a child context; the outer Function owns saved tensors.

    Exercise the eager forward/backward contract used by xpu_speed. Direct
    fullgraph capture of Python context mutation is not part of this contract;
    public API compilation is covered by test_compiled_leaves.
    """
    from mojo_opset.functions import attention

    impl, _, device = accuracy_backend
    if op == "swa":
        impl = check_swa_case(accuracy_backend)
    function = attention.SwaFunction if op == "swa" else attention.FlashAttentionFunction
    cu = torch.tensor([0, 128], device=device, dtype=torch.int32)
    options = (
        (cu, cu, True, 63, 4, 128**-0.5, False, impl != "torch_npu", impl)
        if op == "swa"
        else (None, cu, cu, 128, 128, 0.0, 128**-0.5, True, True, impl)
    )

    class AttentionContext:
        def __init__(self):
            self._saved = ()

        def save_for_backward(self, *tensors):
            self._saved = tensors

        @property
        def saved_tensors(self):
            return self._saved

    class Fused(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            inner = AttentionContext()
            output = function.forward(inner, q, k, v, *options)
            ctx.save_for_backward(*inner.saved_tensors)
            inner.save_for_backward()
            ctx.inner = inner
            return output

        @staticmethod
        def backward(ctx, grad):
            inner = ctx.inner
            inner.save_for_backward(*ctx.saved_tensors)
            grads = function.backward(inner, grad)
            inner.save_for_backward()
            return grads[:3]

    # Keep two forwards alive, then run backward in reverse order to catch
    # accidentally shared child contexts or caches between invocations.
    pending = []
    for _ in range(2):
        inputs = _make_attention_inputs(device, torch.bfloat16, 128, 4, 2)
        expected = function.apply(*inputs, *options)
        upstream = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, inputs, upstream)
        actual = Fused.apply(*inputs)
        assert_close(actual, expected, torch.bfloat16, rtol=0, atol=0)
        pending.append((inputs, actual, upstream, expected_grads))

    for inputs, actual, upstream, expected_grads in reversed(pending):
        actual_grads = torch.autograd.grad(actual, inputs, upstream)
        assert_close(actual_grads, expected_grads, torch.bfloat16, rtol=0, atol=0)
    assert_repeatable(Fused.apply, pending[0][0])


@pytest.mark.api("functions.flash_attention")
@pytest.mark.accuracy
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_flash_attention_runtime_policy(accuracy_backend, compiled, phase, deterministic):
    implementation, target, device = accuracy_backend
    if implementation != "torch_npu":
        pytest.skip("torch_npu runtime determinism contract")
    preload("flash_attention", implementation=implementation, target=target)
    inputs = _make_attention_inputs(device, torch.bfloat16, 128)
    cu = torch.tensor([0, 128], device=device, dtype=torch.int32)

    def run(q, k, v):
        # Torch 2.13 Dynamo rejects duplicate Tensor inputs to Function.apply.
        # A zero-copy view keeps Q/K offsets equal without duplicating storage.
        return F.flash_attention(q, k, v, cu, cu.view_as(cu), 128, 128, causal=True, implementation=implementation)

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # AOTAutograd forbids enabling determinism between a compiled forward
        # and its backward. Warm under the stricter policy, then optionally relax it.
        torch.use_deterministic_algorithms(True)
        call = compile_fullgraph(run) if compiled else run
        # Warm both graphs before changing the runtime policy.
        torch.autograd.grad(call(*inputs).sum(), inputs)
        output = call(*inputs) if phase == "backward" else None
        torch.use_deterministic_algorithms(deterministic)
        if phase == "forward":
            output = call(*inputs)
        grads = torch.autograd.grad(output.sum(), inputs)
        assert torch.isfinite(output).all()
        assert all(torch.isfinite(grad).all() for grad in grads)
        assert torch.are_deterministic_algorithms_enabled() == deterministic
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch._dynamo.reset()
