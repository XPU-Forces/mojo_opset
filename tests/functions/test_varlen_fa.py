from contextlib import nullcontext
from itertools import product

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._compile import compile_fullgraph

VARLEN_FA_CASES = [
    (([96, 128], [96, 128]), (8, 1)),
    (([129, 257], [513, 1025]), (8, 2)),
    (([1, 7], [2049, 4097]), (5, 3)),
    (([128], [128]), (4, 4)),
    (([129], [2049]), (8, 1)),
]


def make_varlen_fa_inputs(device, dtype, lengths, heads):
    qlens, klens = lengths
    q = torch.randn(sum(qlens), heads[0], 128, dtype=dtype, device=device)
    k = torch.randn(sum(klens), heads[1], 128, dtype=dtype, device=device)
    v = torch.randn_like(k)
    cq, ck = [torch.tensor([0, *ls], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32) for ls in lengths]
    return q, k, v, cq, ck


@pytest.mark.api("functions.varlen_fa_infer")
@pytest.mark.parametrize(
    "dtype,causal,interleave,lengths,heads,case",
    [
        pytest.param(
            dtype, causal, interleave, lengths, heads, "default", id=f"{dtype}-{causal}-{interleave}-{lengths}-{heads}"
        )
        for dtype, causal, interleave, (lengths, heads) in product(
            (torch.bfloat16, torch.float16), (False, True), (False, True), VARLEN_FA_CASES
        )
    ]
    + [
        pytest.param(torch.bfloat16, True, False, ([129], [257]), (5, 3), "stream", id="stream"),
    ]
    + [
        pytest.param(torch.bfloat16, True, False, ([4], [4]), (8, 2), case, id=case)
        for case in ("zero_length", "short_kv", "cpu_cu", "end", "dtype", "scale", "layout")
    ],
)
@pytest.mark.accuracy
def test_varlen_fa(accuracy_backend, dtype, causal, interleave, lengths, heads, case):
    implementation, _, device = accuracy_backend
    if case != "default" and implementation == "torch_reference":
        pytest.skip("native launch contract")
    stream = torch.npu.Stream() if case == "stream" else None
    context = torch.npu.stream(stream) if stream is not None else nullcontext()
    with context:
        q, k, v, cq, ck = make_varlen_fa_inputs(device, dtype, lengths, heads)
        kwargs = dict(is_causal=causal, gqa_interleave=interleave, softmax_scale=0.1 if case == "default" else None)
        if case == "zero_length":
            cq = ck = torch.tensor([0, 0, 4], dtype=torch.int32, device=device)
        elif case == "short_kv":
            k, v = k[:3], v[:3]
            ck = torch.tensor([0, 3], dtype=torch.int32, device=device)
        elif case == "cpu_cu":
            cq, ck = cq.cpu(), ck.cpu()
        elif case == "end":
            cq = torch.tensor([0, 3], dtype=torch.int32, device=device)
        elif case == "dtype":
            k = k.float()
        elif case == "scale":
            kwargs["softmax_scale"] = float("nan")
        elif case == "layout":
            q = q.transpose(0, 1).contiguous().transpose(0, 1)
        inputs = (q, k, v, cq, ck)
        if case not in ("default", "stream"):
            with pytest.raises((RuntimeError, ValueError)):
                F.varlen_fa_infer(*inputs, implementation=implementation, **kwargs)
            return
        actual = F.varlen_fa_infer(*inputs, implementation=implementation, **kwargs)
    if stream is not None:
        stream.synchronize()
    expected = F.varlen_fa_infer(*inputs, implementation="torch_reference", **kwargs)
    assert not actual.requires_grad
    assert_close(actual, expected, dtype)


@pytest.mark.reference
@pytest.mark.parametrize(
    "interleave,expected_heads",
    [
        (False, [0, 0, 1, 1, 2]),
        (True, [0, 1, 2, 0, 1]),
    ],
)
@pytest.mark.accuracy
def test_reference_head_mapping(interleave, expected_heads):
    q = torch.zeros(3, 5, 8)
    k = torch.zeros(4, 3, 8)
    v = torch.arange(3, dtype=q.dtype).view(1, 3, 1).expand_as(k)
    cq, ck = torch.tensor([0, 3], dtype=torch.int32), torch.tensor([0, 4], dtype=torch.int32)
    out = F.varlen_fa_infer(q, k, v, cq, ck, gqa_interleave=interleave, implementation="torch_reference")
    expected = torch.tensor(expected_heads, dtype=q.dtype).view(1, 5, 1).expand_as(q)
    torch.testing.assert_close(out, expected)


@pytest.mark.reference
@pytest.mark.accuracy
def test_causal_alignment():
    q, k = torch.zeros(2, 1, 4), torch.zeros(4, 1, 4)
    v = torch.arange(4, dtype=q.dtype).view(4, 1, 1).expand_as(k)
    out = F.varlen_fa_infer(q, k, v, torch.tensor([0, 2]), torch.tensor([0, 4]), implementation="torch_reference")
    torch.testing.assert_close(out[:, 0, 0], torch.tensor([1.0, 1.5]))


@pytest.mark.reference
@pytest.mark.accuracy
def test_no_autograd():
    q = torch.randn(4, 2, 8, requires_grad=True)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        F.varlen_fa_infer(q, q, q, cu, implementation="torch_reference")
    with torch.no_grad():
        assert not F.varlen_fa_infer(q, q, q, cu, implementation="torch_reference").requires_grad


@pytest.mark.accuracy
@pytest.mark.api("functions.varlen_fa_infer")
def test_compile(accuracy_backend):
    implementation, _, device = accuracy_backend
    if implementation == "torch_reference":
        pytest.skip("native leaf contract")
    preload("varlen_fa_infer", implementation="native")
    q = torch.randn(129, 5, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(257, 3, 128, device=device, dtype=q.dtype)
    v = torch.randn_like(k)
    cq = torch.tensor([0, 129], device=device, dtype=torch.int32)
    ck = torch.tensor([0, 257], device=device, dtype=torch.int32)
    torch.library.opcheck(
        torch.ops.mojo_npu_native_a2.varlen_fa_infer.default,
        (q, k, v, cq, ck, True, 128**-0.5, False),
        test_utils=("test_schema", "test_faketensor"),
    )
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def run(q, k, v, cq, ck):
        return F.varlen_fa_infer(q, k, v, cq, ck, implementation="native")

    try:
        compiled = compile_fullgraph(run, backend=backend)
        torch.testing.assert_close(compiled(q, k, v, cq, ck), run(q, k, v, cq, ck), rtol=0, atol=0)
        assert len(graphs) == 1
        assert any("mojo_npu_native_a2.varlen_fa_infer" in str(node.target) for node in graphs[0].graph.nodes)
    finally:
        torch._dynamo.reset()
