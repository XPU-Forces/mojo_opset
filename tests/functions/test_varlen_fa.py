from contextlib import nullcontext
from itertools import product

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

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
def test_varlen_fa(varlen_backend, dtype, causal, interleave, lengths, heads, case):
    implementation, device = varlen_backend
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
