from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset.kernels.torch.flex_attention import _block_mask_to_dense
from mojo_opset.utils.flex_attention_mask import create_flex_block_mask
from tests._checks import assert_close
from tests._checks import assert_repeatable


def _require_flex(backend):
    impl, _, device = backend
    if device != "npu" or impl not in (None, "triton", "torch"):
        pytest.skip("The migrated FlexAttention providers are NPU Triton and torch")
    return impl, device


CASES = [(2, 2, 96, 112, "full"), (2, 2, 197, 173, "window"), (4, 2, 65, 81, "causal")]
CASES += [(2, 2, length, length, "causal") for length in (1, 127, 128, 129)]


def make_inputs(case, device):
    qheads, kheads, qlen, klen, pattern = case

    def mask_mod(b, h, qi, ki):
        if pattern == "full":
            return (qi >= 0) & (ki >= 0)
        if pattern == "window":
            return (qi >= ki) & (qi - ki <= 17)
        return qi >= ki

    mask = create_flex_block_mask(mask_mod, Q_LEN=qlen, KV_LEN=klen, device=device, BLOCK_SIZE=128, stripe_q_blocks=1)
    inputs = (
        torch.randn(1, qheads, qlen, 32, dtype=torch.bfloat16, device=device, requires_grad=True),
        torch.randn(1, kheads, klen, 32, dtype=torch.bfloat16, device=device, requires_grad=True),
        torch.randn(1, kheads, klen, 32, dtype=torch.bfloat16, device=device, requires_grad=True),
    )
    return inputs, mask, mask_mod


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", CASES)
def test_flex_attention_v2_accuracy(accuracy_backend, case):
    impl, device = _require_flex(accuracy_backend)
    inputs, mask, mask_mod = make_inputs(case, device)
    qidx, kidx = torch.arange(case[2], device=device)[:, None], torch.arange(case[3], device=device)[None, :]
    assert_close(_block_mask_to_dense(mask, case[2], case[3], device), mask_mod(0, 0, qidx, kidx), rtol=0, atol=0)
    refs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    actual = F.flex_attention_v2(*inputs, block_mask=mask, enable_gqa=True, implementation=impl)
    expected = F.flex_attention_v2(*refs, block_mask=mask, enable_gqa=True, implementation="torch")
    assert_close(actual, expected, torch.bfloat16, rtol=2e-2, atol=2e-2)
    upstream = torch.randn_like(actual)
    for a, e in zip(torch.autograd.grad(actual, inputs, upstream), torch.autograd.grad(expected, refs, upstream)):
        assert_close(a, e, torch.bfloat16, rtol=2e-2, atol=5e-2)


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", [CASES[0], CASES[2]])
def test_padding_after_nan_matmul(accuracy_backend, case):
    impl, target, device = accuracy_backend
    if target != "npu.a2" or impl not in (None, "triton"):
        pytest.skip("Ascend Triton padded dP regression")
    # Make stale matmul state observable instead of relying on a previous CI job.
    poison = torch.full((2048, 2048), float("nan"), device=device, dtype=torch.bfloat16)
    ones = torch.ones_like(poison)
    try:
        torch.mm(poison, ones)
        torch.npu.synchronize()
        torch.manual_seed(0)
        test_flex_attention_v2_accuracy(accuracy_backend, case)
    finally:
        poison.zero_()
        torch.mm(poison, ones)
        torch.npu.synchronize()


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.bitwise
@pytest.mark.parametrize("case", CASES)
def test_bitwise(accuracy_backend, case):
    impl, device = _require_flex(accuracy_backend)
    inputs, mask, _ = make_inputs(case, device)
    assert_repeatable(partial(F.flex_attention_v2, block_mask=mask, enable_gqa=True, implementation=impl), inputs)


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_a5_delta_strides_and_tail(accuracy_backend, dtype, strided):
    impl, target, device = accuracy_backend
    if not (target == "npu.a5" or target.startswith("npu.a5.")) or impl not in (None, "triton"):
        pytest.skip("Ascend A5 fused backward delta")
    from mojo_opset.kernels.npu_a5_triton.flex_attention_v2 import _compute_delta

    shape = (2, 257, 3, 128) if strided else (2, 3, 257, 128)
    output = torch.randn(shape, dtype=dtype, device=device)
    upstream = torch.randn_like(output)
    if strided:
        output = output.transpose(1, 2)
        upstream = upstream.transpose(1, 2)
    expected = (output.float() * upstream.float()).sum(dim=-1)
    actual = _compute_delta(output, upstream)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.accuracy
def test_a5_multiple_kv_tiles_and_query_heads(accuracy_backend):
    impl, target, device = accuracy_backend
    if not (target == "npu.a5" or target.startswith("npu.a5.")) or impl not in (None, "triton"):
        pytest.skip("Ascend A5 KV tile ownership and GQA accumulation")
    from mojo_opset.kernels.npu_a5_triton.flex_attention_v2 import flex_attention_bwd_impl
    from mojo_opset.kernels.npu_a5_triton.flex_attention_v2 import flex_attention_fwd_impl

    torch.manual_seed(37)
    q = torch.randn(2, 6, 513, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 2, 385, 128, dtype=q.dtype, device=device)
    v = torch.randn_like(k)

    def mask_mod(b, h, qi, ki):
        return (qi >= ki) & ((qi - ki) <= 149)

    mask = create_flex_block_mask(mask_mod, Q_LEN=513, KV_LEN=385, device=device, BLOCK_SIZE=128)
    output, lse = flex_attention_fwd_impl(q, k, v, mask)
    upstream = torch.randn_like(output)
    grads = flex_attention_bwd_impl(upstream, q, k, v, output, lse, mask)
    refs = tuple(x.detach().cpu().double().requires_grad_(True) for x in (q, k, v))
    qi = torch.arange(513)[:, None]
    ki = torch.arange(385)[None, :]
    ref_output = torch.nn.functional.scaled_dot_product_attention(
        refs[0], refs[1].repeat_interleave(3, 1), refs[2].repeat_interleave(3, 1),
        attn_mask=mask_mod(0, 0, qi, ki),
    )
    ref_grads = torch.autograd.grad(ref_output, refs, upstream.cpu().double())
    for actual, expected in zip((output, *grads), (ref_output, *ref_grads)):
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-2, atol=5e-2)
