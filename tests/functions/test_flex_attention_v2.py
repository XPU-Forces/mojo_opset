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
        refs[0],
        refs[1].repeat_interleave(3, 1),
        refs[2].repeat_interleave(3, 1),
        attn_mask=mask_mod(0, 0, qi, ki),
    )
    ref_grads = torch.autograd.grad(ref_output, refs, upstream.cpu().double())
    for actual, expected in zip((output, *grads), (ref_output, *ref_grads)):
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-2, atol=5e-2)


# Representative 8:1 KV cases cover both storage formats for each core mask.
# Fold in GQA, multiple batches, head dimensions, tails and split-Q coverage.
_DROPOUT_CASES = [
    pytest.param((1031, "causal", 128, None), 0.1, False, id="causal-uint8"),
    pytest.param((4097, "causal", 128, 16), 0.01, True, id="causal-packed-long-split"),
    pytest.param((1031, "disjoint", 128, 1), 0.01, False, id="disjoint-uint8-split"),
    pytest.param((1031, "disjoint", 128, 1), 0.1, True, id="disjoint-packed-split"),
    pytest.param((1031, "empty_rows", 64, 1), 0.1, False, id="empty-rows-uint8-dim64"),
    pytest.param((1031, "empty_rows", 128, 1), 0.01, True, id="empty-rows-packed"),
    pytest.param((257, "all_empty", 128, None), 0.01, False, id="all-empty"),
]


def _require_a5_dropout(backend):
    impl, target, device = backend
    if not (target == "npu.a5" or target.startswith("npu.a5.")) or impl not in (None, "triton", "torch"):
        pytest.skip("Prepared dropout is supported by A5 Triton and the Torch reference")
    return impl, device


def _dropout_inputs(case, device):
    qlen, pattern, dim, _ = case
    klen = (qlen + 7) // 8
    prefix, gap, window = (max(1, (klen * factor + 136) // 137) for factor in (8, 4, 4))

    def mask_mod(b, h, qi, ki):
        end = (qi + 1) // 8
        if pattern == "all_empty":
            return (qi < 0) & (ki < 0)
        if pattern == "disjoint":
            return ((ki < end) & (ki < prefix)) | ((ki >= end + gap) & (ki < end + gap + window))
        if pattern == "empty_rows":
            return (ki < end) & (qi % 7 != 0)
        return ki < end

    mask = create_flex_block_mask(mask_mod, Q_LEN=qlen, KV_LEN=klen, device=device, BLOCK_SIZE=128)
    batch = 2 if pattern == "disjoint" else 1
    q = torch.randn(batch, 6, qlen, dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn(batch, 2, klen, dim, dtype=q.dtype, device=device, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    return (q, k, v), mask, mask_mod


def _dense_dropout_oracle(inputs, mask_mod, state):
    # Independent CPU FP64 attention and CPU bit unpacking; no Triton/Torch
    # provider helpers participate in this expected result.
    refs = tuple(x.detach().cpu().double().requires_grad_(True) for x in inputs)
    q, k, v = refs
    threshold = {0.1: 229, 0.01: 252}[state.dropout_p]
    keep_probability = 1 - torch.tensor(state.dropout_p, dtype=torch.float32)
    drop_scale = float(keep_probability.reciprocal())
    assert state.threshold == threshold
    assert state.scale == drop_scale
    keep = torch.zeros(q.shape[0], q.shape[1], q.shape[2], k.shape[2], dtype=torch.bool)
    tiles = state.tiles.cpu()
    for qb, row in enumerate(state.layout.block_table.cpu().tolist()):
        for kb, slot in enumerate(row):
            if slot < 0:
                continue
            tile = tiles[:, :, slot]
            if state.bitpacked:
                bits = tile.to(torch.int32).unsqueeze(2) >> torch.arange(8).view(1, 1, 8, 1, 1)
                bits = (bits & 1).reshape(q.shape[0], q.shape[1], 128, 128).bool()
            else:
                bits = tile <= threshold
            qs, ks = slice(qb * 128, min((qb + 1) * 128, q.shape[2])), slice(kb * 128, min((kb + 1) * 128, k.shape[2]))
            keep[:, :, qs, ks] = bits[:, :, : qs.stop - qs.start, : ks.stop - ks.start]
    visible = mask_mod(0, 0, torch.arange(q.shape[2])[:, None], torch.arange(k.shape[2])[None, :])
    group = q.shape[1] // k.shape[1]
    scores = (q @ k.repeat_interleave(group, 1).transpose(-1, -2)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~visible, -torch.inf)
    probabilities = torch.where(visible.any(-1, keepdim=True), scores, 0).softmax(-1).masked_fill(~visible, 0)
    output = ((probabilities * keep) @ v.repeat_interleave(group, 1)) * drop_scale
    return refs, output, visible


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.accuracy
@pytest.mark.parametrize("case,probability,bitpacked", _DROPOUT_CASES)
def test_dropout_accuracy(accuracy_backend, case, probability, bitpacked):
    from mojo_opset.utils.attention_dropout import create_flex_dropout_mask

    impl, device = _require_a5_dropout(accuracy_backend)
    torch.manual_seed(53)
    inputs, mask, mask_mod = _dropout_inputs(case, device)
    state = create_flex_dropout_mask(
        *inputs[:2], mask, probability, bitpacked=bitpacked, seed=53, dkv_chunk_size=case[3]
    )
    actual = F.flex_attention_v2(*inputs, block_mask=mask, enable_gqa=True, dropout=state, implementation=impl)
    upstream = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, inputs, upstream)
    refs, expected, visible = _dense_dropout_oracle(inputs, mask_mod, state)
    expected_grads = torch.autograd.grad(expected, refs, upstream.cpu().double())
    for index, (value, oracle) in enumerate(zip((actual, *grads), (expected, *expected_grads))):
        assert value.dtype == inputs[0].dtype
        value = value.detach().cpu().double()
        assert_close(value, oracle, rtol=0.04, atol=0.04)
        assert float((value - oracle).norm() / oracle.norm().clamp_min(1e-30)) < 0.02
        empty = ~visible.any(-1) if index < 2 else ~visible.any(0)
        assert not value[:, :, empty].count_nonzero()


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.bitwise
@pytest.mark.parametrize("probability,bitpacked", [(0.1, False), (0.01, True)], ids=["uint8", "packed"])
def test_dropout_bitwise(accuracy_backend, probability, bitpacked):
    from mojo_opset.utils.attention_dropout import create_flex_dropout_mask

    impl, device = _require_a5_dropout(accuracy_backend)
    inputs, mask, _ = _dropout_inputs((1031, "disjoint", 128, 1), device)
    state = create_flex_dropout_mask(*inputs[:2], mask, probability, bitpacked=bitpacked, seed=53, dkv_chunk_size=1)
    assert_repeatable(
        partial(F.flex_attention_v2, block_mask=mask, enable_gqa=True, dropout=state, implementation=impl), inputs
    )


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.bitwise
def test_zero_dropout_matches_plain_attention(accuracy_backend):
    from mojo_opset.utils.attention_dropout import create_flex_dropout_mask
    from tests._checks import assert_bitwise_equal

    impl, device = _require_a5_dropout(accuracy_backend)
    inputs, mask, _ = _dropout_inputs((1031, "causal", 128, None), device)
    state = create_flex_dropout_mask(*inputs[:2], mask, 0, bitpacked=True, seed=53)
    first = F.flex_attention_v2(*inputs, block_mask=mask, enable_gqa=True, implementation=impl)
    second = F.flex_attention_v2(*inputs, block_mask=mask, enable_gqa=True, dropout=state, implementation=impl)
    upstream = torch.randn_like(first)
    assert_bitwise_equal(first, second)
    assert_bitwise_equal(torch.autograd.grad(first, inputs, upstream), torch.autograd.grad(second, inputs, upstream))


def _philox_reference(seed, offset, count):
    values, mask = [], (1 << 32) - 1
    for counter in range(offset, offset + count):
        c0, c1, c2, c3 = counter & mask, counter >> 32, 0, 0
        k0, k1 = seed & mask, seed >> 32
        for _ in range(10):
            a, b = 0xD2511F53 * c0, 0xCD9E8D57 * c2
            c0, c1, c2, c3 = (b >> 32) ^ c1 ^ k0, b & mask, (a >> 32) ^ c3 ^ k1, a & mask
            k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
        for word in (c0, c1, c2, c3):
            values.extend((word >> shift) & 255 for shift in (0, 8, 16, 24))
    return torch.tensor(values, dtype=torch.uint8)


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "threshold,tiles,seed,offset",
    [
        pytest.param(229, 1, 53, 0, id="p0.1-single-tile"),
        pytest.param(252, 5, 0x123456789ABCDEF, 2**32 - 7, id="p0.01-tail-counter-carry"),
    ],
)
def test_dropout_rng_matches_integer_oracle(accuracy_backend, threshold, tiles, seed, offset):
    impl, device = _require_a5_dropout(accuracy_backend)
    if impl == "torch":
        pytest.skip("Triton RNG kernel check")
    from mojo_opset.kernels.npu_a5_triton._attention_dropout import _fill_q_planes
    from mojo_opset.kernels.npu_a5_triton._attention_dropout import _fill_random_bytes

    expected = _philox_reference(seed, offset, tiles * 1024)
    raw = torch.empty(tiles * 16384, dtype=torch.uint8, device=device)
    _fill_random_bytes[(1,)](raw.view(torch.int32), seed, offset, raw.numel() // 4, BLOCK=2048)
    assert_close(raw.cpu(), expected, rtol=0, atol=0)
    backing = torch.full((tiles * 2048 + 64,), 165, dtype=torch.uint8, device=device)
    _fill_q_planes[(1,)](backing[:-64].view(torch.int32), seed, offset, tiles, threshold)
    bits = (expected.to(torch.int32) <= threshold).reshape(tiles, 8, 16, 128)
    packed = (bits.to(torch.int32) << torch.arange(8).reshape(1, 8, 1, 1)).sum(1).to(torch.uint8)
    assert_close(backing[:-64].cpu(), packed.flatten(), rtol=0, atol=0)
    assert (backing[-64:] == 165).all()
