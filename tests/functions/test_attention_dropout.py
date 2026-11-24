"""Dropout mask replay and gradients, including packed tails and GQA."""

from types import SimpleNamespace

import pytest
import torch

from torch.utils.checkpoint import checkpoint

from mojo_opset import Target
from mojo_opset import functions as F
from mojo_opset.functions._dispatch import resolve_implementation


def swa_dropout_reference(q, k, v, cq, ck, mask, p, window, sink, causal=True, interleave=False):
    """Independent FP32 autograd oracle consuming a packed integer keep mask."""
    outputs = []
    heads = q.shape[1]
    for i in range(len(cq) - 1):
        qs, qe = int(cq[i]), int(cq[i + 1])
        ks, ke = int(ck[i]), int(ck[i + 1])
        head_indices = torch.arange(heads) % k.shape[1] if interleave else torch.arange(heads) // (heads // k.shape[1])
        qi = q[qs:qe].transpose(0, 1)
        ki = k[ks:ke, head_indices].transpose(0, 1)
        vi = v[ks:ke, head_indices].transpose(0, 1)
        scores = qi.float() @ ki.float().transpose(-1, -2) * q.shape[-1] ** -0.5
        qr = torch.arange(qe - qs)[:, None] + (ke - ks) - (qe - qs)
        kr = torch.arange(ke - ks)[None, :]
        if causal:
            local = qr <= kr + window if window is not None else False
            prefix = kr < sink if sink is not None else False
            valid = (qr >= kr) & (local | prefix) if window is not None or sink is not None else qr >= kr
            scores = scores.masked_fill(~valid, float("-inf"))
        if causal and window is not None:
            prefix_width = ((sink or 0) + 127) // 128 * 128
            anchor = torch.arange(qe - qs)[:, None] // 128 * 128 + (ke - ks) - (qe - qs)
            start = (anchor - window).clamp_min(0) // 128 * 128
            slots = torch.where(kr < prefix_width, kr, prefix_width + kr - start)
        else:
            slots = kr.expand(qe - qs, -1)
        safe = slots.clamp(0, mask.shape[-1] * 32 - 1)
        packed = mask[torch.arange(qs, qe)[:, None, None], torch.arange(heads)[None, :, None], safe[:, None, :] // 32]
        keep = ((packed.to(torch.int64) >> (safe[:, None, :] % 32)) & 1).transpose(0, 1)
        weights = scores.softmax(-1) * keep / (1 - p)
        outputs.append((weights @ vi.float()).transpose(0, 1))
    return torch.cat(outputs)


@pytest.mark.reference
@pytest.mark.parametrize("p", [-0.1, 1.0, float("nan")])
def test_swa_dropout_invalid_probability(p):
    q = torch.randn(4, 2, 8)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="dropout_p"):
        F.swa(q, q, q, cu, cu, dropout_p=p, implementation="torch")


@pytest.mark.reference
def test_swa_zero_dropout_preserves_rng():
    q = torch.randn(4, 2, 8, requires_grad=True)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    state = torch.get_rng_state()
    F.swa(q, q, q, cu, cu, implementation="torch").sum().backward()
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.reference
def test_swa_mask_fake_layout():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from mojo_opset.kernels.npu_a5_triton._attention_dropout import generate_mask

    with FakeTensorMode():
        q = torch.empty(131072, 6, 128, dtype=torch.bfloat16)
        k = torch.empty(131072, 2, 128, dtype=q.dtype)
        mask = generate_mask(q, k, 0.1, True, 1023, 4)
        assert mask.shape == (131072, 6, 44)
        assert mask.dtype == torch.int32
        assert mask.numel() * mask.element_size() == 132 * 1024**2


@pytest.mark.api("functions.swa")
@pytest.mark.accuracy
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_swa_dropout_accuracy(accuracy_backend, interleave, causal):
    implementation, target, device = accuracy_backend
    implementation = implementation or resolve_implementation("swa", Target.parse(target))
    if implementation != "triton" or Target.parse(target).arch_key != "npu.a5":
        pytest.skip("Dropout currently targets the A5 Triton training provider")
    from mojo_opset.kernels.npu_a5_triton._attention_dropout import generate_mask

    torch.manual_seed(7)
    cq, ck = [torch.tensor(x, dtype=torch.int32, device=device) for x in ([0, 129, 386], [0, 137, 418])]
    q = torch.randn(386, 6, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(418, 2, 128, device=device, dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    dy = torch.randn_like(q)
    kwargs = dict(
        local_window_size=127,
        global_window_size=4,
        output_f32=True,
        dropout_p=0.1,
        is_causal=causal,
        gqa_interleave=interleave,
        implementation=implementation,
    )
    state = torch.npu.get_rng_state()
    mask = generate_mask(q, k, 0.1, causal, 127, 4)
    after_mask = torch.npu.get_rng_state()
    out = F.swa(q, k, v, cq, ck, dropout_mask=mask, **kwargs)
    assert torch.equal(after_mask, torch.npu.get_rng_state())
    inputs = [t.detach().float().cpu().requires_grad_() for t in (q, k, v)]
    ref = swa_dropout_reference(*inputs, cq.cpu(), ck.cpu(), mask.cpu(), 0.1, 127, 4, causal, interleave)
    actual = torch.autograd.grad(out, (q, k, v), dy)
    expected = torch.autograd.grad(ref, inputs, dy.float().cpu())
    for a, b in zip((out, *actual), (ref, *expected)):
        a = a.detach().float().cpu()
        assert (a - b).norm() / b.norm() < 0.02
    assert torch.equal(after_mask, torch.npu.get_rng_state())

    def run(a, b, c):
        return F.swa(a, b, c, cq, ck, **kwargs)

    torch.npu.set_rng_state(state)
    replay = checkpoint(run, q, k, v, use_reentrant=False)
    replay_grads = torch.autograd.grad(replay, (q, k, v), dy)
    for a, b in zip((out, *actual), (replay, *replay_grads)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.api("functions.flash_attention")
@pytest.mark.accuracy
@pytest.mark.parametrize("p", [0.0, 0.01, 0.1])
def test_full_dropout_native_and_checkpoint(accuracy_backend, p):
    implementation, target, device = accuracy_backend
    implementation = implementation or resolve_implementation("flash_attention", Target.parse(target))
    if implementation != "torch_npu" or torch.device(device).type != "npu":
        pytest.skip("CANN training provider only")
    import torch_npu

    from mojo_opset.kernels.npu_torch_npu.attention import _mask

    q = torch.randn(386, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(386, 2, 128, device=device, dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    cu = torch.tensor([0, 129, 386], dtype=torch.int32, device=device)
    dy = torch.randn_like(q)
    mask, mode = _mask(None, True, q.device)
    state = torch.npu.get_rng_state()
    ref = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        4,
        input_layout="TND",
        atten_mask=mask,
        scale=128**-0.5,
        keep_prob=1 - p,
        actual_seq_qlen=(129, 386),
        actual_seq_kvlen=(129, 386),
        sparse_mode=mode,
    )[0]
    rg = torch.autograd.grad(ref, (q, k, v), dy)

    def run(a, b, c):
        return F.flash_attention(a, b, c, cu, cu, 257, 257, dropout_p=p, causal=True, implementation=implementation)

    torch.npu.set_rng_state(state)
    out = run(q, k, v)
    grads = torch.autograd.grad(out, (q, k, v), dy)
    torch.npu.set_rng_state(state)
    replay = checkpoint(run, q, k, v, use_reentrant=False)
    replay_grads = torch.autograd.grad(replay, (q, k, v), dy)
    for actual, expected, recomputed in zip((out, *grads), (ref, *rg), (replay, *replay_grads)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, recomputed, rtol=0, atol=0)


@pytest.mark.reference
def test_swa_fused_context_without_autograd_metadata():
    from mojo_opset.functions.attention import SwaFunction

    q, k, v = [torch.randn(7, 2, 8) for _ in range(3)]
    cu = torch.tensor([0, 7], dtype=torch.int32)
    ctx = SimpleNamespace()
    ctx.save_for_backward = lambda *xs: setattr(ctx, "saved_tensors", xs)
    out = SwaFunction.forward(ctx, q, k, v, cu, cu, True, 3, 1, 8**-0.5, False, True, "torch")
    grads = SwaFunction.backward(ctx, torch.ones_like(out))[:3]
    assert all(g.shape == q.shape and torch.isfinite(g).all() for g in grads)


@pytest.mark.reference
@pytest.mark.parametrize(
    "q_len,kv_len,pattern",
    [
        (1031, 513, "causal"),
        (1031, 1031, "swa"),
        (1031, 513, "disjoint"),
        (1031, 513, "empty_rows"),
        (257, 257, "all_empty"),
        (512, 512, "all_full"),
    ],
)
def test_flex_dropout_layout_and_dkv_schedule_match_reference(q_len, kv_len, pattern):
    from mojo_opset.kernels.npu_a5_triton._attention_dropout import _build_dkv_split_schedule
    from mojo_opset.utils.attention_dropout import _DropoutLayout
    from mojo_opset.utils.flex_attention_mask import create_flex_block_mask

    patterns = {
        "all_empty": lambda qi, ki: (qi < 0) & (ki < 0),
        "all_full": lambda qi, ki: (qi >= 0) & (ki >= 0),
        "swa": lambda qi, ki: (qi >= ki) & ((qi - ki) <= 300),
        "disjoint": lambda qi, ki: ((ki < qi // 2) & (ki < 140)) | ((ki >= qi // 2 + 130) & (ki < qi // 2 + 280)),
        "empty_rows": lambda qi, ki: (ki <= qi // 2) & (qi % 7 != 0),
        "causal": lambda qi, ki: ki <= qi // 2,
    }
    bm = create_flex_block_mask(
        lambda b, h, qi, ki: patterns[pattern](qi, ki), Q_LEN=q_len, KV_LEN=kv_len, device="cpu"
    )

    def ref_rows(prefix):
        counts = getattr(bm, f"{prefix}_num_blocks")[0, 0].tolist()
        idx = getattr(bm, f"{prefix}_indices")[0, 0].tolist()
        f_counts = getattr(bm, f"full_{prefix}_num_blocks", None)
        f_idx = getattr(bm, f"full_{prefix}_indices", None)
        rows = [[(c, False) for c in idx[r][: counts[r]]] for r in range(len(counts))]
        if f_counts is not None:
            for r, fc in enumerate(f_counts[0, 0].tolist()):
                rows[r].extend((c, True) for c in f_idx[0, 0, r][:fc].tolist())
        return rows

    nq, nk = (q_len + 127) // 128, (kv_len + 127) // 128
    expected_table = torch.full((nq, nk), -1, dtype=torch.int32)
    slot = 0
    for qb, entries in enumerate(ref_rows("kv")):
        for kb, _ in entries:
            expected_table[qb, kb] = slot
            slot += 1

    layout, reverse_counts = _DropoutLayout.from_block_mask(bm, q_len, kv_len)
    assert layout.num_active == slot
    assert torch.equal(layout.block_table, expected_table)
    ref_rev = ref_rows("q")
    for chunk_size in (None, 0, 1, 16):
        for balance in (False, True):
            assert _build_dkv_split_schedule(
                reverse_counts, 4, 24, chunk_size, balance_moderate_rows=balance
            ) == _build_dkv_split_schedule(ref_rev, 4, 24, chunk_size, balance_moderate_rows=balance)


@pytest.mark.reference
def test_flex_dropout_layout_validation_and_cache(monkeypatch):
    import gc
    import weakref

    from torch.nn.attention.flex_attention import BlockMask
    from torch.nn.attention.flex_attention import noop_mask

    from mojo_opset.kernels.npu_a5_triton import _attention_dropout as a5_dropout
    from mojo_opset.utils import attention_dropout
    from mojo_opset.utils.flex_attention_mask import create_flex_block_mask

    t = lambda x: torch.tensor(x, dtype=torch.int32)

    def make(**overrides):
        fields = dict(
            seq_lengths=(256, 256),
            kv_num_blocks=t([[[1, 1]]]),
            kv_indices=t([[[[0, 0], [1, 0]]]]),
            full_kv_num_blocks=None,
            full_kv_indices=None,
            q_num_blocks=t([[[1, 1]]]),
            q_indices=t([[[[0, 0], [1, 0]]]]),
            full_q_num_blocks=None,
            full_q_indices=None,
            BLOCK_SIZE=(128, 128),
            mask_mod=noop_mask,
        )
        fields.update(overrides)
        return BlockMask(**fields)

    invalid_cases = [
        (make(), 0, 256, "positive"),
        (make(), 384, 256, "grid"),
        (make(kv_num_blocks=torch.ones((2, 1, 2), dtype=torch.int32)), 256, 256, "batch/head-independent"),
        (make(kv_indices=t([[[[2, 0], [1, 0]]]])), 256, 256, "Invalid or duplicate"),
        (make(kv_num_blocks=t([[[2, 1]]]), kv_indices=t([[[[0, 0], [1, 0]]]])), 256, 256, "Invalid or duplicate"),
        (
            make(
                full_kv_num_blocks=t([[[1, 0]]]),
                full_kv_indices=t([[[[0], [0]]]]),
                full_q_num_blocks=t([[[1, 0]]]),
                full_q_indices=t([[[[0], [0]]]]),
            ),
            256,
            256,
            "Invalid or duplicate",
        ),
        (make(q_indices=t([[[[1, 0], [0, 0]]]])), 256, 256, "disagree"),
        (
            make(
                kv_num_blocks=t([[[0, 1]]]),
                full_kv_num_blocks=t([[[1, 0]]]),
                full_kv_indices=t([[[[0], [0]]]]),
                full_q_num_blocks=t([[[0, 0]]]),
                full_q_indices=t([[[[0], [0]]]]),
            ),
            256,
            256,
            "disagree",
        ),
    ]
    for bad_mask, ql, kl, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            attention_dropout._DropoutLayout.from_block_mask(bad_mask, ql, kl)

    bm = create_flex_block_mask(
        lambda b, h, qi, ki: (qi >= ki) & ((qi - ki) <= 256), Q_LEN=512, KV_LEN=512, device="cpu"
    )
    q, k = torch.randn(1, 4, 512, 128, dtype=torch.bfloat16), torch.randn(1, 2, 512, 128, dtype=torch.bfloat16)
    builds = 0
    orig_build = attention_dropout._DropoutLayout._build

    def count_build(*args):
        nonlocal builds
        builds += 1
        return orig_build(*args)

    monkeypatch.setattr(attention_dropout._DropoutLayout, "_build", count_build)
    monkeypatch.setattr(a5_dropout, "get_num_cores", lambda kind: 24)

    s1 = attention_dropout.create_flex_dropout_mask(q, k, bm, 0.1, bitpacked=True)
    s2 = attention_dropout.create_flex_dropout_mask(q, k, bm, 0.0, bitpacked=False)
    split1 = a5_dropout.allocate_dropout_mask(q, k, bm, 0.1, bitpacked=True, dkv_chunk_size=1)
    split2 = a5_dropout.allocate_dropout_mask(q, k, bm, 0.1, bitpacked=False, dkv_chunk_size=1)
    unsplit = a5_dropout.allocate_dropout_mask(q, k, bm, 0.1, bitpacked=True, dkv_chunk_size=0)
    assert builds == 1
    assert s1.layout.block_table.data_ptr() == s2.layout.block_table.data_ptr() == split1.layout.block_table.data_ptr()
    assert split1.layout.dkv_schedule is not None and unsplit.layout.dkv_schedule is None
    assert split1.layout.dkv_schedule.data_ptr() == split2.layout.dkv_schedule.data_ptr()
    assert split1.layout.dkv_num_tasks > 0

    bm.kv_num_blocks.add_(0)
    s3 = attention_dropout.create_flex_dropout_mask(q, k, bm, 0.1, bitpacked=True)
    assert builds == 2 and torch.equal(s3.layout.block_table, s1.layout.block_table)

    gc.disable()
    try:
        tmp = create_flex_block_mask(lambda b, h, qi, ki: qi >= ki, Q_LEN=256, KV_LEN=256, device="cpu")
        ref = weakref.ref(tmp)
        st = attention_dropout.create_flex_dropout_mask(
            torch.empty(1, 2, 256, 128), torch.empty(1, 2, 256, 128), tmp, 0.0, bitpacked=True
        )
        del st, tmp
        assert ref() is None
    finally:
        gc.enable()


def _philox4x32_reference(seed, counters):
    mask = (1 << 32) - 1
    c0, c1 = counters & mask, (counters >> 32) & mask
    c2, c3 = torch.zeros_like(counters), torch.zeros_like(counters)
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        a, b = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((b >> 32) & mask) ^ c1 ^ k0, b & mask, ((a >> 32) & mask) ^ c3 ^ k1, a & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return torch.stack((c0, c1, c2, c3), dim=-1).to(torch.uint32)


def _triton_packed_mask_reference(num_words, p, seed):
    threshold = int((1.0 - p) * 255.0)
    blocks = (num_words + 1023) // 1024
    raw = _philox4x32_reference(seed, torch.arange(blocks * 2048, dtype=torch.int64)).view(torch.uint8)
    bits = (raw.to(torch.int32) <= threshold).reshape(blocks * 2, 8, 16, 128)
    packed = (bits.to(torch.int32) << torch.arange(8).reshape(1, 8, 1, 1)).sum(1).to(torch.uint8)
    return packed.view(torch.int32).flatten()[:num_words]


@pytest.mark.api("functions.swa")
@pytest.mark.accuracy
@pytest.mark.parametrize("num_words", [1, 15, 16, 17, 1023, 1024, 1025, 65537])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.1, 0.5, 1.0 - 1e-8])
def test_swa_triton_dropout_mask_bitwise_and_tail(accuracy_backend, num_words, p):
    implementation, target, device = accuracy_backend
    implementation = implementation or resolve_implementation("swa", Target.parse(target))
    if implementation != "triton" or Target.parse(target).arch_key != "npu.a5":
        pytest.skip("A5 Triton SWA dropout mask generator only")
    from mojo_opset.kernels._npu_triton_utils import get_num_cores
    from mojo_opset.kernels.npu_a5_triton._attention_dropout import _fill_packed_mask
    from mojo_opset.kernels.npu_a5_triton._attention_dropout import generate_mask

    key = 0x7FFFFFFF9ABCDEF0
    seed = torch.tensor([key & 0xFFFFFFFF, key >> 32], dtype=torch.int64).to(device=device, dtype=torch.int32)
    threshold = int((1.0 - p) * 255)
    buf = torch.full((num_words + 64,), 0x12345678, device=device, dtype=torch.int32)
    grid = (min(get_num_cores("vector"), (num_words + 1023) // 1024),)
    _fill_packed_mask[grid](buf[:num_words], seed, num_words, threshold)
    torch.testing.assert_close(buf[:num_words].cpu(), _triton_packed_mask_reference(num_words, p, key), rtol=0, atol=0)
    assert torch.all(buf[num_words:].cpu() == 0x12345678)

    if num_words == 1025 and p == 0.1:
        q = torch.empty(1025, 1, 128, device=device, dtype=torch.bfloat16)
        k = torch.empty(64, 1, 128, device=device, dtype=torch.bfloat16)
        state = torch.npu.get_rng_state()
        explicit = generate_mask(q, k, p, False, None, None, seed)
        assert torch.equal(state, torch.npu.get_rng_state())
        torch.testing.assert_close(
            explicit.flatten().cpu(), _triton_packed_mask_reference(q.shape[0] * 2, p, key), rtol=0, atol=0
        )
        first = generate_mask(q, k, p, False, None, None)
        torch.npu.set_rng_state(state)
        torch.testing.assert_close(first, generate_mask(q, k, p, False, None, None), rtol=0, atol=0)


@pytest.mark.api("functions.flex_attention_v2")
@pytest.mark.accuracy
@pytest.mark.parametrize("has_full,nq,nk", [(True, 1100, 128), (False, 19, 13), (True, 16, 16)])
def test_a5_triton_dropout_block_table_accuracy(accuracy_backend, has_full, nq, nk):
    implementation, target, device = accuracy_backend
    implementation = implementation or resolve_implementation("flex_attention_v2", Target.parse(target))
    if implementation != "triton" or Target.parse(target).arch_key != "npu.a5":
        pytest.skip("A5 Triton sparse dropout block_table builder only")
    from torch.nn.attention.flex_attention import BlockMask
    from torch.nn.attention.flex_attention import _dense_to_ordered
    from torch.nn.attention.flex_attention import noop_mask

    from mojo_opset.kernels.npu_a5_triton._attention_dropout import build_dropout_layout_entry
    from mojo_opset.utils.attention_dropout import _DropoutLayout

    qi, ki = torch.arange(nq, device=device)[:, None], torch.arange(nk, device=device)[None, :]
    partial = ((qi // max(nq // nk, 1)) == ki) & (qi % 7 != 0)
    full = (((qi // max(nq // nk, 1)) > ki) & (((qi // max(nq // nk, 1)) - ki) <= 8)) if has_full else None
    kv_num, kv_idx = _dense_to_ordered(partial[None, None])
    q_num, q_idx = _dense_to_ordered(partial.T[None, None].contiguous())
    fkv_num, fkv_idx = _dense_to_ordered(full[None, None]) if has_full else (None, None)
    fq_num, fq_idx = _dense_to_ordered(full.T[None, None].contiguous()) if has_full else (None, None)
    bm_npu = BlockMask(
        (nq * 128, nk * 128), kv_num, kv_idx, fkv_num, fkv_idx, q_num, q_idx, fq_num, fq_idx, (128, 128), noop_mask
    )
    bm_cpu = BlockMask(
        (nq * 128, nk * 128),
        kv_num.cpu(),
        kv_idx.cpu(),
        None if fkv_num is None else fkv_num.cpu(),
        None if fkv_idx is None else fkv_idx.cpu(),
        q_num.cpu(),
        q_idx.cpu(),
        None if fq_num is None else fq_num.cpu(),
        None if fq_idx is None else fq_idx.cpu(),
        (128, 128),
        noop_mask,
    )
    actual = build_dropout_layout_entry(bm_npu, nq, nk)
    expected_layout, expected_rev = _DropoutLayout.from_block_mask(bm_cpu, nq * 128, nk * 128)
    assert actual.num_active == expected_layout.num_active
    assert actual.reverse_counts == expected_rev
    assert torch.equal(actual.block_table.cpu(), expected_layout.block_table)

    if nq == 16:
        from mojo_opset.utils.attention_dropout import create_flex_dropout_mask
        from mojo_opset.utils.flex_attention_mask import create_flex_block_mask

        verified_bm = create_flex_block_mask(
            lambda b, h, qi, ki: (qi >= ki) & ((qi - ki) <= 256), Q_LEN=2048, KV_LEN=2048, device=device
        )
        q = torch.empty(1, 2, 2048, 128, device=device, dtype=torch.bfloat16)
        state = create_flex_dropout_mask(q, q, verified_bm, 0.1, bitpacked=True, seed=7)
        ref_bm = create_flex_block_mask(
            lambda b, h, qi, ki: (qi >= ki) & ((qi - ki) <= 256), Q_LEN=2048, KV_LEN=2048, device="cpu"
        )
        ref_layout, _ = _DropoutLayout.from_block_mask(ref_bm, 2048, 2048)
        assert torch.equal(state.layout.block_table.cpu(), ref_layout.block_table)

