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
