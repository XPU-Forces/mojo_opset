"""Native 8-bit Philox decisions packed directly into a 1-bit keep mask."""

import pytest
import torch

from mojo_opset import Target
from mojo_opset.kernels.npu_a5_native.attention_dropout import generate_mask


def philox4x32(seed, counters):
    mask = (1 << 32) - 1
    c0, c1 = counters & mask, (counters >> 32) & mask
    c2, c3 = torch.zeros_like(counters), torch.zeros_like(counters)
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        a, b = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((b >> 32) & mask) ^ c1 ^ k0, b & mask, ((a >> 32) & mask) ^ c3 ^ k1, a & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return torch.stack((c0, c1, c2, c3), dim=-1).to(torch.uint32)


def packed_mask_reference(num_words, p, seed):
    packets = (num_words + 15) // 16
    words = philox4x32(seed, torch.arange(packets * 64, dtype=torch.int64)).to(torch.int64).flatten()
    halves = torch.stack((words & 65535, words >> 16), -1)
    draws = halves.reshape(packets, 2, 2, 128).transpose(2, 3).reshape(-1)
    keep = (draws[: num_words * 32] & 255) <= int((1.0 - p) * 255)
    return (keep.reshape(num_words, 32).to(torch.int64) * (1 << torch.arange(32))).sum(-1).to(torch.int32)


@pytest.mark.reference
def test_native_dropout_fake_layout():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(1)
        out = generate_mask(q, 131072 * 6 * 44, 0.1)
        assert out.dtype == torch.int32
        assert out.numel() * out.element_size() == 132 * 1024**2


@pytest.mark.accuracy
@pytest.mark.api("functions.swa")
@pytest.mark.parametrize("num_words", [1, 15, 16, 17, 1023, 1024, 1025, 65537])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.1, 0.5, 1.0 - 1e-8])
def test_native_dropout_bitwise(accuracy_backend, num_words, p):
    _, target, device = accuracy_backend
    if Target.parse(target).arch_key != "npu.a5":
        pytest.skip("A5 native mask generator only")
    key = 0x7FFFFFFF9ABCDEF0
    seed = torch.tensor([key & 0xFFFFFFFF, key >> 32], dtype=torch.int64).to(device=device, dtype=torch.int32)
    q = torch.empty(1, device=device, dtype=torch.bfloat16)
    state = torch.npu.get_rng_state()
    actual = generate_mask(q, num_words, p, seed)
    torch.testing.assert_close(actual.cpu(), packed_mask_reference(num_words, p, key), rtol=0, atol=0)
    assert torch.equal(state, torch.npu.get_rng_state())


@pytest.mark.accuracy
@pytest.mark.api("functions.swa")
def test_native_dropout_tail_and_rng_replay(accuracy_backend):
    _, target, device = accuracy_backend
    if Target.parse(target).arch_key != "npu.a5":
        pytest.skip("A5 native mask generator only")
    q = torch.empty(1, device=device, dtype=torch.bfloat16)
    state = torch.npu.get_rng_state()
    first = generate_mask(q, 1025, 0.1)
    torch.npu.set_rng_state(state)
    torch.testing.assert_close(first, generate_mask(q, 1025, 0.1), rtol=0, atol=0)
    seed = torch.tensor([123, 456], device=device, dtype=torch.int32)
    for count in (1, 15, 17, 1023, 1025):
        buf = torch.full((count + 64,), 0x12345678, device=device, dtype=torch.int32)
        torch.ops.mojo_native_attention_dropout.mask(seed, buf[:count], 229)
        assert torch.all(buf[count:].cpu() == 0x12345678)
