import pytest
import torch

from mojo_opset import functions as F


def make_case(n, ring, dtype, slot_dtype, device):
    blocks = max(1, (n + ring - 1) // ring)
    backing = torch.randn((blocks, 2, ring + 1, 128), device=device, dtype=dtype)
    key = torch.randn((n, 128), device=device, dtype=dtype)
    gate = torch.randn_like(key)
    slots = torch.arange(n, device=device, dtype=slot_dtype)
    slots[1::7] = torch.iinfo(slot_dtype).min
    return backing, key, gate, slots


@pytest.mark.reference
def test_kpool_seed_tail_cache_torch():
    backing = torch.full((1, 2, 8, 128), -7, dtype=torch.float16)
    key = torch.arange(6, dtype=torch.float16)[:, None].expand(-1, 128).contiguous()
    gate = key + 10
    slots = torch.tensor([0, -1, 2, 3, 4, 5], dtype=torch.int64)
    expected = backing.clone()
    for row in (2, 3, 4, 5):
        expected[0, 0, row] = key[row]
        expected[0, 1, row] = gate[row]
    assert F.kpool_seed_tail_cache(backing, key, gate, slots, implementation="torch") is None
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype,slot_dtype", [(torch.bfloat16, torch.int64), (torch.float16, torch.int32)])
@pytest.mark.parametrize("n,ring", [(0, 4), (1, 8), (4, 8), (5, 5), (9, 7), (10, 16), (33, 32), (16384, 16)])
def test_kpool_seed_tail_cache_accuracy(accuracy_backend, dtype, slot_dtype, n, ring):
    implementation, _, device = accuracy_backend
    backing, key, gate, slots = make_case(n, ring, dtype, slot_dtype, "cpu")
    expected = backing.clone()
    F.kpool_seed_tail_cache(expected[:, :, :ring], key, gate, slots, implementation="torch")
    actual = backing.to(device)
    assert F.kpool_seed_tail_cache(
        actual[:, :, :ring], key.to(device), gate.to(device), slots.to(device), implementation=implementation
    ) is None
    torch.testing.assert_close(actual.cpu().view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
