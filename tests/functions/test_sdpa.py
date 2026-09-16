from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_accuracy
from tests._checks import assert_close
from tests._checks import assert_repeatable

SDPA_CASES = [(1, 5, 1, 128, 4096, 32), (1, 4, 2, 64, 512, None)]


def generate_diffusion_attention_mask(
    seq_length: int,
    block_size: int,
) -> torch.Tensor:
    total_length = seq_length * 2
    i = torch.arange(total_length).unsqueeze(1)
    j = torch.arange(total_length).unsqueeze(0)
    block_i = i // block_size
    block_j = j // block_size

    same_block = block_i == block_j
    cross = (j >= seq_length) & (i < seq_length) & (((j - seq_length) // block_size) < block_i)
    lower_tri = (i >= seq_length) & (j >= seq_length) & (block_j < block_i)

    return same_block | cross | lower_tri


def make_sdpa_case(case, device):
    b, hq, hk, d, length, block = case
    q = torch.randn(b, hq, length, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, hk, length, d, device=device, dtype=q.dtype)
    v = torch.randn_like(k)
    if block is None:
        q, k, v = [x * 0.25 for x in (q, k, v)]
    mask = None if block is None else generate_diffusion_attention_mask(length // 2, block).to(device)
    return q, k, v, mask


@pytest.mark.api("functions.sdpa_infer")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", SDPA_CASES)
def test_sdpa(accuracy_backend, case):
    impl, _, device = accuracy_backend
    args = make_sdpa_case(case, device)
    opts = dict(scale=case[3] ** -0.5, enable_gqa=True)
    actual = F.sdpa_infer(*args, implementation=impl, **opts)
    expected = F.sdpa_infer(*args, implementation="torch_reference", **opts)
    assert_close(actual, expected, args[0].dtype)


@pytest.mark.api("functions.diffusion_attention")
@pytest.mark.accuracy
@pytest.mark.parametrize("shape", [(1, 4, 2, 256, 64), (1, 4, 2, 512, 64), (1, 5, 1, 512, 128)])
def test_diffusion(accuracy_backend, shape):
    b, hq, hk, length, d = shape
    device = accuracy_backend[2]
    inputs = tuple(torch.randn(b, h, length, d, device=device, dtype=torch.bfloat16) for h in (hq, hk, hk))
    mask = generate_diffusion_attention_mask(length // 2, 32).to(device)
    if length % 512 and accuracy_backend[0] != "torch_reference":
        with pytest.raises(NotImplementedError, match="divisible by 512"):
            F.diffusion_attention(*inputs, mask, scale=d**-0.5, enable_gqa=True, implementation=accuracy_backend[0])
        return
    assert_accuracy(
        partial(F.diffusion_attention, mask=mask, scale=d**-0.5, enable_gqa=True),
        inputs,
        implementation=accuracy_backend[0],
    )


@pytest.mark.api("functions.diffusion_attention")
@pytest.mark.bitwise
@pytest.mark.parametrize("shape", [(1, 4, 2, 256, 64), (1, 4, 2, 512, 64), (1, 5, 1, 512, 128)])
def test_diffusion_bitwise(accuracy_backend, shape):
    impl, _, device = accuracy_backend
    b, hq, hk, length, d = shape
    inputs = tuple(
        torch.randn(b, h, length, d, device=device, dtype=torch.bfloat16, requires_grad=True) for h in (hq, hk, hk)
    )
    mask = generate_diffusion_attention_mask(length // 2, 32).to(device)
    if length % 512 and accuracy_backend[0] != "torch_reference":
        with pytest.raises(NotImplementedError, match="divisible by 512"):
            F.diffusion_attention(*inputs, mask, scale=d**-0.5, enable_gqa=True, implementation=accuracy_backend[0])
        return
    assert_repeatable(
        partial(F.diffusion_attention, mask=mask, scale=d**-0.5, enable_gqa=True, implementation=impl), inputs
    )
