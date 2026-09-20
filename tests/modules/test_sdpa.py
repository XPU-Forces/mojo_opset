import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

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


@pytest.mark.api("modules.Sdpa", ops=["sdpa_infer"])
@pytest.mark.accuracy
@pytest.mark.parametrize("case", SDPA_CASES)
def test_sdpa(accuracy_backend, case):
    impl, _, device = accuracy_backend
    args = make_sdpa_case(case, device)
    actual = modules.Sdpa(case[3] ** -0.5, True, implementation=impl)(*args)
    expected = modules.Sdpa(case[3] ** -0.5, True, implementation="torch_reference")(*args)
    assert_close(actual, expected, args[0].dtype)
