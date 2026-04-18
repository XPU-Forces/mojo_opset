import pytest
import torch

from mojo_opset import MojoMRoPE
from mojo_opset.utils.platform import get_torch_device
from tests.utils import bypass_not_implemented

torch.random.manual_seed(42)


def compute_cos_sin_cache(head_dim, rotary_dim, max_position, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim // 2, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_position, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = freqs.repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


def adjust_mrope_section(mrope_section, actual_rotary_dim):
    mrope_section_adjusted = []
    remaining = actual_rotary_dim // 2
    for i, s in enumerate(mrope_section):
        if i == len(mrope_section) - 1:
            mrope_section_adjusted.append(remaining)
        else:
            adj_s = min(s, remaining)
            mrope_section_adjusted.append(adj_s)
            remaining -= adj_s
    return mrope_section_adjusted


def prepare_test_inputs(num_tokens, n_qh, n_kh, head_dim, mrope_section, device, dtype=torch.float32):
    rotary_dim = sum(mrope_section) * 2
    rope_percentage = rotary_dim / head_dim
    actual_rotary_dim = int(head_dim * rope_percentage)

    mrope_section_adjusted = adjust_mrope_section(mrope_section, actual_rotary_dim)

    positions = torch.randint(0, 1000, (3, num_tokens), device=device, dtype=torch.long)
    cos_cache, sin_cache = compute_cos_sin_cache(head_dim, rotary_dim, 4000, base=10000.0)

    half_head_dim = head_dim // 2
    cos_3d = torch.zeros(3, num_tokens, half_head_dim, device=device, dtype=torch.float32)
    sin_3d = torch.zeros(3, num_tokens, half_head_dim, device=device, dtype=torch.float32)

    half_rotary_dim = actual_rotary_dim // 2
    for dim_idx in range(3):
        pos = positions[dim_idx]
        cos_3d[dim_idx, :, :half_rotary_dim] = cos_cache[pos][:, :half_rotary_dim]
        sin_3d[dim_idx, :, :half_rotary_dim] = sin_cache[pos][:, :half_rotary_dim]

    query = torch.randn(num_tokens, n_qh * head_dim, device=device, dtype=dtype)
    key = torch.randn(num_tokens, n_kh * head_dim, device=device, dtype=dtype)

    return query, key, cos_3d, sin_3d, mrope_section_adjusted


@pytest.mark.parametrize("num_tokens", [1, 16, 32])
@pytest.mark.parametrize("n_qh", [8, 16])
@pytest.mark.parametrize("n_kh", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("mrope_section", [[16, 24, 24], [0, 32, 32]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_interleaved", [False, True])
@bypass_not_implemented
def test_mrope(
    num_tokens,
    n_qh,
    n_kh,
    head_dim,
    mrope_section,
    dtype,
    is_interleaved,
):
    """
    Unified MRoPE test covering all scenarios:
    - num_tokens: token sequence length
    - n_qh/n_kh: query/key heads
    - head_dim: dimension per head
    - mrope_section: T/H/W section configuration
    - dtype: data type
    - is_interleaved: interleaved mode flag
    """
    device = get_torch_device()
    query, key, cos_table, sin_table, mrope_section_adj = prepare_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device, dtype=dtype
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(
        mrope_ref, query, key, cos_table, sin_table, mrope_section_adj, is_interleaved, head_dim=head_dim
    )


@pytest.mark.parametrize("num_tokens", [16, 32])
@pytest.mark.parametrize("n_qh, n_kh", [(16, 8), (8, 4)])
@pytest.mark.parametrize(
    "mrope_section,head_dim",
    [
        ([8, 8, 8], 64),
        ([4, 16, 12], 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_interleaved", [False, True])
@bypass_not_implemented
def test_mrope_nope_dim(
    num_tokens,
    n_qh,
    n_kh,
    mrope_section,
    head_dim,
    dtype,
    is_interleaved,
):
    """
    Test MRoPE with nope_dim > 0 (rotary_dim < head_dim).
    Requires passing head_dim explicitly since it cannot be inferred from cos_table.
    """
    device = get_torch_device()
    query, key, cos_table, sin_table, mrope_section_adj = prepare_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device, dtype=dtype
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(
        mrope_ref, query, key, cos_table, sin_table, mrope_section_adj, is_interleaved, head_dim=head_dim
    )
