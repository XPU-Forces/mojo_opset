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

    q = torch.randn(num_tokens, n_qh * head_dim, device=device, dtype=dtype)
    k = torch.randn(num_tokens, n_kh * head_dim, device=device, dtype=dtype)

    return q, k, cos_3d, sin_3d, mrope_section_adjusted


@pytest.mark.parametrize("num_tokens", [1, 16, 32])
@pytest.mark.parametrize("n_qh", [8, 16])
@pytest.mark.parametrize("n_kh", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_mrope_fwd_non_interleaved(num_tokens, n_qh, n_kh, head_dim, dtype):
    """Test MRoPE forward pass (non-interleaved mode)."""
    device = get_torch_device()
    mrope_section = [16, 24, 24]

    q, k, cos_3d, sin_3d, mrope_section_adj = prepare_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device, dtype=dtype
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(mrope_ref, q, k, cos_3d, sin_3d, mrope_section_adj, False)


@pytest.mark.parametrize("num_tokens", [1, 16, 32])
@pytest.mark.parametrize("n_qh", [8, 16])
@pytest.mark.parametrize("n_kh", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_mrope_fwd_interleaved(num_tokens, n_qh, n_kh, head_dim, dtype):
    """Test MRoPE forward pass (interleaved mode)."""
    device = get_torch_device()
    mrope_section = [16, 24, 24]

    q, k, cos_3d, sin_3d, mrope_section_adj = prepare_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device, dtype=dtype
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(mrope_ref, q, k, cos_3d, sin_3d, mrope_section_adj, True)


@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("n_qh", [16])
@pytest.mark.parametrize("n_kh", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("mrope_section", [[16, 24, 24], [32, 16, 16]])
@bypass_not_implemented
def test_mrope_sections(num_tokens, n_qh, n_kh, head_dim, mrope_section):
    """Test MRoPE with different section configurations (non-interleaved)."""
    device = get_torch_device()

    q, k, cos_3d, sin_3d, mrope_section_adj = prepare_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(mrope_ref, q, k, cos_3d, sin_3d, mrope_section_adj, False)


@bypass_not_implemented
def test_mrope_single_token():
    """Test MRoPE with single token input."""
    device = get_torch_device()
    num_tokens = 1
    n_qh = 8
    n_kh = 8
    head_dim = 128
    mrope_section = [16, 24, 24]

    q, k, cos_3d, sin_3d, mrope_section_adj = prepare_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(mrope_ref, q, k, cos_3d, sin_3d, mrope_section_adj, False)


@bypass_not_implemented
def test_mrope_small_head_dim():
    """Test MRoPE with small head dimension."""
    device = get_torch_device()
    num_tokens = 16
    n_qh = 8
    n_kh = 4
    head_dim = 64
    mrope_section = [8, 16, 8]

    q, k, cos_3d, sin_3d, mrope_section_adj = prepare_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(mrope_ref, q, k, cos_3d, sin_3d, mrope_section_adj, False)
