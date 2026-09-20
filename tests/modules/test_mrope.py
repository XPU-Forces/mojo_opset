import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

MROPE_CASES = [
    (tokens, q_heads, k_heads, 128, section, interleaved)
    for tokens in (1, 32, 128)
    for q_heads, k_heads, section, interleaved in [
        (28, 4, [16, 24, 24], False),
        (40, 8, [16, 24, 24], False),
        (16, 8, [24, 20, 20], True),
        (32, 8, [24, 20, 20], True),
    ]
] + [
    (tokens, q_heads, k_heads, dim, section, False)
    for tokens in (16, 64)
    for q_heads, k_heads in ((32, 4), (64, 8))
    for dim, section in ((128, [8, 12, 12]), (128, [12, 18, 18]), (96, [8, 12, 12]))
]


def make_mrope_case(case, dtype, device):
    tokens, q_heads, k_heads, dim, section, interleaved = case
    rotary_dim = sum(section) * 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim // 2, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(4000, dtype=torch.float32), inv_freq).repeat_interleave(2, -1)
    cache_cos, cache_sin = freqs.cos().to(device), freqs.sin().to(device)
    positions = torch.randint(1000, (3, tokens), device=device)
    cos, sin = cache_cos[positions], cache_sin[positions]
    q = torch.randn(tokens, q_heads * dim, device=device, dtype=dtype)
    k = torch.randn(tokens, k_heads * dim, device=device, dtype=dtype)
    return (q, k, cos, sin), dict(mrope_section=section, is_interleaved=interleaved, head_dim=dim)


@pytest.mark.api("modules.MRoPE", "modules.MRoPEInplace", ops=["mrope"])
@pytest.mark.parametrize("case", MROPE_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.accuracy
def test_mrope(accuracy_backend, case, dtype, inplace):
    implementation, _, device = accuracy_backend
    inputs, options = make_mrope_case(case, dtype, device)
    q, k, cos, sin = inputs
    original_q, original_k = q.clone(), k.clone()
    function = (
        modules.MRoPEInplace(inplace=True, implementation=implementation)
        if inplace
        else modules.MRoPE(implementation=implementation)
    )
    reference = (
        modules.MRoPEInplace(inplace=True, implementation="torch_reference")
        if inplace
        else modules.MRoPE(implementation="torch_reference")
    )
    actual = function(*inputs, **options)
    expected = reference(original_q.clone(), original_k.clone(), cos, sin, **options)
    assert_close(actual, expected, rtol=6e-3 if inplace else 1e-2, atol=3e-2 if inplace else 1e-2)
    if inplace:
        assert actual[0] is q and actual[1] is k
    else:
        assert_close(q, original_q, rtol=0, atol=0)
        assert_close(k, original_k, rtol=0, atol=0)
