"""qk_norm_rope accuracy suite (task: rms_norm_rope Ascend adaptation).

Reference: the eager ``torch`` provider (upstream _eager_qk_norm_rope).
Tolerances follow the CANN opbase mixed-tolerance standard; bitwise boundary
cases use torch.equal. The fused-provider numeric contract (BF16 rounding
before FP32 rotation) only applies to fused implementations.
"""

import pytest
import torch

from mojo_opset import Target
from mojo_opset import functions as F
from mojo_opset.functions._dispatch import resolve_implementation

HEAD_DIM, ROTARY_DIM, EPS = 128, 96, 1e-6

# CANN opbase mixed tolerance standard: per-element |a-g| <= atol + rtol*|g|,
# matched_ratio >= 0.99 and max_abs_error <= max(floor, 32*ULP(golden)).
_TOL = {
    torch.float16: (2.0**-10, 2.0**-10, 1e-1),
    torch.bfloat16: (2.0**-7, 2.0**-7, 1e-1),
    torch.float32: (2.0**-13, 2.0**-13, 1e-2),
}


def assert_mixed_tolerance(actual, golden, dtype):
    atol, rtol, floor = _TOL[dtype]
    err = (actual.float() - golden.float()).abs()
    assert (err <= atol + rtol * golden.float().abs()).float().mean().item() >= 0.99
    index = err.flatten().argmax()
    golden_low = golden.flatten()[index].to(dtype).cpu()
    ulp = torch.nextafter(golden_low, torch.full_like(golden_low, float("inf"))) - golden_low
    assert err.flatten()[index].item() <= max(floor, 32.0 * ulp.float().item())


def make_inputs(
    tokens,
    heads,
    head_dim=HEAD_DIM,
    rotary_dim=ROTARY_DIM,
    dtype=torch.bfloat16,
    table_dtype=None,
    strided=False,
    device="npu",
    seed=17,
):
    torch.manual_seed(seed)
    if strided:
        q = (torch.randn(tokens, heads, head_dim * 2, device=device, dtype=dtype) * 0.3)[:, :, :head_dim]
        k = (torch.randn(tokens, heads, head_dim * 2, device=device, dtype=dtype) * 0.3)[:, :, :head_dim]
    else:
        q = torch.randn(tokens, heads, head_dim, device=device, dtype=dtype) * 0.3
        k = torch.randn(tokens, heads, head_dim, device=device, dtype=dtype) * 0.3
    qw = torch.randn(head_dim, device=device, dtype=dtype) * 0.1 + 1.0
    kw = torch.randn(head_dim, device=device, dtype=dtype) * 0.1 + 1.0
    half = rotary_dim // 2
    cos = torch.rand(tokens, half, device=device, dtype=torch.float32) * 2 - 1
    sin = torch.sqrt((1 - cos**2).clamp(0, 1))
    table = torch.cat((cos, sin), dim=-1).to(table_dtype or dtype).contiguous()
    return q, k, qw, kw, table


def _manual_reference(q, k, q_weight, k_weight, rope_table, eps, head_dim, rotary_dim):
    half = rotary_dim // 2

    def norm_rope(x, w):
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        n = (xf * inv * w.float()).to(x.dtype).float()
        cos = rope_table[..., :half].unsqueeze(1).float()
        sin = rope_table[..., half:].unsqueeze(1).float()
        first, second = n[..., :half], n[..., half:rotary_dim]
        return torch.cat((first * cos - second * sin, second * cos + first * sin, n[..., rotary_dim:]), dim=-1)

    return norm_rope(q, q_weight), norm_rope(k, k_weight)


def _kernel_math_replica(x, weight, rope_table, eps, head_dim, rotary_dim):
    """Kernel numeric order in torch: BF16 round before FP32 rotation, one inv_rms per head."""
    half = rotary_dim // 2
    xf = x.float()
    inv_rms = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / head_dim + eps)
    normalized = (xf * inv_rms * weight.float()).to(torch.bfloat16)
    pair_x = torch.cat((x[..., half:rotary_dim], x[..., :half], x[..., rotary_dim:]), dim=-1).float()
    pair_w = torch.cat((weight[half:rotary_dim], weight[:half], weight[rotary_dim:])).float()
    pair_norm = (pair_x * inv_rms * pair_w).to(torch.bfloat16).float()
    cos_h = rope_table[..., :half].float().unsqueeze(1)
    sin_h = rope_table[..., half:].float().unsqueeze(1)
    pad = cos_h.new_zeros((*cos_h.shape[:-1], head_dim - rotary_dim))
    cos = torch.cat((cos_h, cos_h, pad), dim=-1)
    sin = torch.cat((sin_h, sin_h, pad), dim=-1)
    first = normalized.float() * cos - pair_norm * sin
    second = normalized.float() * cos + pair_norm * sin
    d = torch.arange(head_dim, device=x.device).view(1, 1, head_dim)
    out = torch.where(d < half, first, torch.where(d < rotary_dim, second, normalized.float()))
    return out.to(x.dtype)


def _select(backend):
    implementation, target, _ = backend
    return implementation or resolve_implementation("qk_norm_rope", Target.parse(target))


@pytest.mark.reference
@pytest.mark.parametrize("tokens", [1, 257, 1024])
def test_reference_crosscheck(tokens):
    q, k, qw, kw, table = make_inputs(tokens, 8, device="cpu")
    a = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation="torch")
    b = _manual_reference(q, k, qw, kw, table, EPS, HEAD_DIM, ROTARY_DIM)
    for x, y in zip(a, b):
        assert_mixed_tolerance(x, y, torch.bfloat16)


@pytest.mark.api("functions.qk_norm_rope")
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("tokens", [1, 257, 1024, 4096, 16384, 65536])
@pytest.mark.parametrize("heads", [8, 56])
def test_half_split_matches_eager(accuracy_backend, tokens, heads, strided):
    implementation, _, device = accuracy_backend
    q, k, qw, kw, table = make_inputs(tokens, heads, strided=strided, device=device)
    expected = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation="torch")
    actual = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation=implementation)
    for x, y in zip(actual, expected):
        assert_mixed_tolerance(x, y, q.dtype)


@pytest.mark.api("functions.qk_norm_rope")
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("tokens", [257, 4096])
@pytest.mark.parametrize("heads", [57])
def test_half_split_head_tile_tail(accuracy_backend, tokens, heads, strided):
    """heads=57: head tile is 8, the last group holds one valid head; masked lanes
    must not write out of bounds."""
    implementation, _, device = accuracy_backend
    q, k, qw, kw, table = make_inputs(tokens, heads, strided=strided, device=device)
    expected = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation="torch")
    actual = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation=implementation)
    for x, y in zip(actual, expected):
        assert_mixed_tolerance(x, y, q.dtype)


@pytest.mark.api("functions.qk_norm_rope")
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
def test_half_split_table_dtype(accuracy_backend, table_dtype):
    implementation, _, device = accuracy_backend
    q, k, qw, kw, table = make_inputs(1024, 56, table_dtype=table_dtype, device=device)
    expected = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation="torch")
    actual = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation=implementation)
    for x, y in zip(actual, expected):
        assert_mixed_tolerance(x, y, q.dtype)


@pytest.mark.api("functions.qk_norm_rope")
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(("head_dim", "rotary_dim"), [(64, 64), (128, 128), (256, 128)])
def test_interleaved_matches_eager(accuracy_backend, head_dim, rotary_dim, strided):
    implementation, _, device = accuracy_backend
    if _select(accuracy_backend) == "torch_npu":
        pytest.skip("torch_npu qk_norm_rope supports the half-split layout only")
    q, k, qw, kw, table = make_inputs(
        257, 8, head_dim=head_dim, rotary_dim=rotary_dim, strided=strided, device=device, seed=11
    )
    expected = F.qk_norm_rope(q, k, qw, kw, table, EPS, interleaved=True, implementation="torch")
    actual = F.qk_norm_rope(q, k, qw, kw, table, EPS, interleaved=True, implementation=implementation)
    for x, y in zip(actual, expected):
        assert_mixed_tolerance(x, y, q.dtype)


def _exact_inputs(tokens, heads, mode, device, seed=23):
    """x=ones gives variance=1 and inv_rms=1 under any reduction tree; identity and
    swap tables leave no FMA ambiguity, so outputs are bitwise decidable."""
    torch.manual_seed(seed)
    w = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1 + 1.0
    x = torch.ones(tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    half = ROTARY_DIM // 2
    if mode == "identity":
        cos, sin = torch.ones(tokens, half), torch.zeros(tokens, half)
    else:
        cos, sin = torch.zeros(tokens, half), torch.ones(tokens, half)
    table = torch.cat((cos, sin), dim=-1).to(torch.bfloat16).to(device).contiguous()
    return x, w, table


@pytest.mark.api("functions.qk_norm_rope")
@pytest.mark.parametrize("tokens", [1, 257, 4096])
def test_bitwise_identity_rotation(accuracy_backend, tokens):
    """Identity table: norm=x*w is exact and the rotation adds no perturbation, so
    every output (including the non-rotary tail) equals the weight bitwise."""
    implementation, _, device = accuracy_backend
    x, w, table = _exact_inputs(tokens, 56, "identity", device)
    actual = F.qk_norm_rope(x, x, w, w, table, 0.0, implementation=implementation)
    assert torch.equal(actual[0], w.view(1, 1, HEAD_DIM).expand(tokens, 56, HEAD_DIM))


@pytest.mark.api("functions.qk_norm_rope")
@pytest.mark.parametrize("tokens", [1, 257])
def test_bitwise_pair_reuses_statistics(accuracy_backend, tokens):
    """Swap table: first half = -w[48:96], second half = +w[0:48], tail passes
    through — the pair elements must reuse the head's single inv_rms bitwise."""
    implementation, _, device = accuracy_backend
    x, w, table = _exact_inputs(tokens, 56, "swap", device)
    actual = F.qk_norm_rope(x, x, w, w, table, 0.0, implementation=implementation)
    half = ROTARY_DIM // 2
    expected = torch.cat((-w[half:ROTARY_DIM], w[:half], w[ROTARY_DIM:]))
    assert torch.equal(actual[0], expected.view(1, 1, HEAD_DIM).expand(tokens, 56, HEAD_DIM))


def _require_fused_contract(backend):
    selected = _select(backend)
    if selected != "triton":
        pytest.skip(f"BF16-round numeric contract targets the fused provider, selected {selected!r}")
    return selected


@pytest.mark.api("functions.qk_norm_rope")
def test_statistical_bf16_round_boundary(accuracy_backend):
    """Random inputs vs the kernel-order replica: >=99.99% bitwise equal; residual
    bf16 round-boundary flips stay within the mixed tolerance."""
    _require_fused_contract(accuracy_backend)
    implementation, _, device = accuracy_backend
    q, k, qw, kw, table = make_inputs(4096, 56, device=device, seed=31)
    actual = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation=implementation)[0]
    expected = _kernel_math_replica(q, qw, table, EPS, HEAD_DIM, ROTARY_DIM)
    assert (actual != expected).float().mean().item() <= 1e-4
    assert_mixed_tolerance(actual, expected, q.dtype)


@pytest.mark.api("functions.qk_norm_rope")
def test_bf16_midpoint_round_direction(accuracy_backend):
    """Weights on exact BF16 midpoints must round half-to-even at the kernel's
    intermediate cast; the identity table exposes every lane's decision."""
    _require_fused_contract(accuracy_backend)
    implementation, _, device = accuracy_backend
    tokens, heads = 257, 8
    half = ROTARY_DIM // 2
    base = torch.tensor([257 + 2 * ((d * 7) % 100) for d in range(HEAD_DIM)], dtype=torch.float32)
    w = (base * (2.0**-8) * torch.tensor([2.0 ** (d // 64) for d in range(HEAD_DIM)])).contiguous().to(device)
    x = torch.ones(tokens, heads, HEAD_DIM, device=device, dtype=torch.float32)
    table = torch.cat((torch.ones(tokens, half), torch.zeros(tokens, half)), dim=-1).to(device).contiguous()
    actual = F.qk_norm_rope(x, x, w, w, table, 0.0, implementation=implementation)[0]
    expected = w.to(torch.bfloat16).to(torch.float32).view(1, 1, HEAD_DIM).expand(tokens, heads, HEAD_DIM)
    assert torch.equal(actual, expected)


@pytest.mark.api("functions.qk_norm_rope")
def test_pair_shared_inv_rms_variance_sensitive(accuracy_backend):
    """Rotary halves with ~256x magnitude asymmetry: the pair must reuse the
    head's single inv_rms instead of recomputing statistics over another dim set."""
    _require_fused_contract(accuracy_backend)
    implementation, _, device = accuracy_backend
    torch.manual_seed(37)
    tokens, heads = 1024, 8
    half = ROTARY_DIM // 2
    scale = torch.ones(HEAD_DIM, dtype=torch.float32)
    scale[:half] = 8.0
    scale[half:ROTARY_DIM] = 0.03

    def make():
        x = (torch.randn(tokens, heads, HEAD_DIM, device=device, dtype=torch.float32) * scale.to(device)).to(
            torch.bfloat16
        )
        w = (torch.randn(HEAD_DIM, device=device, dtype=torch.float32) * 0.1 + 1.0).to(torch.bfloat16)
        cos = torch.rand(tokens, half, device=device, dtype=torch.float32) * 2 - 1
        sin = torch.sqrt((1 - cos**2).clamp(0, 1))
        return x, w, torch.cat((cos, sin), dim=-1).to(torch.bfloat16).contiguous()

    q, qw, table = make()
    k, kw, _ = make()
    actual = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation=implementation)[0]
    assert_mixed_tolerance(actual, _kernel_math_replica(q, qw, table, EPS, HEAD_DIM, ROTARY_DIM), q.dtype)
    expected = F.qk_norm_rope(q, k, qw, kw, table, EPS, implementation="torch")[0]
    assert_mixed_tolerance(actual, expected, q.dtype)
