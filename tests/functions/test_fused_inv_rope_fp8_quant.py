"""Tests for functions.fused_inv_rope_fp8_quant.

Tolerances follow docs/operator_accuracy_standard.md §2.2: fp8 E4M3 2^-3,
bf16 2^-7, fp16 2^-10 and FP32 scales 2^-13. The packed ``int32`` UE8M0 scales
are integer data and compared exactly. The rotation is a pure float path, so NaN
inputs are additionally checked bit-exactly (the quantised epilogue turns NaN into
NaN fp8, which §2.3's ``equal_nan=False`` rule cannot express).
"""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_bitwise_equal
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

_TOL = {
    torch.float16: (2.0**-10, 2.0**-10),
    torch.bfloat16: (2.0**-7, 2.0**-7),
    torch.float32: (2.0**-13, 2.0**-13),
    torch.float8_e4m3fn: (2.0**-3, 2.0**-3),
}
FP8_DTYPE = torch.float8_e4m3fn
EPS = 1e-10
MAX_POS = 4096

# (name, tokens, heads, n_groups, heads_per_group, nope_dim, rope_dim, quant_group_size)
SHAPE_CASES = [
    ("ds41_T1", 1, 128, 16, 8, 448, 64, 128),
    ("ds41_T7", 7, 128, 16, 8, 448, 64, 128),
    ("ds41_T33", 33, 128, 16, 8, 448, 64, 128),
    ("ds41_T256", 256, 128, 16, 8, 448, 64, 128),
    ("ds41_T2048", 2048, 128, 16, 8, 448, 64, 128),
    ("tp2_T256", 256, 64, 8, 8, 448, 64, 128),
    ("hpg16_T64", 64, 128, 8, 16, 448, 64, 128),
    ("head256_T64", 64, 32, 8, 4, 192, 64, 128),
    # rope_dim == quant_group_size -> original-structure fallback
    ("tail0_T64", 64, 64, 8, 8, 448, 64, 64),
    # chunks_per_head == 1 -> original-structure fallback
    ("chunks1_T64", 64, 32, 8, 4, 96, 32, 128),
    ("rope32_T64", 64, 64, 8, 8, 480, 32, 64),
]


def _cos_sin_cache(rope_dim, device, max_pos=MAX_POS):
    half = rope_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _run(implementation, tokens, heads, n_groups, heads_per_group, nope_dim, rope_dim,
         quant_group_size, dtype, device, quantize=True, tas=False, seed=0):
    torch.manual_seed(seed)
    head_dim = nope_dim + rope_dim
    o = torch.randn(tokens, heads, head_dim, dtype=torch.float32, device=device).to(dtype)
    positions = torch.randint(0, MAX_POS, (tokens,), dtype=torch.int64, device=device)
    cache = _cos_sin_cache(rope_dim, device)
    kwargs = dict(
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
        tma_aligned_scales=tas,
        quantize=quantize,
    )
    actual = F.fused_inv_rope_fp8_quant(o, positions, cache, implementation=implementation, **kwargs)
    expected = F.fused_inv_rope_fp8_quant(o, positions, cache, implementation="torch", **kwargs)
    return actual, expected, (o, positions, cache, kwargs)


def _check(actual, expected, quantize, tas):
    out, scale = actual
    ref_out, ref_scale = expected
    assert out.shape == ref_out.shape and out.dtype == ref_out.dtype
    assert str(out.stride()) == str(ref_out.stride()), (
        f"output layout: {out.stride()} != {ref_out.stride()}"
    )
    rtol, atol = _TOL[out.dtype]
    assert_close(out.float(), ref_out.float(), torch.float32, rtol=rtol, atol=atol, name="out")
    if not quantize:
        assert scale.numel() == 0 and ref_scale.numel() == 0
        return
    assert scale.shape == ref_scale.shape and scale.dtype == ref_scale.dtype
    assert str(scale.stride()) == str(ref_scale.stride()), (
        f"scale layout: {scale.stride()} != {ref_scale.stride()}"
    )
    if tas:
        assert torch.equal(scale, ref_scale), "packed UE8M0 words differ"
    else:
        rtol, atol = _TOL[torch.float32]
        assert_close(scale, ref_scale, torch.float32, rtol=rtol, atol=atol, name="scale")


def _skip_without_fp8(target):
    if target.startswith("npu.a2"):
        pytest.skip("910B (A2) has no FP8: quantize=True fails fast there")


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "name,tokens,heads,n_groups,heads_per_group,nope_dim,rope_dim,quant_group_size",
    SHAPE_CASES,
    ids=[case[0] for case in SHAPE_CASES],
)
def test_fused_inv_rope_fp8_quant_shapes(
    accuracy_backend, name, tokens, heads, n_groups, heads_per_group, nope_dim, rope_dim, quant_group_size
):
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    actual, expected, _ = _run(
        implementation, tokens, heads, n_groups, heads_per_group, nope_dim, rope_dim,
        quant_group_size, torch.bfloat16, device, seed=tokens,
    )
    _check(actual, expected, True, False)


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize("quantize", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, FP8_DTYPE], ids=["bf16", "fp16", "fp8"])
def test_fused_inv_rope_fp8_quant_dtypes(accuracy_backend, dtype, quantize):
    implementation, target, device = accuracy_backend
    if quantize:
        _skip_without_fp8(target)
    actual, expected, _ = _run(
        implementation, 64, 128, 16, 8, 448, 64, 128, dtype, device, quantize=quantize, seed=21
    )
    _check(actual, expected, quantize, False)


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.accuracy
def test_fused_inv_rope_fp8_quant_tma_aligned_scales(accuracy_backend):
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    actual, expected, _ = _run(
        implementation, 129, 128, 16, 8, 448, 64, 128, torch.bfloat16, device,
        quantize=True, tas=True, seed=22,
    )
    _check(actual, expected, True, True)


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.accuracy
def test_fused_inv_rope_fp8_quant_scale_padding(accuracy_backend):
    """Rows [tokens, align(tokens, 4)) of the MN-major scale buffer must be zero."""
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    for tokens in (5, 129, 130):
        (_, scale), _, _ = _run(
            implementation, tokens, 128, 16, 8, 448, 64, 128, torch.bfloat16, device, seed=tokens
        )
        aligned = (tokens + 3) // 4 * 4
        if aligned == tokens:
            continue
        n_groups, heads_per_group, rope_dim = 16, 8, 64
        d = heads_per_group * (448 + rope_dim)
        inner = d // 128
        padding = torch.as_strided(
            scale,
            (n_groups, inner, aligned - tokens),
            (inner * aligned, aligned, 1),
            storage_offset=tokens,
        )
        assert torch.equal(padding, torch.zeros_like(padding)), f"padding rows not zero at T={tokens}"


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.accuracy
def test_fused_inv_rope_fp8_quant_empty_batch(accuracy_backend):
    """Empty tensor (docs §1.4): empty outputs, no launch."""
    implementation, _, device = accuracy_backend
    o = torch.empty(0, 128, 512, dtype=torch.bfloat16, device=device)
    positions = torch.empty(0, dtype=torch.int64, device=device)
    cache = _cos_sin_cache(64, device)
    out, scale = F.fused_inv_rope_fp8_quant(
        o, positions, cache, 16, 8, quantize=False, implementation=implementation
    )
    assert out.shape == (0, 16, 4096) and out.numel() == 0
    assert scale.numel() == 0


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.accuracy
def test_fused_inv_rope_fp8_quant_quant_empty_batch(accuracy_backend):
    """Empty tensor on the quantising entry: empty fp8 output and scale buffer."""
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    o = torch.empty(0, 128, 512, dtype=torch.bfloat16, device=device)
    positions = torch.empty(0, dtype=torch.int64, device=device)
    cache = _cos_sin_cache(64, device)
    out, scale = F.fused_inv_rope_fp8_quant(
        o, positions, cache, 16, 8, implementation=implementation
    )
    assert out.shape == (0, 16, 4096) and out.dtype == FP8_DTYPE
    assert scale.numel() == 0


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_inv_rope_fp8_quant_special_values(accuracy_backend, dtype):
    """NaN / +-inf on the rotation path (quantize=False): bit-exact agreement."""
    implementation, _, device = accuracy_backend
    tokens, heads = 8, 32
    torch.manual_seed(31)
    o = torch.randn(tokens, heads, 512, dtype=torch.float32, device=device)
    o[0, 0, 0] = float("nan")
    o[1, 1, 448] = float("nan")
    o[2, 2, 511] = float("inf")
    o[3, 3, 5] = float("-inf")
    o[4, 4, 449] = float("inf")
    o = o.to(dtype)
    positions = torch.randint(0, MAX_POS, (tokens,), dtype=torch.int64, device=device)
    cache = _cos_sin_cache(64, device)
    actual = F.fused_inv_rope_fp8_quant(
        o, positions, cache, 8, 4, quantize=False, implementation=implementation
    )
    expected = F.fused_inv_rope_fp8_quant(
        o, positions, cache, 8, 4, quantize=False, implementation="torch"
    )
    assert_bitwise_equal(actual[0], expected[0])


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.bitwise
def test_fused_inv_rope_fp8_quant_repeatable(accuracy_backend):
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    torch.manual_seed(32)
    o = torch.randn(256, 128, 512, dtype=torch.float32, device=device).to(torch.bfloat16)
    positions = torch.randint(0, MAX_POS, (256,), dtype=torch.int64, device=device)
    cache = _cos_sin_cache(64, device)
    assert_repeatable(
        partial(
            F.fused_inv_rope_fp8_quant,
            positions=positions,
            cos_sin_cache=cache,
            n_groups=16,
            heads_per_group=8,
            implementation=implementation,
        ),
        (o,),
        backward=False,
    )


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
def test_fused_inv_rope_fp8_quant_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("fused_inv_rope_fp8_quant", implementation="triton", target=target)
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    o = torch.randn(7, 128, 512, dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, MAX_POS, (7,), dtype=torch.int64, device=device)
    cache = _cos_sin_cache(64, device)
    torch.library.opcheck(
        namespace.fused_inv_rope_fp8_quant_fwd.default,
        (o, positions, cache, 16, 8, 448, 64, 128, False, False),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_fused_inv_rope_fp8_quant_compiled(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU Triton compile contract")
    implementation = implementation or "triton"
    preload("fused_inv_rope_fp8_quant", implementation=implementation)
    torch.manual_seed(33)
    o = torch.randn(64, 128, 512, dtype=torch.float32, device=device).to(torch.bfloat16)
    positions = torch.randint(0, MAX_POS, (64,), dtype=torch.int64, device=device)
    cache = _cos_sin_cache(64, device)

    def run(o, positions, cache):
        return F.fused_inv_rope_fp8_quant(
            o, positions, cache, 16, 8, 448, 64, 128, False, False,
            implementation=implementation,
        )

    try:
        eager = run(o, positions, cache)
        actual = compile_fullgraph(run, backend=backend)(o, positions, cache)
        assert actual[0].stride() == eager[0].stride()
        assert actual[0].shape == eager[0].shape
        assert torch.equal(actual[0].float(), eager[0].float())
    finally:
        torch._dynamo.reset()


# --------------------------- reference-only checks ---------------------------


def _cpu_oracle(o, positions, cache, n_groups, heads_per_group, nope_dim, rope_dim):
    """Literal FP32 transcription of the vllm inverse-RoPE on CPU."""
    num_tokens, _, head_dim = o.shape
    d = heads_per_group * head_dim
    x = o.float()
    offsets = torch.arange(head_dim)
    is_rope = offsets >= nope_dim
    rope_local = offsets - nope_dim
    half = rope_dim // 2
    index = (rope_local >> 1).clamp(min=0)
    cos = cache[positions.long()][:, index]
    sin = cache[positions.long()][:, half + index]
    cos = torch.where(is_rope, cos, torch.ones_like(cos))
    sin = torch.where(is_rope, sin, torch.zeros_like(sin))
    partner = torch.where(is_rope, x[..., offsets ^ 1], torch.zeros_like(x))
    add = x * cos[:, None, :] + partner * sin[:, None, :]
    sub = x * cos[:, None, :] - partner * sin[:, None, :]
    rotated = torch.where((rope_local & 1) == 0, add, sub)
    return torch.where(is_rope, rotated, x).reshape(num_tokens, n_groups, d).to(o.dtype)


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reference_matches_cpu_oracle(dtype):
    """The torch provider must equal the literal CPU FP32 rotation, bit-exactly."""
    torch.manual_seed(41)
    tokens, heads, nope_dim, rope_dim = 5, 16, 448, 64
    o = torch.randn(tokens, heads, nope_dim + rope_dim).to(dtype)
    positions = torch.randint(0, MAX_POS, (tokens,), dtype=torch.int64)
    half = rope_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(MAX_POS, dtype=torch.float32), inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    actual, _ = F.fused_inv_rope_fp8_quant(
        o, positions, cache, 2, 8, quantize=False, implementation="torch"
    )
    expected = _cpu_oracle(o, positions, cache, 2, 8, nope_dim, rope_dim)
    assert torch.equal(actual, expected)
