"""Tests for functions.rope_quant_insert.

The records are integer/byte layouts (FP8 values, UE8M0 scale bytes, BF16 tail), so
the whole backing buffer is compared byte-exactly, which is stricter than the
floating-point rule in docs/operator_accuracy_standard.md §2.2 and still leaves
the FP8 value tolerance satisfied. Untouched slots, non-group-tail tokens and the
region past ``num_tokens`` are covered by the same full-buffer comparison.
"""

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_bitwise_equal
from tests._compile import compile_fullgraph

FP8_MAX = 448.0
E8M0_MIN = 1e-4
VARIANTS = ["fp8", "mxfp8", "plain_bf16", "plain_fp8"]


# ------------------------- independent CPU oracle ------------------------- #


def _rotate_tail(latent_rows, cs_rows):
    row = latent_rows.float()
    cos, sin = cs_rows[:, :32], cs_rows[:, 32:]
    even, odd = row[:, 448::2], row[:, 449::2]
    out = torch.empty_like(row[:, 448:])
    out[:, 0::2] = even * cos - odd * sin
    out[:, 1::2] = odd * cos + even * sin
    return out


def _ue8m0(x, block):
    tiles = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    amax = tiles.abs().amax(-1).clamp(min=E8M0_MIN)
    exponent = torch.ceil(torch.log2(amax * (1.0 / FP8_MAX)))
    q = (tiles * torch.exp2(-exponent)[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    enc = (exponent + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    return q.view(torch.uint8).reshape(*x.shape[:-1], x.shape[-1]), enc


def _oracle_backing(backing, variant, latent, positions, cos_sin, slots, compress_ratio, fp8_scale):
    """Literal transcription of the vllm write pattern on the whole backing."""
    expected = backing.clone()
    boundary = (slots >= 0) & ((positions + 1) % compress_ratio == 0)
    index = torch.nonzero(boundary).flatten()
    if index.numel() == 0:
        return expected
    block = 64
    record = {"fp8": 584, "mxfp8": 528}.get(variant, 512)
    page_elems = backing.shape[1]
    slot = slots[index]
    page, row = (slot // block).long(), (slot % block).long()
    position = positions[index]
    cs_rows = cos_sin[position // compress_ratio * compress_ratio].float()
    rotated = _rotate_tail(latent[index], cs_rows)

    if variant == "fp8":
        q, enc = _ue8m0(latent[index].float()[:, :448], 64)
        q_off = (row * 576)[:, None] + torch.arange(448)
        expected[page[:, None], q_off] = q
        r_off = (row * 576 + 448)[:, None] + torch.arange(128)
        expected[page[:, None], r_off] = rotated.to(torch.bfloat16).view(torch.uint8)
        enc8 = torch.zeros((index.numel(), 8), dtype=torch.uint8)
        enc8[:, :7] = enc
        s_off = (block * 576 + row * 8)[:, None] + torch.arange(8)
        expected[page[:, None], s_off] = enc8
    elif variant == "mxfp8":
        full = torch.cat((latent[index].float()[:, :448], rotated), dim=-1)
        q, enc = _ue8m0(full, 32)
        v_off = (row * 528)[:, None] + torch.arange(512)
        expected[page[:, None], v_off] = q
        sc_off = (row * 528 + 512)[:, None] + torch.arange(16)
        expected[page[:, None], sc_off] = enc
    else:
        values = torch.cat((latent[index, :448], rotated.to(torch.bfloat16)), dim=-1)
        if variant == "plain_fp8":
            values = (values.float() * (1.0 / float(fp8_scale))).clamp(-FP8_MAX, FP8_MAX).to(
                torch.float8_e4m3fn
            )
        off = (row * record)[:, None] + torch.arange(record)
        expected[page[:, None], off] = values
    return expected


# ------------------------------- case builder ------------------------------ #


def _make_case(variant, num_tokens, compress_ratio, block=64, page_pad=0, seed=0,
                max_pos=512, contiguous_slots=False):
    torch.manual_seed(seed)
    if contiguous_slots:
        # positions are the request's own token positions, so the cos/sin table
        # must span them.
        max_pos = max(max_pos, num_tokens)
    latent = torch.randn(num_tokens + 5, 512, dtype=torch.bfloat16)  # extra rows must not be read
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64)
    positions[0::3] = (positions[0::3] // compress_ratio) * compress_ratio
    positions[1::3] = (positions[1::3] // compress_ratio) * compress_ratio + compress_ratio - 1
    angles = torch.randn(max_pos, 32)
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1)

    nslots = max(num_tokens, 1) * 2 + block
    num_blocks = (nslots + block - 1) // block
    if contiguous_slots:
        # Production prefill fills one compressed block at a time and processes
        # tokens in position order: consecutive group tails land on consecutive
        # slots at consecutive positions.
        slots = torch.arange(num_tokens, dtype=torch.int64)
        positions = torch.arange(num_tokens, dtype=torch.int64)
    else:
        slots = torch.randperm(num_blocks * block)[:num_tokens].to(torch.int64)
        slots[::7] = -1

    prev = torch.cat((torch.tensor([-2], dtype=slots.dtype), slots[:-1]))
    prev_pos = torch.cat((torch.tensor([-2]), positions[:-1]))
    slot_contig = ((slots >= 0) & (slots == prev + 1) & (positions == prev_pos + 1)
                 & ((positions + 1) % compress_ratio == 0)).to(torch.uint8)

    if variant in ("fp8", "mxfp8"):
        record = 584 if variant == "fp8" else 528
        page_elems = ((block * record + 511) // 512) * 512 + page_pad
        backing = torch.randint(0, 256, (num_blocks, page_elems), dtype=torch.uint8)
        fp8_scale = None
    else:
        page_elems = block * 512 + page_pad
        dtype = torch.bfloat16 if variant == "plain_bf16" else torch.float8_e4m3fn
        backing = (
            torch.randint(0, 256, (num_blocks * page_elems * dtype.itemsize,), dtype=torch.uint8)
            .view(dtype)
            .view(num_blocks, page_elems)
        )
        fp8_scale = torch.tensor([0.7], dtype=torch.float32)
    return dict(
        variant=variant,
        latent=latent[:num_tokens],
        positions=positions,
        cos_sin=cos_sin,
        backing=backing,
        num_blocks=num_blocks,
        block=block,
        page_elems=page_elems,
        slots=slots,
        compress_ratio=compress_ratio,
        fp8_scale=fp8_scale,
        slot_contig=slot_contig,
        num_tokens=num_tokens,
    )


def _cache_view(backing, case):
    record = {"fp8": 584, "mxfp8": 528}.get(case["variant"], 512)
    return backing.as_strided(
        (case["num_blocks"], case["block"], record),
        (case["page_elems"], record, 1),
    )


def _run(implementation, case, device):
    actual_backing = case["backing"].clone().to(device)
    expected_backing = case["backing"].clone().to(device)
    kwargs = dict(
        latent=case["latent"].to(device),
        positions=case["positions"].to(device),
        cos_sin_cache=case["cos_sin"].to(device),
        slot_mapping=case["slots"].to(device),
        compress_ratio=case["compress_ratio"],
    )
    fp8_scale = None if case["fp8_scale"] is None else case["fp8_scale"].to(device)
    F.rope_quant_insert(
        **kwargs,
        kv_cache=_cache_view(actual_backing, case),
        fp8_scale=fp8_scale,
        slot_contig=case["slot_contig"].to(device),
        implementation=implementation,
    )
    F.rope_quant_insert(
        **kwargs,
        kv_cache=_cache_view(expected_backing, case),
        fp8_scale=fp8_scale,
        implementation="torch",
    )
    assert torch.equal(
        actual_backing.view(torch.uint8), expected_backing.view(torch.uint8)
    ), f"backing mismatch: {case['variant']} T={case['num_tokens']}"
    return actual_backing


def _skip_variant(target, variant):
    if target.startswith("npu.a2") and variant != "plain_bf16":
        pytest.skip("910B (A2) has no FP8: only the plain bf16 record is supported")


# ---------------------------------- tests ---------------------------------- #


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "variant,num_tokens,compress_ratio",
    [
        ("fp8", 256, 2),
        ("fp8", 2048, 2),
        ("fp8", 4096, 1),
        # Non-8-aligned token counts exercise the masked tail of the RPP unroll
        # and the partial-group fallback of the contiguous-run path.
        ("fp8", 255, 2),
        ("fp8", 1023, 2),
        ("mxfp8", 256, 2),
        ("mxfp8", 2048, 2),
        ("mxfp8", 1023, 2),
        ("plain_bf16", 96, 2),
        ("plain_bf16", 2048, 1),
        ("plain_bf16", 255, 2),
        ("plain_fp8", 96, 2),
        ("plain_fp8", 255, 2),
    ],
    ids=lambda value: str(value),
)
def test_rope_quant_insert_variants(accuracy_backend, variant, num_tokens, compress_ratio):
    implementation, target, device = accuracy_backend
    _skip_variant(target, variant)
    case = _make_case(variant, num_tokens, compress_ratio, seed=num_tokens)
    _run(implementation, case, device)


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
@pytest.mark.parametrize("block", [1, 8, 32, 128])
@pytest.mark.parametrize("page_pad", [0, 512])
@pytest.mark.parametrize("variant", ["fp8", "plain_bf16"])
def test_rope_quant_insert_page_geometry(accuracy_backend, variant, block, page_pad):
    """Block size 1..128 and a padded page stride (the view is strided)."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, variant)
    case = _make_case(variant, 96, 2, block=block, page_pad=page_pad, seed=block)
    _run(implementation, case, device)


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
@pytest.mark.parametrize("variant", ["plain_bf16", "plain_fp8"])
@pytest.mark.parametrize("tokens", [256, 4096])
def test_rope_quant_insert_contiguous_group(accuracy_backend, variant, tokens):
    """Block-contiguous slots (production prefill) take the 2D group path."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, variant)
    case = _make_case(variant, tokens, 1, seed=tokens, contiguous_slots=True)
    _run(implementation, case, device)


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
@pytest.mark.parametrize("variant", ["fp8", "mxfp8", "plain_bf16", "plain_fp8"])
def test_rope_quant_insert_slot_contig_fast_path(accuracy_backend, variant):
    """Block-contiguous slots+positions: every group takes the 2D merged path."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, variant)
    case = _make_case(variant, 512, 1, seed=31, contiguous_slots=True)
    _run(implementation, case, device)


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
def test_rope_quant_insert_slot_contig_mixed(accuracy_backend):
    """Alternating runs: groups 0-7 contiguous, 8-15 scattered, 16+ contiguous."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, "mxfp8")
    case = _make_case("mxfp8", 512, 1, seed=37, contiguous_slots=True)
    slots = case["slots"].clone()
    slots[64:128] = slots[64:128][torch.randperm(64)]
    case["slots"] = slots
    prev = torch.cat((torch.tensor([-2], dtype=slots.dtype), slots[:-1]))
    prev_pos = torch.cat((torch.tensor([-2]), case["positions"][:-1]))
    case["slot_contig"] = ((slots >= 0) & (slots == prev + 1)
                         & (case["positions"] == prev_pos + 1)).to(torch.uint8)
    _run(implementation, case, device)


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
@pytest.mark.parametrize("scatter_from", [10, 12, 15])
def test_rope_quant_insert_short_run_at_group_head(accuracy_backend, scatter_from):
    """A run shorter than RPP whose start sits on a group boundary must fall
    back: the group interior contains a zero contig byte, so the 2D path is
    not taken and scattered rows land on their true slots."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, "mxfp8")
    case = _make_case("mxfp8", 32, 1, seed=53, contiguous_slots=True)
    slots = case["slots"].clone()
    slots[scatter_from:] = slots[scatter_from:][torch.randperm(32 - scatter_from)]
    case["slots"] = slots
    prev = torch.cat((torch.tensor([-2], dtype=slots.dtype), slots[:-1]))
    prev_pos = torch.cat((torch.tensor([-2]), case["positions"][:-1]))
    case["slot_contig"] = ((slots >= 0) & (slots == prev + 1)
                           & (case["positions"] == prev_pos + 1)).to(torch.uint8)
    _run(implementation, case, device)


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
def test_rope_quant_insert_slot_contig_input_validation(accuracy_backend):
    """slot_contig with wrong dtype/shape/device is rejected."""
    _, _, device = accuracy_backend
    case = _make_case("mxfp8", 32, 1, seed=41)
    kw = dict(
        latent=case["latent"].to(device),
        positions=case["positions"].to(device),
        cos_sin_cache=case["cos_sin"].to(device),
        kv_cache=_cache_view(case["backing"].to(device), case),
        slot_mapping=case["slots"].to(device),
        compress_ratio=1,
    )
    import pytest as _pytest
    with _pytest.raises(ValueError):
        F.rope_quant_insert(**kw, slot_contig=torch.zeros(32, dtype=torch.float32, device=device))
    with _pytest.raises(ValueError):
        F.rope_quant_insert(**kw, slot_contig=torch.zeros(16, dtype=torch.uint8, device=device))


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
def test_rope_quant_insert_semantics(accuracy_backend):
    """Non-tail tokens and PAD slots leave their physical row untouched."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, "plain_bf16")
    case = _make_case("plain_bf16", 33, 2, block=8, page_pad=256, seed=7)
    before = case["backing"].clone().view(torch.uint8)
    after = _run(implementation, case, device).cpu().view(torch.uint8)
    block = case["block"]
    row_bytes = 512 * 2  # bf16 record
    skipped = 0
    for t in range(case["num_tokens"]):
        slot = int(case["slots"][t])
        position = int(case["positions"][t])
        if slot < 0:
            skipped += 1
            continue
        if (position + 1) % 2 == 0:
            continue
        skipped += 1
        page, row = slot // block, slot % block
        assert torch.equal(
            after[page, row * row_bytes : (row + 1) * row_bytes],
            before[page, row * row_bytes : (row + 1) * row_bytes],
        ), f"non-tail token {t} modified its slot"
    assert skipped > 0


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
def test_rope_quant_insert_empty_batch(accuracy_backend):
    """Empty tensor (docs §1.4): the cache is untouched."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, "plain_bf16")
    case = _make_case("plain_bf16", 0, 2, seed=11)
    before = case["backing"].clone()
    kwargs = dict(
        latent=case["latent"].to(device),
        positions=case["positions"].to(device),
        cos_sin_cache=case["cos_sin"].to(device),
        slot_mapping=case["slots"].to(device),
        compress_ratio=2,
    )
    backing = case["backing"].clone().to(device)
    F.rope_quant_insert(
        **kwargs, kv_cache=_cache_view(backing, case), implementation=implementation
    )
    assert torch.equal(backing.cpu().view(torch.uint8), before.view(torch.uint8))


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.accuracy
def test_rope_quant_insert_special_values(accuracy_backend):
    """NaN / +-inf latent on the plain bf16 path: bit-exact agreement."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, "plain_bf16")
    case = _make_case("plain_bf16", 16, 2, seed=13)
    latent = case["latent"].clone()
    latent[0, 0] = float("nan")
    latent[1, 448] = float("nan")
    latent[2, 511] = float("inf")
    latent[3, 5] = float("-inf")
    case["latent"] = latent
    _run(implementation, case, device)


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.bitwise
def test_rope_quant_insert_idempotent(accuracy_backend):
    """A second identical insert must not change any byte."""
    implementation, target, device = accuracy_backend
    _skip_variant(target, "fp8")
    case = _make_case("fp8", 256, 2, seed=17)
    first = _run(implementation, case, device)
    second = first.clone()
    F.rope_quant_insert(
        latent=case["latent"].to(device),
        positions=case["positions"].to(device),
        cos_sin_cache=case["cos_sin"].to(device),
        kv_cache=_cache_view(second, case),
        slot_mapping=case["slots"].to(device),
        compress_ratio=2,
        implementation=implementation,
    )
    assert_bitwise_equal(first, second)


@pytest.mark.api("functions.rope_quant_insert")
def test_rope_quant_insert_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("rope_quant_insert", implementation="triton", target=target)
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    case = _make_case("plain_bf16", 8, 2, seed=19)
    backing = case["backing"].clone().to(device)
    torch.library.opcheck(
        namespace.rope_quant_insert_fwd.default,
        (
            case["latent"].to(device),
            case["positions"].to(device),
            case["cos_sin"].to(device),
            _cache_view(backing, case),
            case["slots"].to(device),
            2,
            None,
            case["slot_contig"].to(device),
        ),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_rope_quant_insert_compiled(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU Triton compile contract")
    implementation = implementation or "triton"
    preload("rope_quant_insert", implementation=implementation)
    case = _make_case("plain_bf16", 64, 2, seed=23)
    backing = case["backing"].clone().to(device)
    latent = case["latent"].to(device)
    positions = case["positions"].to(device)
    cos_sin = case["cos_sin"].to(device)
    slots = case["slots"].to(device)

    def run(cache, latent, positions, cos_sin, slots):
        F.rope_quant_insert(
            latent,
            positions,
            cos_sin,
            cache,
            slots,
            2,
            implementation=implementation,
        )
        return (cache,)

    try:
        eager_backing = backing.clone()
        (eager,) = run(_cache_view(eager_backing, case), latent, positions, cos_sin, slots)
        compiled_backing = backing.clone()
        (actual,) = compile_fullgraph(run, backend=backend)(
            _cache_view(compiled_backing, case), latent, positions, cos_sin, slots
        )
        assert torch.equal(
            compiled_backing.view(torch.uint8), eager_backing.view(torch.uint8)
        )
    finally:
        torch._dynamo.reset()


# --------------------------- reference-only checks --------------------------- #


@pytest.mark.reference
@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_matches_oracle(variant):
    """The torch provider must equal the literal vllm write pattern on CPU."""
    case = _make_case(variant, 64, 2, seed=29)
    backing = case["backing"].clone()
    F.rope_quant_insert(
        latent=case["latent"],
        positions=case["positions"],
        cos_sin_cache=case["cos_sin"],
        kv_cache=_cache_view(backing, case),
        slot_mapping=case["slots"],
        compress_ratio=2,
        fp8_scale=case["fp8_scale"],
        implementation="torch",
    )
    expected = _oracle_backing(
        case["backing"],
        variant,
        case["latent"],
        case["positions"],
        case["cos_sin"],
        case["slots"],
        2,
        case["fp8_scale"],
    )
    assert torch.equal(backing.view(torch.uint8), expected.view(torch.uint8))
