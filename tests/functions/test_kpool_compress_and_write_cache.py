from dataclasses import dataclass
from dataclasses import replace
from importlib import import_module

import pytest
import torch

from torch._subclasses.fake_tensor import FakeTensorMode

from mojo_opset import Target
from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._compile import compile_fullgraph

HEAD = 128
PAGE = 64
FP8 = torch.float8_e4m3fn


@dataclass(frozen=True)
class Case:
    case_id: str
    n: int
    pool: int
    index: str = "int64"
    blocks: int = 4
    layout: str = "linear"
    round_scale: bool = True
    mode: str = "write"
    seed: int = 20261103
    page: int = PAGE


PERF_CASES = [
    Case(f"p{pool}_n{n}", n, pool, blocks=max(1, (n + PAGE - 1) // PAGE))
    for pool in (4, 16)
    for n in (1, 3, 8, 64, 512, 4096)
]
CORRECTNESS_CASES = [
    Case(f"anchor_p{pool}_n{n}_{index}", n, pool, index, blocks, "offset17" if n == 1 else "linear")
    for n, pool, blocks in ((1, 16, 1), (3, 4, 2), (8, 4, 4), (8, 16, 4))
    for index in ("int32", "int64")
] + [
    Case("empty", 0, 4, mode="write_return"),
    Case("cross_page", 5, 16, "int32", 2, "cross_page"),
    Case("page32", 5, 16, "int32", 2, "cross_page", page=32),
    Case("page128", 5, 4, "int64", 2, "cross_page", page=128),
    Case("all_masked", 5, 4, layout="all_masked"),
    Case("write_return", 8, 16, round_scale=False, mode="write_return"),
    Case("return_only", 3, 4, mode="return"),
]


def make_inputs(case):
    rng = torch.Generator().manual_seed(case.seed)
    shape = (case.n, case.pool, HEAD)
    k = torch.randn(shape, generator=rng).to(torch.bfloat16)
    score = torch.randn(shape, generator=rng).to(k.dtype)
    ape = torch.randn((case.pool, HEAD), generator=rng)
    cache = torch.full((case.blocks, case.page, HEAD + 4), 0xA5, dtype=torch.uint8)
    loc = torch.arange(case.n, dtype=getattr(torch, case.index))
    mask = None
    if case.layout == "offset17":
        loc[0] = 17
    elif case.layout == "cross_page":
        loc[:] = torch.tensor([case.page, case.page - 1, -1, 0, 2 * case.page - 1])
        mask = torch.tensor([True, True, False, True, False])
    elif case.layout == "all_masked":
        loc.fill_(-1)
        mask = torch.zeros(case.n, dtype=torch.bool)
    return cache, k, score, ape, loc, mask


def reference(case, k, score, ape, mask):
    if mask is not None:
        k, score = k[mask], score[mask]
    logits = score.float() + ape
    weights = (logits - logits.amax(dim=1, keepdim=True)).exp()
    numerator = torch.zeros((k.shape[0], HEAD), dtype=torch.float32)
    denominator = torch.zeros_like(numerator)
    for slot in range(case.pool):
        numerator = torch.addcmul(numerator, k[:, slot].float(), weights[:, slot])
        denominator = denominator + weights[:, slot]
    pooled = numerator / denominator
    # Bound FP32 exp/division and accumulation differences before BF16 rounding.
    # Use weighted magnitudes so near-zero pooled values retain an absolute bound.
    eps = torch.finfo(torch.float32).eps
    weighted_abs = (k.float().abs() * weights).sum(dim=1) / denominator
    pool_error = 32 * eps * weighted_abs
    pooled_bf16 = pooled.bfloat16().float()
    pool_lo = (pooled - pool_error).bfloat16().float()
    pool_hi = (pooled + pool_error).bfloat16().float()
    pool_radius = torch.maximum(pooled_bf16 - pool_lo, pool_hi - pooled_bf16)

    x = pooled_bf16
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = x.reshape(x.shape[0], HEAD // (2 * stride), 2, stride)
        a, b = pairs.unbind(2)
        x = torch.stack((a + b, a - b), dim=2).reshape(x.shape[0], HEAD)
    normalization = 0.08838834764831845
    rotated = x * normalization
    # Each Hadamard coefficient has the same magnitude. A BF16 midpoint
    # crossing before rotation can therefore affect even near-zero outputs.
    radius = (
        pool_radius.sum(dim=1, keepdim=True) + 8 * eps * pooled_bf16.abs().sum(dim=1, keepdim=True)
    ) * normalization
    x = rotated.bfloat16().float()
    lower = (rotated - radius).bfloat16().float()
    upper = (rotated + radius).bfloat16().float()
    scale = x.abs().amax(dim=1).clamp_min(1e-4) * (1 / 448.0)
    if case.round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return (x / scale[:, None]).clamp(-448, 448).to(FP8), scale, lower, upper


def offsets(loc, page):
    loc = loc.long()
    page_base = (loc // page) * page * (HEAD + 4)
    row = loc % page
    return (
        page_base[:, None] + row[:, None] * HEAD + torch.arange(HEAD),
        page_base[:, None] + page * HEAD + row[:, None] * 4 + torch.arange(4),
    )


def check_payload(actual_k, actual_scale, expected_k, expected_scale, lower, upper, rounded):
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0 if rounded else 0.008, atol=0)
    actual = actual_k.view(torch.uint8).to(torch.int16)
    expected = expected_k.view(torch.uint8).to(torch.int16)
    # Monotonic FP8 codes with +0 and -0 treated as the same value.
    actual = torch.where(actual >= 128, 128 - actual, actual)
    expected = torch.where(expected >= 128, 128 - expected, expected)
    delta = (actual - expected).abs()
    observed = actual_k.float()
    # Compare in numeric space against the FP8-rounded endpoints using the
    # scale already checked above. This keeps the allowance local to each value.
    scale = actual_scale[:, None]
    lo = (lower / scale).clamp(-448, 448).to(FP8).float()
    hi = (upper / scale).clamp(-448, 448).to(FP8).float()
    outside = (observed < lo) | (observed > hi)
    bad = ((delta > 1) & outside) | ~torch.isfinite(observed)
    if bad.any():
        row, column = bad.nonzero()[0].tolist()
        raise AssertionError(
            f"FP8 mismatch at row {row}, dim {column}: got {observed[row, column].item()}, "
            f"expected {expected_k[row, column].float().item()}, "
            f"interval [{lo[row, column].item()}, {hi[row, column].item()}], "
            f"scale {actual_scale[row].item()}, max code distance {delta.max().item()}"
        )


def check(case, original, device_values, result):
    cache, k, score, ape, loc, mask = original
    expected_k, expected_scale, lower, upper = reference(case, k, score, ape, mask)
    actual = device_values[0].cpu()
    untouched = torch.ones(cache.numel(), dtype=torch.bool)
    if case.mode != "return":
        k_off, s_off = offsets(loc if mask is None else loc[mask], case.page)
        flat = actual.flatten()
        check_payload(
            flat[k_off].view(FP8),
            flat[s_off].flatten().view(torch.float32),
            expected_k,
            expected_scale,
            lower,
            upper,
            case.round_scale,
        )
        untouched[k_off] = False
        untouched[s_off] = False
    if not torch.equal(actual.flatten()[untouched], cache.flatten()[untouched]):
        raise AssertionError("cache bytes outside write targets changed")
    if case.mode == "write":
        if result is not None:
            raise AssertionError("unexpected returned tensors")
    else:
        result_k, result_scale = (value.cpu() for value in result)
        check_payload(
            result_k,
            result_scale,
            expected_k,
            expected_scale,
            lower,
            upper,
            case.round_scale,
        )
    for initial, final in zip(original[1:], device_values[1:]):
        if initial is not None and not torch.equal(initial, final.cpu()):
            raise AssertionError("an input tensor changed")


def run_case(case, values, implementation):
    cache, key, score, ape, loc, mask = values
    return F.kpool_compress_and_write_cache(
        cache,
        key,
        score,
        ape,
        loc,
        case.pool,
        write_mask=mask,
        round_scale=case.round_scale,
        return_compressed=case.mode != "write",
        write_cache=case.mode != "return",
        implementation=implementation,
    )


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", PERF_CASES + CORRECTNESS_CASES, ids=lambda case: case.case_id)
def test_kpool_compress_and_write_cache_accuracy(accuracy_backend, case):
    implementation, target, device = accuracy_backend
    if Target.parse(target).arch_key not in {"npu.a2", "npu.a5"}:
        pytest.skip("KPool compression kernels support Ascend A2 and A5")

    original = make_inputs(case)
    cpu_values = tuple(value.clone() if value is not None else None for value in original)
    check(case, original, cpu_values, run_case(case, cpu_values, "torch"))
    device_values = tuple(value.to(device) if value is not None else None for value in original)
    check(case, original, device_values, run_case(case, device_values, implementation))


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
def test_kpool_compress_and_write_cache_padded_pages(accuracy_backend):
    implementation, _, device = accuracy_backend
    case = Case("padded_pages", 65, 4)
    cache, key, score, ape, loc, mask = make_inputs(case)
    loc = loc.flip(0)
    payload_bytes = case.page * (HEAD + 4)
    backing = torch.full((case.blocks, payload_bytes + 16), 0xA5, dtype=torch.uint8)

    for target_device, provider in (("cpu", "torch"), (device, implementation)):
        actual_backing = backing.to(target_device, copy=True)
        actual_cache = actual_backing[:, 4 : 4 + payload_bytes].view(cache.shape)
        values = (actual_cache, *(value.to(target_device, copy=True) for value in (key, score, ape, loc)), mask)
        check(case, (cache, key, score, ape, loc, mask), values, run_case(case, values, provider))
        torch.testing.assert_close(actual_backing[:, :4].cpu(), backing[:, :4])
        torch.testing.assert_close(actual_backing[:, 4 + payload_bytes :].cpu(), backing[:, 4 + payload_bytes :])


@pytest.mark.reference
@pytest.mark.accuracy
def test_kpool_compress_and_write_cache_rejects_mixed_devices():
    case = PERF_CASES[0]
    cache, key, score, ape, loc, mask = make_inputs(case)
    values = (cache, key, score, ape, loc.to("meta"), mask)
    with pytest.raises(ValueError, match="same device"):
        run_case(case, values, "torch")


DISPATCH_CASES = [
    Case(f"boundary_p{pool}_n{n}", n, pool, blocks=(n + PAGE - 1) // PAGE + 1)
    for pool, sizes in ((4, (63, 64, 65, 511, 512, 513)), (16, (63, 64, 65, 511, 512, 513, 2047, 2048, 2049)))
    for n in sizes
]


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", DISPATCH_CASES, ids=lambda case: case.case_id)
def test_kpool_compress_and_write_cache_dispatch(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    original = make_inputs(case)
    # Reverse slots so complete tiles exercise the scattered-write fallback.
    original = (*original[:4], original[4].flip(0), original[5])
    device_values = tuple(value.to(device) if value is not None else None for value in original)
    check(case, original, device_values, run_case(case, device_values, implementation))


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("variant", ("noncontiguous", "stream"))
def test_kpool_compress_and_write_cache_input_layout_and_stream(accuracy_backend, variant):
    implementation, _, device = accuracy_backend
    case = Case(variant, 65, 4)
    original = make_inputs(case)
    device_values = list(value.to(device) if value is not None else None for value in original)
    if variant == "noncontiguous":
        for i in (1, 2, 3):
            value = device_values[i]
            device_values[i] = value.transpose(-1, -2).contiguous().transpose(-1, -2)
            assert not device_values[i].is_contiguous()
        result = run_case(case, device_values, implementation)
    else:
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            device_values[1].copy_(original[1].to(device))
            result = run_case(case, device_values, implementation)
        stream.synchronize()
    check(case, original, device_values, result)


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
@pytest.mark.parametrize("mode", ("write", "write_return", "return"))
@pytest.mark.parametrize("pool,n", ((4, 3), (4, 64), (16, 512), (16, 2049)))
def test_kpool_compress_and_write_cache_fullgraph(accuracy_backend, backend, mode, pool, n):
    implementation, _, device = accuracy_backend
    preload("kpool_compress_and_write_cache", implementation=implementation)
    case = Case("compile", n, pool, blocks=(n + PAGE - 1) // PAGE + 1, mode=mode)
    cache = make_inputs(case)[0].to(device)

    def run(cache, key, score, ape, loc):
        return F.kpool_compress_and_write_cache(
            cache,
            key,
            score,
            ape,
            loc,
            pool,
            return_compressed=mode != "write",
            write_cache=mode != "return",
            implementation=implementation,
        )

    try:
        compiled = compile_fullgraph(run, backend=backend)
        for step in (1, 2):
            _, key, score, ape, loc, mask = make_inputs(replace(case, seed=case.seed + step))
            loc = (loc + step) % (case.blocks * case.page)
            original = (cache.cpu().clone(), key, score, ape, loc, mask)
            values = (cache, *(value.to(device) for value in original[1:5]), None)
            result = compiled(*values[:5])
            check(case, original, values, result)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
@pytest.mark.parametrize("layout,n", (("cross_page", 5), ("all_masked", 5), ("linear", 0)))
def test_kpool_compress_and_write_cache_masked_fullgraph(accuracy_backend, backend, layout, n):
    implementation, _, device = accuracy_backend
    preload("kpool_compress_and_write_cache", implementation=implementation)
    case = Case("compile_mask", n, 4, layout=layout)
    original = make_inputs(case)
    values = tuple(value.to(device) if value is not None else None for value in original)

    def run(cache, key, score, ape, loc, mask):
        return F.kpool_compress_and_write_cache(
            cache,
            key,
            score,
            ape,
            loc,
            4,
            write_mask=mask,
            implementation=implementation,
        )

    try:
        compiled = compile_fullgraph(run, backend=backend)
        result = compiled(*values)
        check(case, original, values, result)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
def test_kpool_compress_and_write_cache_schema_and_fake(accuracy_backend):
    implementation, _, device = accuracy_backend
    preload("kpool_compress_and_write_cache", implementation=implementation)
    forward, _ = load_impl("kpool_compress_and_write_cache", implementation)
    if implementation == "torch":
        pytest.skip("Schema checks apply to the Triton custom operators")
    provider = import_module(forward.__module__)
    values = make_inputs(PERF_CASES[0])
    args = tuple(value.to(device) for value in values[:5])
    for leaf_name, mode in (
        ("_kpool_write_leaf", "write"),
        ("_kpool_write_return_leaf", "write_return"),
        ("_kpool_return_leaf", "return"),
    ):
        leaf = getattr(provider, leaf_name)
        schema = leaf._opoverload._schema
        assert bool(schema.arguments[0].alias_info and schema.arguments[0].alias_info.is_write) == (mode != "return")
        assert all(arg.alias_info is None for arg in schema.arguments[1:])
        leaf_args = (*args, 4, None, True) if mode == "write" else (*args, 4, True)
        torch.library.opcheck(leaf, leaf_args, test_utils=("test_schema", "test_faketensor"))
        with FakeTensorMode() as fake:
            result = F.kpool_compress_and_write_cache(
                *(fake.from_tensor(value) for value in args),
                4,
                return_compressed=mode != "write",
                write_cache=mode != "return",
                implementation=implementation,
            )
            if mode == "write":
                assert result is None
            else:
                key, scale = result
                assert key.shape == (1, HEAD) and key.dtype == FP8
                assert scale.shape == (1,) and scale.dtype == torch.float32
                assert key.device == args[1].device and scale.device == args[1].device


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
def test_kpool_compress_and_write_cache_fullgraph_rejects_unaligned_cache(accuracy_backend):
    implementation, _, device = accuracy_backend
    if implementation == "torch":
        pytest.skip("Runtime alignment check belongs to the Triton operator")
    preload("kpool_compress_and_write_cache", implementation=implementation)
    case = PERF_CASES[0]
    cache, key, score, ape, loc, _ = make_inputs(case)
    backing = torch.empty(cache.numel() + 1, dtype=torch.uint8, device=device)
    cache = backing[1:].view(cache.shape)
    key, score, ape, loc = (value.to(device) for value in (key, score, ape, loc))

    def run(cache, key, score, ape, loc):
        return F.kpool_compress_and_write_cache(cache, key, score, ape, loc, 4, implementation=implementation)

    try:
        compiled = compile_fullgraph(run)
        with pytest.raises(ValueError, match="4-byte aligned"):
            compiled(cache, key, score, ape, loc)
    finally:
        torch._dynamo.reset()
