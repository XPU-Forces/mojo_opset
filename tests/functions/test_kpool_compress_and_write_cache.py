from dataclasses import dataclass

import pytest
import torch

from mojo_opset import Target
from mojo_opset import functions as F

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
