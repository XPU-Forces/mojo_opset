"""rope_quant_insert performance cases (fp8 / mxfp8 / plain records, production shapes)."""

import pytest
import torch

from mojo_opset import functions

MAX_POS = 4096
VARIANTS = [("fp8", 584), ("mxfp8", 528), ("plain_bf16", 512)]
CASES = [
    ("decode_256_cr2", 256, 2),
    ("mid_2048_cr2", 2048, 2),
    ("prefill_4096_cr1", 4096, 1),
    # Non-64-aligned token counts: exercise the masked tail of the RPP unroll
    # and the partial-group fallback of the contiguous-run path.
    ("mid_1023_cr2", 1023, 2),
    ("decode_255_cr2", 255, 2),
]


def _inputs(variant, record, num_tokens, compress_ratio, device, block=64, seed=0):
    torch.manual_seed(seed)
    latent = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, MAX_POS, (num_tokens,), dtype=torch.int64, device=device)
    positions[0::3] = (positions[0::3] // compress_ratio) * compress_ratio
    positions[1::3] = (positions[1::3] // compress_ratio) * compress_ratio + compress_ratio - 1
    angles = torch.randn(MAX_POS, 32, dtype=torch.float32, device=device)
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1)
    slots = torch.randperm(num_tokens * 2 + block, device=device)[:num_tokens].to(torch.int64)
    if record != 512:
        page_elems = ((block * record + 511) // 512) * 512
        backing = torch.randint(0, 256, (num_tokens * 2 // block + 1, page_elems), dtype=torch.uint8, device=device)
    else:
        page_elems = block * record
        dtype = torch.bfloat16
        backing = torch.randint(
            0, 256, ((num_tokens * 2 // block + 1) * page_elems * dtype.itemsize,),
            dtype=torch.uint8, device=device,
        ).view(dtype).view(num_tokens * 2 // block + 1, page_elems)
    cache = backing.as_strided(
        (backing.shape[0], block, record), (backing.shape[1], record, 1)
    )
    return latent, positions, cos_sin, cache, slots


@pytest.mark.api("functions.rope_quant_insert")
@pytest.mark.parametrize("case,num_tokens,compress_ratio", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("variant,record", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_rope_quant_insert(benchmark, perf_environment, variant, record, num_tokens, compress_ratio, case):
    _, device, target, implementation = perf_environment
    if str(target).startswith("npu.a2") and variant != "plain_bf16":
        pytest.skip("910B (A2) has no FP8")
    latent, positions, cos_sin, cache, slots = _inputs(
        variant, record, num_tokens, compress_ratio, device
    )

    def factory():
        return lambda: functions.rope_quant_insert(
            latent, positions, cos_sin, cache, slots, compress_ratio,
            implementation=implementation,
        )

    benchmark(
        factory=factory,
        op="rope_quant_insert",
        case=case,
        variant=variant,
        tokens=num_tokens,
        compress_ratio=compress_ratio,
        dtype="uint8" if record != 512 else "bfloat16",
        phase="forward",
    )
