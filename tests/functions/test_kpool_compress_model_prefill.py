"""Synthetic prefill coverage for the vLLM KPool compression caller.

The fixtures model pool-ending candidate rows, batched requests and paged cache
capacity. Request lengths and context limits are synthetic test parameters.
"""

from dataclasses import dataclass

import pytest
import torch

from mojo_opset import Target
from tests.functions.test_kpool_compress_and_write_cache import HEAD
from tests.functions.test_kpool_compress_and_write_cache import PAGE
from tests.functions.test_kpool_compress_and_write_cache import Case
from tests.functions.test_kpool_compress_and_write_cache import check
from tests.functions.test_kpool_compress_and_write_cache import run_case


@dataclass(frozen=True)
class ModelPrefillCase:
    case_id: str
    pool: int
    request_lengths: tuple[int, ...]
    context_limit: int = 16384
    start_position: int = 0

    @property
    def n(self):
        return sum(self.request_lengths)

    @property
    def writes(self):
        return sum(length // self.pool for length in self.request_lengths)

    @property
    def pages_per_request(self):
        slots = (self.context_limit + self.pool - 1) // self.pool
        return (slots + PAGE - 1) // PAGE

    @property
    def operator_case(self):
        return Case(self.case_id, self.n, self.pool, blocks=len(self.request_lengths) * self.pages_per_request)


MODEL_PREFILL_CASES = [
    ModelPrefillCase("A_p4_t4", 4, (4,)),
    ModelPrefillCase("A_p4_t16", 4, (16,)),
    ModelPrefillCase("A_p4_t64", 4, (64,)),
    ModelPrefillCase("A_p4_t256", 4, (256,)),
    ModelPrefillCase("A_p4_t512", 4, (512,)),
    ModelPrefillCase("A_p4_t2048", 4, (2048,)),
    ModelPrefillCase("A_p4_t4096", 4, (4096,)),
    ModelPrefillCase("A_p4_t8192", 4, (8192,)),
    ModelPrefillCase("A_p4_t16384", 4, (16384,)),
    ModelPrefillCase("A_p4_t24576", 4, (24576,), context_limit=24576),
    ModelPrefillCase("A_p4_t32768", 4, (32768,), context_limit=32768),
    ModelPrefillCase("B_p4_t8192_b32_equal", 4, (256,) * 32),
    ModelPrefillCase("B_p4_t8192_b32_ragged", 4, (255,) * 31 + (287,)),
    ModelPrefillCase("B_p4_t16384_b32_equal", 4, (512,) * 32),
    ModelPrefillCase("B_p4_t16384_b32_ragged", 4, (511,) * 31 + (543,)),
    ModelPrefillCase("B_p4_t24576_b256_equal", 4, (96,) * 256),
    ModelPrefillCase("B_p4_t24576_b256_ragged", 4, (95,) * 255 + (351,)),
    ModelPrefillCase("B_p4_t32768_b128_equal", 4, (256,) * 128),
    ModelPrefillCase("B_p4_t32768_b128_ragged", 4, (255,) * 127 + (383,)),
    ModelPrefillCase("C_p4_t8192_ctx16384", 4, (8192,), start_position=8192),
    ModelPrefillCase("C_p4_t8192_ctx131072", 4, (8192,), context_limit=131072, start_position=122880),
    ModelPrefillCase("C_p4_t8192_ctx262144", 4, (8192,), context_limit=262144, start_position=253952),
    ModelPrefillCase("D_p4_t5", 4, (5,)),
    ModelPrefillCase("D_p4_t255", 4, (255,)),
    ModelPrefillCase("D_p4_t257", 4, (257,)),
    ModelPrefillCase("D_p4_t260", 4, (260,)),
    ModelPrefillCase("D_p4_t48_b16_all_masked", 4, (3,) * 16),
    ModelPrefillCase("E_p4_t65536_stress", 4, (65536,), context_limit=65536),
    ModelPrefillCase("E_p16_t16_compat", 16, (16,)),
    ModelPrefillCase("E_p16_t17_compat", 16, (17,)),
    ModelPrefillCase("E_p16_t8192_compat", 16, (8192,)),
    ModelPrefillCase("E_p16_t65536_compat", 16, (65536,), context_limit=65536),
]
MODEL_SKIP_CASES = [
    ModelPrefillCase("D_caller_t0", 4, (0,)),
    ModelPrefillCase("D_caller_t1", 4, (1,)),
    ModelPrefillCase("D_caller_t3", 4, (3,)),
]


def make_model_prefill_inputs(case):
    """Materialize the caller's overlapping gathers and pool-granular slot mask.

    Each request owns a shuffled, disjoint physical page table. Positions start
    on pool boundaries, as required by the upstream caller. Input preparation
    stays outside operator-only performance timing, including E2E timing.
    """
    assert case.pool in (4, 16)
    assert case.start_position % case.pool == 0
    assert all(0 <= length <= case.context_limit - case.start_position for length in case.request_lengths)
    if case.n < case.pool:
        return None  # The model caller returns before invoking the operator.

    rng = torch.Generator().manual_seed(20261119)
    key = torch.randn((case.n, HEAD), generator=rng).bfloat16()
    score = torch.randn((case.n, HEAD), generator=rng).bfloat16()
    ape = torch.randn((case.pool, HEAD), generator=rng)
    op_case = case.operator_case
    cache = torch.full((op_case.blocks, PAGE, HEAD + 4), 0xA5, dtype=torch.uint8)
    page_table = torch.randperm(op_case.blocks, generator=rng).reshape(-1, case.pages_per_request)
    loc = torch.full((case.n,), -1, dtype=torch.int64)
    begin = 0
    for request, length in enumerate(case.request_lengths):
        local = torch.arange(length)
        position = case.start_position + local
        complete = position % case.pool == case.pool - 1
        pool_id = position[complete] // case.pool
        physical_page = page_table[request, pool_id // PAGE]
        loc[begin + local[complete]] = physical_page * PAGE + pool_id % PAGE
        begin += length

    pos = torch.arange(case.n)
    mask = (loc >= 0) & (pos >= case.pool - 1)
    indices = (pos - (case.pool - 1)).clamp_min(0)[:, None] + torch.arange(case.pool)[None, :]
    return cache, key[indices], score[indices], ape, loc, mask


@pytest.mark.reference
@pytest.mark.parametrize("case", MODEL_PREFILL_CASES, ids=lambda case: case.case_id)
def test_model_prefill_fixture_contract(case):
    cache, key, score, _, loc, mask = make_model_prefill_inputs(case)
    assert key.shape == score.shape == (case.n, case.pool, HEAD)
    assert mask.sum().item() == case.writes
    assert (loc[~mask] == -1).all()
    assert torch.unique(loc[mask]).numel() == case.writes
    assert ((loc[mask] >= 0) & (loc[mask] < cache.shape[0] * PAGE)).all()
    # Independently enumerate each request's legal complete pools. Every valid
    # gathered window must belong to that request and end at the selected row.
    begin = 0
    selected = []
    for length in case.request_lengths:
        selected.extend(range(begin + case.pool - 1, begin + length, case.pool))
        begin += length
    assert mask.nonzero().flatten().tolist() == selected
    if selected:
        rng = torch.Generator().manual_seed(20261119)
        flat_key = torch.randn((case.n, HEAD), generator=rng).bfloat16()
        flat_score = torch.randn((case.n, HEAD), generator=rng).bfloat16()
        window = torch.tensor(selected)[:, None] - case.pool + 1 + torch.arange(case.pool)[None, :]
        expected_key, expected_score = flat_key[window], flat_score[window]
        assert torch.equal(key[mask], expected_key)
        assert torch.equal(score[mask], expected_score)


@pytest.mark.reference
@pytest.mark.parametrize("case", MODEL_SKIP_CASES, ids=lambda case: case.case_id)
def test_model_prefill_short_batch_skips_operator(case):
    assert make_model_prefill_inputs(case) is None


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", MODEL_PREFILL_CASES, ids=lambda case: case.case_id)
def test_kpool_compress_model_prefill_accuracy(accuracy_backend, case):
    implementation, target, device = accuracy_backend
    if Target.parse(target).arch_key not in {"npu.a2", "npu.a5"}:
        pytest.skip("KPool compression kernels support Ascend A2 and A5")
    original = make_model_prefill_inputs(case)
    cpu_values = tuple(value.clone() for value in original)
    check(case.operator_case, original, cpu_values, run_case(case.operator_case, cpu_values, "torch"))
    device_values = tuple(value.to(device) for value in original)
    check(case.operator_case, original, device_values, run_case(case.operator_case, device_values, implementation))
