"""Synthetic model-call coverage, separate from the original 12 dense-pool cases.

The caller follows vLLM 0e5b2ba's nvidia/sparse_indexer.py::_kpool_compress_insert:
N is the current prefill token count; only pool-ending candidate rows are valid.
The recipe's tested deployment uses 8192 batched tokens, 16 sequences and a 16384
context. The additional 1M-context case uses the checkpoint configuration limit,
still with an 8192-token step. These are source-derived fixtures, not model traces.

Sources: vllm-project/vllm@0e5b2ba118cfb3763a3640942e02dd8d93e8ed6c,
vllm/models/glm5next/nvidia/sparse_indexer.py and recipes/models/zai-org/GLM-5.3-Flash.yaml;
huggingface/transformers@5f6629d89508568afaadf5d60cace11ec0edebbc,
src/transformers/models/glm5_next/configuration_glm5_next.py.
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


# Pool 16 is the model default. Pool 4 is explicitly compatibility coverage.
MODEL_PREFILL_CASES = [
    ModelPrefillCase("model_p16_first_pool", 16, (16,)),
    ModelPrefillCase("model_p16_partial_tail", 16, (17,)),
    ModelPrefillCase("model_p16_before_page", 16, (1023,)),
    ModelPrefillCase("model_p16_full_page", 16, (1024,)),
    ModelPrefillCase("model_p16_after_page", 16, (1025,)),
    ModelPrefillCase("model_p16_half_token_budget", 16, (4096,)),
    ModelPrefillCase("model_p16_token_budget_context_end", 16, (8192,), start_position=8192),
    ModelPrefillCase("model_p16_token_budget_mixed16", 16, (511,) * 15 + (527,)),
    ModelPrefillCase("model_p16_context1m_last_chunk", 16, (8192,), context_limit=1048576, start_position=1040384),
    ModelPrefillCase("compat_p4_first_pool", 4, (4,)),
    ModelPrefillCase("compat_p4_partial_tail", 4, (5,)),
    ModelPrefillCase("compat_p4_full_page", 4, (256,)),
    ModelPrefillCase("compat_p4_token_budget_mixed16", 4, (511,) * 15 + (527,)),
]
NO_COMPLETION_CASE = ModelPrefillCase("model_p16_no_complete_request", 16, (15,) * 16)
MODEL_SKIP_CASES = [
    ModelPrefillCase("model_empty", 16, (0,)),
    ModelPrefillCase("model_p16_short_batch", 16, (15,)),
    ModelPrefillCase("compat_p4_short_batch", 4, (3,)),
]


def make_model_prefill_inputs(case):
    """Materialize the caller's overlapping gathers and pool-granular slot mask.

    Each request owns a shuffled, disjoint physical page table. Positions start
    on pool boundaries, as required by the upstream caller. Input preparation
    stays outside operator-only performance timing, including E2E timing.
    """
    assert case.pool in (4, 16) and len(case.request_lengths) <= 16
    assert 0 <= case.n <= 8192 and case.start_position % case.pool == 0
    assert all(0 <= length <= case.context_limit - case.start_position for length in case.request_lengths)
    if case.n < case.pool:
        return None  # The model caller returns before invoking the operator.

    rng = torch.Generator().manual_seed(20261118)
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
@pytest.mark.parametrize("case", MODEL_PREFILL_CASES + [NO_COMPLETION_CASE], ids=lambda case: case.case_id)
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
        rng = torch.Generator().manual_seed(20261118)
        flat_key = torch.randn((case.n, HEAD), generator=rng).bfloat16()
        flat_score = torch.randn((case.n, HEAD), generator=rng).bfloat16()
        expected_key = torch.stack([flat_key[end - case.pool + 1 : end + 1] for end in selected])
        expected_score = torch.stack([flat_score[end - case.pool + 1 : end + 1] for end in selected])
        assert torch.equal(key[mask], expected_key)
        assert torch.equal(score[mask], expected_score)


@pytest.mark.reference
@pytest.mark.parametrize("case", MODEL_SKIP_CASES, ids=lambda case: case.case_id)
def test_model_prefill_short_batch_skips_operator(case):
    assert make_model_prefill_inputs(case) is None


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", MODEL_PREFILL_CASES + [NO_COMPLETION_CASE], ids=lambda case: case.case_id)
def test_kpool_compress_model_prefill_accuracy(accuracy_backend, case):
    implementation, target, device = accuracy_backend
    if Target.parse(target).arch_key not in {"npu.a2", "npu.a5"}:
        pytest.skip("KPool compression kernels support Ascend A2 and A5")
    original = make_model_prefill_inputs(case)
    cpu_values = tuple(value.clone() for value in original)
    check(case.operator_case, original, cpu_values, run_case(case.operator_case, cpu_values, "torch"))
    device_values = tuple(value.to(device) for value in original)
    check(case.operator_case, original, device_values, run_case(case.operator_case, device_values, implementation))
