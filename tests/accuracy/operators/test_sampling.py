import random

from typing import List
from typing import Union

import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset import MojoApplyPenaltiesTempurate
from mojo_opset import MojoJoinProbRejectSampling
from mojo_opset import MojoRejectSampling
from mojo_opset import MojoTopPFilter
from mojo_opset import MojoTopPSampling


@pytest.mark.parametrize(
    "shape, topk, topp, min_tokens_to_keep",
    [((20, 151936), 1000, 0.75, 1)],
)
@bypass_not_implemented
def test_topp_sampling(shape, topk, topp, min_tokens_to_keep):
    logits = torch.randn(shape, dtype=torch.float32)
    top_p_sampling = MojoTopPSampling(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)
    top_p_sampling_ref = MojoTopPSampling._registry.get("torch")(
        top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk
    )

    top_p_sampling.forward_diff_with(top_p_sampling_ref, logits)


@pytest.mark.parametrize(
    "shape, topk, topp, min_tokens_to_keep",
    [((20, 151936), 1000, 0.75, 1), ((60, 155136), 100, 0.7, 1)],
)
@bypass_not_implemented
def test_topp_filter(shape, topk, topp, min_tokens_to_keep):
    logits = torch.randn(shape, dtype=torch.float32)
    top_p_filter = MojoTopPFilter()
    top_p_filter_ref = MojoTopPFilter._registry.get("torch")()

    top_p_filter.forward_diff_with(
        top_p_filter_ref,
        logits=logits,
        top_p=topp,
        min_tokens_to_keep=min_tokens_to_keep,
        rand_top_k=topk,
    )


@pytest.mark.parametrize(
    "batch_size, vocab_size, spec_step",
    [(15, 155136, 3)],
)
@bypass_not_implemented
def test_reject_sampler(batch_size, vocab_size, spec_step):
    torch.manual_seed(42)
    target_logits = torch.randn((batch_size, 1 + spec_step, vocab_size), dtype=torch.float32)
    draft_tokens = torch.randint(0, vocab_size, (batch_size, spec_step))
    draft_probs = torch.ones((batch_size, spec_step), dtype=torch.float32)

    reject_sampling = MojoRejectSampling()
    ref_reject_sampling = MojoRejectSampling._registry.get("torch")()

    batch_size = target_logits.shape[0]

    ref_token_ids, ref_accept_len = ref_reject_sampling(target_logits, draft_tokens, draft_probs, 42)
    ttx_token_ids, ttx_accept_len = reject_sampling(target_logits, draft_tokens, draft_probs, 42)

    range_mask = torch.arange(spec_step + 1).expand(batch_size, -1).to(ref_accept_len.device)

    ref_mask = range_mask < ref_accept_len.unsqueeze(-1).expand(-1, spec_step + 1)
    ref_token_ids = ref_token_ids * ref_mask

    ttx_mask = range_mask < ttx_accept_len.unsqueeze(-1).expand(-1, spec_step + 1)
    ttx_token_ids = ttx_token_ids * ttx_mask

    torch.testing.assert_close(ref_token_ids.to(torch.float32), ttx_token_ids.to(torch.float32), atol=1e-2, rtol=1e-2)

    torch.testing.assert_close(ref_accept_len.to(torch.float32), ttx_accept_len.to(torch.float32), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "batch_size, vocab_size, spec_step",
    [(15, 155136, 3)],
)
@bypass_not_implemented
def test_join_prob_reject_sampler(batch_size, vocab_size, spec_step):
    torch.manual_seed(42)
    target_logits = torch.randn((batch_size, 1 + spec_step, vocab_size), dtype=torch.float32)
    draft_tokens = torch.randint(0, vocab_size, (batch_size, spec_step))
    draft_probs = torch.ones((batch_size, spec_step), dtype=torch.float32)

    reject_join_prob_sampling = MojoJoinProbRejectSampling()
    ref_join_prob_sampling = MojoJoinProbRejectSampling._registry.get("torch")()

    batch_size = target_logits.shape[0]

    ref_token_ids, ref_accept_len = ref_join_prob_sampling(target_logits, draft_tokens, draft_probs, 42)
    ttx_token_ids, ttx_accept_len = reject_join_prob_sampling(target_logits, draft_tokens, draft_probs, 42)

    range_mask = torch.arange(spec_step + 1).expand(batch_size, -1).to(ref_accept_len.device)

    ref_mask = range_mask < ref_accept_len.unsqueeze(-1).expand(-1, spec_step + 1)
    ref_token_ids = ref_token_ids * ref_mask

    ttx_mask = range_mask < ttx_accept_len.unsqueeze(-1).expand(-1, spec_step + 1)
    ttx_token_ids = ttx_token_ids * ttx_mask

    torch.testing.assert_close(ref_token_ids.to(torch.float32), ttx_token_ids.to(torch.float32), atol=1e-2, rtol=1e-2)

    torch.testing.assert_close(ref_accept_len.to(torch.float32), ttx_accept_len.to(torch.float32), atol=1e-2, rtol=1e-2)


def split_batch_to_list(x: torch.Tensor) -> List[Union[None, torch.Tensor]]:
    result: List[Union[None, torch.Tensor]] = []
    batch_size = x.size(0)

    non_zero_mask = (x != 0).any(dim=1)

    for i in range(batch_size):
        if not non_zero_mask[i]:
            result.append(None)
        else:
            result.append(x[i])

    return result


@pytest.mark.parametrize(
    "shape",
    [(20, 151936)],
)
@bypass_not_implemented
def test_apply_penalties_temp(shape):
    logits = torch.randn(shape, dtype=torch.float32)
    BATCH_SIZE = logits.shape[0]
    VOCAB_SIZE = logits.shape[1]
    token_freqs = torch.randint(0, 5, (BATCH_SIZE, VOCAB_SIZE), device=logits.device, dtype=torch.int32)
    mask = torch.rand((BATCH_SIZE, VOCAB_SIZE), device=logits.device) > 0.05
    token_freqs[mask] = 0
    token_freqs = split_batch_to_list(token_freqs)

    freq_pens = [random.uniform(-0.5, 0.5) for _ in range(BATCH_SIZE)]
    pres_pens = [random.uniform(-0.5, 0.5) for _ in range(BATCH_SIZE)]
    rep_pens = [random.uniform(0.5, 3.0) for _ in range(BATCH_SIZE)]
    temps = [random.uniform(0.1, 2.0) for _ in range(BATCH_SIZE)]

    apply_penalties = MojoApplyPenaltiesTempurate()
    apply_penalties_ref = MojoApplyPenaltiesTempurate._registry.get("torch")()

    apply_penalties.forward_diff_with(apply_penalties_ref, logits, token_freqs, pres_pens, freq_pens, rep_pens, temps)
