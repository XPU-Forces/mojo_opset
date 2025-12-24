import random

from typing import List
from typing import Union

import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoApplyPenaltiesTempurate
from mojo_opset import MojoTopPFilter
from mojo_opset import MojoTopPSampling
from mojo_opset import MojoRejectSampling
from mojo_opset.backends.reference.operators.sampling import RefApplyPenaltiesTempurate
from mojo_opset.backends.reference.operators.sampling import RefTopPFilter
from mojo_opset.backends.reference.operators.sampling import RefTopPSampling
from mojo_opset.backends.reference.operators.sampling import RefRejectSampling

@pytest.mark.parametrize(
    "logits, topk, topp, min_tokens_to_keep",
    [(torch.randn(20, 151936), 1000, 0.75, 1)],
)
@auto_switch_platform()
@bypass_not_implemented
def test_topp_sampling(logits, topk, topp, min_tokens_to_keep):
    top_p_sampling = MojoTopPSampling(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)
    top_p_sampling_ref = RefTopPSampling(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)

    top_p_sampling_ref.forward_diff_with(top_p_sampling, logits)


@pytest.mark.parametrize(
    "logits, topk, topp, min_tokens_to_keep",
    [(torch.randn(20, 151936), 1000, 0.75, 1)],
)
@auto_switch_platform()
@bypass_not_implemented
def test_topp_filter(logits, topk, topp, min_tokens_to_keep):
    top_p_filter = MojoTopPFilter()
    top_p_filter_ref = RefTopPFilter()

    top_p_filter_ref.forward_diff_with(
        top_p_filter, logits=logits, top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk
    )

@pytest.mark.parametrize(
    "target_logits, draft_tokens, draft_probs, spec_step, top_p, rand_top_k",
    [(torch.rand((15, 4, 155136), dtype=torch.float32), torch.randint(0, 155136, (15, 3)), torch.rand((15, 3), dtype=torch.float32), 3, 0.7, 100)],
)
@auto_switch_platform()
@bypass_not_implemented
def test_reject_sampler(target_logits, draft_tokens, draft_probs, spec_step, top_p, rand_top_k):
    torch.manual_seed(42)

    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    
    reject_sampling = MojoRejectSampling()
    ref_reject_sampling = RefRejectSampling()

    ref_reject_sampling.forward_diff_with(
        reject_sampling, target_logits, draft_tokens, draft_probs, spec_step, top_p, rand_top_k
    )

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
    "logits",
    [(torch.randn(20, 151936))],
)
@auto_switch_platform()
@bypass_not_implemented
def test_apply_penalties_temp(logits):
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
    apply_penalties_ref = RefApplyPenaltiesTempurate()

    apply_penalties_ref.forward_diff_with(apply_penalties, logits, token_freqs, pres_pens, freq_pens, rep_pens, temps)
