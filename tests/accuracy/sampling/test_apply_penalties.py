import random

from typing import List
from typing import Union

import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoApplyPenalties


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
def test_topp_sampling(logits):
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

    op = MojoApplyPenalties()

    op.forward_diff(logits, token_freqs, pres_pens, freq_pens, rep_pens, temps)
