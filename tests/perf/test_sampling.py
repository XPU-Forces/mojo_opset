import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoJoinProbRejectSampling
from mojo_opset import MojoRejectSampling
from mojo_opset import MojoTopPFilter


@pytest.mark.parametrize(
    "logits, topk, topp, min_tokens_to_keep",
    [
        (torch.randn(120, 151936), 1000, 0.7, 1),
        (torch.randn(15, 155136), 100, 0.7, 1),
        (torch.randn(18, 155136), 100, 0.7, 1),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_topp_filter(logits, topk, topp, min_tokens_to_keep):
    top_p_filter = MojoTopPFilter()
    top_p_filter_ref = MojoTopPFilter()._registry.get("torch")()

    perf(lambda: top_p_filter_ref(logits, topp, min_tokens_to_keep, topk))  # noqa: F821
    perf(lambda: top_p_filter(logits, topp, min_tokens_to_keep, topk))  # noqa: F821


@pytest.mark.parametrize(
    "target_logits, draft_tokens, draft_probs, spec_step",
    [
        (
            torch.randn((15, 4, 155136), dtype=torch.float32),
            torch.randint(0, 155136, (15, 3)),
            torch.ones((15, 3), dtype=torch.float32),
            3,
        )
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_reject_sampling(target_logits, draft_tokens, draft_probs, spec_step):
    torch.manual_seed(42)

    ref_reject_sampling = MojoRejectSampling()._registry.get("torch")()
    reject_sampling = MojoRejectSampling()

    perf(lambda: ref_reject_sampling(target_logits, draft_tokens, draft_probs))  # noqa: F821
    perf(lambda: reject_sampling(target_logits, draft_tokens, draft_probs))  # noqa: F821


@pytest.mark.parametrize(
    "target_logits, draft_tokens, draft_probs, spec_step, top_p, rand_top_k",
    [
        (
            torch.rand((15, 4, 155136), dtype=torch.float32),
            torch.randint(0, 155136, (15, 3)),
            torch.rand((15, 3), dtype=torch.float32),
            3,
            0.7,
            100,
        )
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_magic_reject_sampling(target_logits, draft_tokens, draft_probs, spec_step, top_p, rand_top_k):
    torch.manual_seed(42)

    ref_reject_sampling = MojoJoinProbRejectSampling()._registry.get("torch")()
    reject_sampling = MojoJoinProbRejectSampling()

    perf(lambda: ref_reject_sampling(target_logits, draft_tokens, draft_probs))  # noqa: F821
    perf(lambda: reject_sampling(target_logits, draft_tokens, draft_probs))  # noqa: F821
