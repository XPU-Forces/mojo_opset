import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoTopPFilter
from mojo_opset import MojoTopPSampling
from mojo_opset.backends.reference.sample import RefTopPFilter
from mojo_opset.backends.reference.sample import RefTopPSampling


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
