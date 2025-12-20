import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoTopPFilter
from mojo_opset.backends.reference.sample import RefTopPFilter


@pytest.mark.parametrize(
    "logits, topk, topp, min_tokens_to_keep",
    [(torch.randn(120, 151936), 1000, 0.7, 1)],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_topp_filter(logits, topk, topp, min_tokens_to_keep):
    top_p_filter = MojoTopPFilter(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)
    top_p_filter_ref = RefTopPFilter(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)

    perf(lambda: top_p_filter_ref(logits))  # noqa: F821
    perf(lambda: top_p_filter(logits))  # noqa: F821
