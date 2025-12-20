import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoTopPFilter
from mojo_opset.backends.reference.operators.sampling import RefTopPFilter


@pytest.mark.parametrize(
    "logits, topk, topp, min_tokens_to_keep",
    [(torch.randn(120, 151936), 1000, 0.7, 1)],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_topp_filter(logits, topk, topp, min_tokens_to_keep):
    top_p_filter = MojoTopPFilter()
    top_p_filter_ref = RefTopPFilter()

    perf(lambda: top_p_filter_ref(logits, topp, min_tokens_to_keep, topk))  # noqa: F821
    perf(lambda: top_p_filter(logits, topp, min_tokens_to_keep, topk))  # noqa: F821
