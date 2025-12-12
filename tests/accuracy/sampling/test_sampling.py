import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoTopPFilter
from mojo_opset import MojoTopPSampling


@pytest.mark.parametrize(
    "logits, topk, topp, min_tokens_to_keep",
    [(torch.randn(20, 151936), 1000, 0.75, 1)],
)
@auto_switch_platform()
@bypass_not_implemented
def test_topp_sampling(logits, topk, topp, min_tokens_to_keep):
    op = MojoTopPSampling(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)

    op.forward_diff(logits)


@pytest.mark.parametrize(
    "logits, topk, topp, min_tokens_to_keep",
    [(torch.randn(20, 151936, dtype=torch.bfloat16), 1000, 0.75, 1)],
)
@auto_switch_platform()
@bypass_not_implemented
def test_topp_filter(logits, topk, topp, min_tokens_to_keep):
    ref_op = MojoTopPFilter(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk, backend="reference")
    ttx_op = MojoTopPFilter(top_p=topp, min_tokens_to_keep=min_tokens_to_keep, rand_top_k=topk)

    ttx_op.forward_diff_with_op(ref_op, logits)
