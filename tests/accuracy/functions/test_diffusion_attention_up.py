import math
import random

import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import auto_switch_platform

from mojo_opset.experimental import MojoDiffusionAttentionUpFunction


def generate_test_data(
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    max_seq_length: int,
    block_size: int,
    sample_min: int,
    sample_max: int,
    # device: str = "npu:0",
):
    assert sample_min > 0 and sample_max >= sample_min and max_seq_length >= sample_max

    query = torch.randn(max_seq_length * 2, q_head_num, head_dim, dtype=torch.bfloat16, requires_grad=True)
    # , device=device)
    key = torch.randn(max_seq_length * 2, kv_head_num, head_dim, dtype=torch.bfloat16, requires_grad=True)
    # , device=device)
    value = torch.randn(max_seq_length * 2, kv_head_num, head_dim, dtype=torch.bfloat16, requires_grad=True)
    # , device=device)

    rest_size = max_seq_length
    seqs = []
    while rest_size > sample_max:
        seqlen = random.randint(sample_min, sample_max)
        seqs.append(seqlen)
        rest_size -= seqlen
    if rest_size > 0:
        seqs.append(rest_size)
    cu = [0]
    for seqlen in seqs:
        cu.append(cu[-1] + seqlen)
    cu_seqlen = torch.tensor(cu[1:], dtype=torch.int32)
    # , device=device)
    scale = 1.0 / math.sqrt(head_dim)

    return query, key, value, cu_seqlen, scale, block_size


@pytest.mark.parametrize(
    "query, key, value, cu_seqlen, scale, block_size",
    [
        pytest.param(
            *generate_test_data(
                q_head_num=8,
                kv_head_num=2,
                head_dim=128,
                max_seq_length=8192,
                block_size=8,
                sample_min=128,
                sample_max=1024,
            )
        ),
    ],
)
# @pytest.mark.skip
@auto_switch_platform()
def test_diffusion_attention_up_func(query, key, value, cu_seqlen, scale, block_size):
    ctx = MockFunctionCtx()
    o = MojoDiffusionAttentionUpFunction.forward(ctx, query, key, value, cu_seqlen, scale, block_size)

    ctx_ref = MockFunctionCtx()
    o_ref = MojoDiffusionAttentionUpFunction._registry.get("torch").forward(
        ctx_ref, query, key, value, cu_seqlen, scale, block_size
    )

    assert_close(o, o_ref)

    do = torch.rand_like(o)
    grads = MojoDiffusionAttentionUpFunction.backward(ctx, do)

    grads_ref = MojoDiffusionAttentionUpFunction._registry.get("torch").backward(ctx_ref, do)

    assert_close(grads, grads_ref)
