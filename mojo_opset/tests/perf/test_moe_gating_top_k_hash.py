"""Performance benchmarks for DeepSeek-V4 hash-based MoE gating."""

import pytest
import torch

from mojo_opset import MojoMoEGatingTopKHash
from mojo_opset.tests.utils import auto_switch_platform, bypass_not_implemented

DSV4_VOCAB_SIZE = 129280
DSV4_SCORING = 2


@pytest.mark.parametrize(
    "experts, routed_scaling_factor",
    [(256, 1.5), (384, 2.5)],
)
@pytest.mark.parametrize(
    "num_tokens",
    [1, 16, 64, 128, 256, 512, 1024, 4096, 8192],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_dsv4_hash_gating(experts, routed_scaling_factor, num_tokens, dtype):
    k = 6
    x = torch.randn(num_tokens, experts, dtype=dtype)
    input_ids = torch.randint(0, DSV4_VOCAB_SIZE, (num_tokens,), dtype=torch.int64)
    tid2eid = torch.stack(
        [torch.randperm(experts, dtype=torch.int32)[:k] for _ in range(DSV4_VOCAB_SIZE)]
    )

    x = x.to("npu")
    input_ids = input_ids.to("npu")
    tid2eid = tid2eid.to("npu")
    op = MojoMoEGatingTopKHash(
        k=k,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-6,
        norm_type=DSV4_SCORING,
    ).to(x.device)

    for _ in range(30):
        op(x, input_ids, tid2eid)
    torch.npu.synchronize()

    perf(lambda: op(x, input_ids, tid2eid))  # noqa: F821


@pytest.mark.parametrize("norm_type", [0, 1, 2])
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_dsv4_norm_type_comparison(norm_type, num_tokens, dtype):
    experts = 256
    k = 6
    x = torch.randn(num_tokens, experts, dtype=dtype)
    input_ids = torch.randint(0, DSV4_VOCAB_SIZE, (num_tokens,), dtype=torch.int64)
    tid2eid = torch.stack(
        [torch.randperm(experts, dtype=torch.int32)[:k] for _ in range(DSV4_VOCAB_SIZE)]
    )

    x = x.to("npu")
    input_ids = input_ids.to("npu")
    tid2eid = tid2eid.to("npu")
    op = MojoMoEGatingTopKHash(
        k=k,
        norm_type=norm_type,
        routed_scaling_factor=1.5,
        eps=1e-6,
    ).to(x.device)

    for _ in range(30):
        op(x, input_ids, tid2eid)
    torch.npu.synchronize()

    perf(lambda: op(x, input_ids, tid2eid))  # noqa: F821
