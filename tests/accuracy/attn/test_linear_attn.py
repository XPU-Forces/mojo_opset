import random

import pytest
import torch
import torch.nn.functional as F

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoDecodeGDN
from mojo_opset import MojoPrefillGDN

dtypes = [torch.float16, torch.bfloat16]

prefill_cases = [
    (1, 128, 24, 4),
    (1, 512, 24, 4),
    (1, 1024, 16, 8),
    (1, 8192, 24, 4),
]


def make_random_cu_seqlens(B, T):
    B = random.randint(1, 4)
    splits = sorted(random.sample(range(1, T), B - 1))
    cu = [0] + splits + [T]
    return torch.tensor(cu, dtype=torch.long)


perfill_param_list = []
for B, T, H, Hk in prefill_cases:
    for dtype in dtypes:
        q = torch.randn(B, T, H, 128, dtype=dtype)
        k = torch.randn(B, T, Hk, 128, dtype=dtype)
        v = torch.randn(B, T, Hk, 256, dtype=dtype)

        beta = torch.rand(B, T, Hk, dtype=torch.float32).sigmoid()
        g = F.logsigmoid(torch.rand(B, T, Hk, dtype=dtype))

        cu_seqlens = make_random_cu_seqlens(B, T)

        perfill_param_list.append((q, k, v, beta, g, cu_seqlens))


@pytest.mark.parametrize(
    "q, k, v, beta, g, cu_seqlens",
    perfill_param_list,
)
@auto_switch_platform()
@bypass_not_implemented
def test_prefill_gdn(q, k, v, beta, g, cu_seqlens):
    op = MojoPrefillGDN(use_qk_l2norm_in_kernel=True, output_final_state=True)

    # for testing
    op.forward_diff(q, k, v, g, beta, cu_seqlens, atol=1e-1, rtol=1e-1)
    # for use
    op(q, k, v, g, beta, cu_seqlens)


decode_cases = [
    (1, 1, 24, 4, 128, 256),
    (2, 1, 16, 8, 64, 128),
    (4, 1, 32, 4, 128, 256),
    (32, 1, 24, 4, 128, 256),
    (256, 1, 24, 4, 128, 256),
]

decode_param_list = []
for B, T, H, Hk, K, V in decode_cases:
    for dtype in dtypes:
        q = torch.randn(B, T, H, 128, dtype=dtype)
        k = torch.randn(B, T, Hk, 128, dtype=dtype)
        v = torch.randn(B, T, Hk, 256, dtype=dtype)

        beta = torch.rand(B, T, Hk, dtype=torch.float32).sigmoid()
        g = F.logsigmoid(torch.rand(B, T, Hk, dtype=dtype))

        init_state = torch.randn(B, Hk, 128, 256, dtype=torch.float32)
        cu_seqlens = None

        decode_param_list.append((q, k, v, beta, g, init_state, cu_seqlens))


@pytest.mark.parametrize(
    "q, k, v, beta, g, init_state, cu_seqlens",
    decode_param_list,
)
@auto_switch_platform()
@bypass_not_implemented
def test_decode_gdn(q, k, v, beta, g, init_state, cu_seqlens):
    op = MojoDecodeGDN(use_qk_l2norm_in_kernel=True)

    # for testing
    op.forward_diff(q, k, v, g, beta, init_state, cu_seqlens, atol=1e-1, rtol=1e-1)
    # for use
    op(q, k, v, g, beta, init_state, cu_seqlens)
