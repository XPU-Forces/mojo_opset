"""Performance cases for functions.l2_norm.

Shapes follow real checkpoint configurations instead of a synthetic sweep:

    Qwen3-32B     hidden_size=5120  head_dim=128
    Qwen3-Next    hidden_size=2048  head_dim=256 (Gated Attention)
    GLM-4.5       hidden_size=5120  head_dim=128  use_qk_norm=true
    GLM-4.5-Air   hidden_size=4096
    MiniMax-M2    hidden_size=3072  q_norm width = 48*128 = 6144, k_norm width = 8*128 = 1024
    DeepSeek-V3   hidden_size=7168  kv_lora_rank=512  q_lora_rank=1536
    Kimi-K2       hidden_size=7168  kv_lora_rank=512  q_lora_rank=1536

The distinct widths are 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144,
7168, 8192, 16384 and 32768. Six are not powers of two, so the masked column path
runs on real widths rather than a synthetic one. 16384 and 32768 additionally
separate the whole-row and chunked kernel paths.

Row counts follow the operator's working-set tiers, keeping each case long enough
for the timers to be meaningful while still separating L2-resident from HBM-bound
traffic:

    Tier A   64 MiB read+write   fits in L2 on both supported NPU generations
    Tier B  256 MiB read+write   past L2 everywhere, HBM bound

``rows = MiB * 2**20 / (2 * D * element_size)``, so the row count of a given width
is dtype dependent; each entry below carries its own rows and dtype. At both tiers
the grid stays far above the vector-core count, so the core count never binds.

BF16 and FP16 alternate across the list. Both dtypes therefore reach every tier,
both kernel paths and both power-of-two families, at half the case count of
running the full matrix.
"""

import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


def _case(rows, dim, dtype):
    """One benchmark row, labelled with its dtype."""
    return pytest.param(rows, dim, dtype, id=f"R{rows}-D{dim}-{str(dtype).rsplit('.', 1)[-1]}")


# D <= 512: row-batched kernel with several rows per program.
NARROW_CASES = [
    # Tier A, 64 MiB
    (131072, 128, torch.bfloat16),  # Qwen3-32B / GLM-4.5 head_dim
    (65536, 256, torch.float16),  # Qwen3-Next GA head_dim
    (32768, 512, torch.bfloat16),  # DeepSeek-V3 / Kimi-K2 kv_lora_rank
    # Tier B, 256 MiB
    (524288, 128, torch.float16),
    (262144, 256, torch.bfloat16),
    (131072, 512, torch.float16),
]

# D > 512: whole-row kernel, one row per program.
WIDE_CASES = [
    # Tier A, 64 MiB
    (16384, 1024, torch.bfloat16),  # MiniMax-M2 k_norm (8 kv_heads * 128)
    (10922, 1536, torch.float16),  # DeepSeek/Kimi q_lora_rank
    (8192, 2048, torch.bfloat16),  # Qwen3-Next hidden_size
    (5461, 3072, torch.float16),  # MiniMax-M2 hidden_size
    (4096, 4096, torch.bfloat16),  # GLM-4.5-Air hidden_size
    (3276, 5120, torch.float16),  # GLM-4.5 / Qwen3-32B hidden
    (2730, 6144, torch.bfloat16),  # MiniMax-M2 q_norm (48 heads * 128)
    (2340, 7168, torch.float16),  # DeepSeek-V3 / Kimi-K2 hidden
    # Tier B, 256 MiB
    (65536, 1024, torch.float16),
    (43690, 1536, torch.bfloat16),
    (32768, 2048, torch.float16),
    (21845, 3072, torch.bfloat16),
    (16384, 4096, torch.float16),
    (13107, 5120, torch.bfloat16),
    (10922, 6144, torch.float16),
    (9362, 7168, torch.bfloat16),
    # Tile tail: an odd row count leaves a single live row in the last program.
    (2341, 7168, torch.bfloat16),
]

# Whole-row versus chunked dispatch, at equal read+write traffic per group.
LARGE_D_CASES = [
    # 64 MiB read+write
    (2048, 8192, torch.float16),
    (1024, 16384, torch.bfloat16),
    (512, 32768, torch.float16),  # chunked
    # 128 MiB read+write
    (4096, 8192, torch.bfloat16),
    (2048, 16384, torch.float16),
    (1024, 32768, torch.bfloat16),  # chunked
    # 256 MiB read+write
    (2048, 32768, torch.float16),  # chunked
]

CASES = [_case(rows, dim, dtype) for rows, dim, dtype in NARROW_CASES + WIDE_CASES + LARGE_D_CASES]


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("rows,dim,dtype", CASES)
@pytest.mark.parametrize("phase", ["forward"])
def test_l2_norm(benchmark, perf_environment, rows, dim, dtype, phase):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(rows, dim, device=device, dtype=dtype)
        call = lambda: functions.l2_norm(x, implementation=implementation)
        return training(call, (x,), phase)

    benchmark(
        factory=factory,
        op="l2_norm",
        shape=[rows, dim],
        dtype=str(dtype),
        phase=phase,
    )
