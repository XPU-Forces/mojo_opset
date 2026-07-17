"""Rotary-position-embedding performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoApplyRoPE
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor



CASES = (
    perf_case(
        "smoke_b1_s128_qh8_kh2_d128",
        tags=("smoke",),
        batch=1,
        seq_len=128,
        q_heads=8,
        k_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
    ),
    perf_case(
        "b32_s8192_qh32_kh8_d128",
        tags=("full",),
        batch=32,
        seq_len=8192,
        q_heads=32,
        k_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
    ),
)


@mojo_perf(name="mojo_apply_rope", target=MojoApplyRoPE, cases=CASES)
def apply_rope_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    seq_len = int(case["seq_len"])
    q_heads = int(case["q_heads"])
    k_heads = int(case["k_heads"])
    head_dim = int(case["head_dim"])
    value_dtype = case["dtype"]
    q_shape = (batch, q_heads, seq_len, head_dim)
    k_shape = (batch, k_heads, seq_len, head_dim)
    rope_shape = (seq_len, head_dim)
    return PerfWorkload(
        inputs={
            "q": tensor(q_shape, value_dtype, creator=torch.randn),
            "k": tensor(k_shape, value_dtype, creator=torch.randn),
            "cos": tensor(rope_shape, torch.float32, creator=torch.randn),
            "sin": tensor(rope_shape, torch.float32, creator=torch.randn),
        },
        outputs={
            "q_rot": tensor(q_shape, value_dtype),
            "k_rot": tensor(k_shape, value_dtype),
        },
        args=("q", "k", "cos", "sin", True),
        flops=6 * (batch * q_heads * seq_len * head_dim + batch * k_heads * seq_len * head_dim),
    )
