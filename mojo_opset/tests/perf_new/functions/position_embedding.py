"""Mojo Function rotary-position-embedding performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoApplyRoPEFunction
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
    ),
    perf_case(
        "full_b32_s8192_qh32_kh8_d128",
        tags=("full",),
        batch=32,
        seq_len=8192,
        q_heads=32,
        k_heads=8,
        head_dim=128,
    ),
)


def _shapes(case: Mapping[str, Any]):
    batch = int(case["batch"])
    seq_len = int(case["seq_len"])
    q_heads = int(case["q_heads"])
    k_heads = int(case["k_heads"])
    head_dim = int(case["head_dim"])
    return (
        (batch, q_heads, seq_len, head_dim),
        (batch, k_heads, seq_len, head_dim),
        (seq_len, head_dim),
    )


@mojo_perf(
    name="mojo_apply_rope_function_forward",
    target=MojoApplyRoPEFunction,
    cases=CASES,
)
def apply_rope_forward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    q_shape, k_shape, rope_shape = _shapes(case)
    return PerfWorkload(
        inputs={
            "q": tensor(q_shape, torch.bfloat16, creator=torch.randn),
            "k": tensor(k_shape, torch.bfloat16, creator=torch.randn),
            "cos": tensor(rope_shape, torch.float32, creator=torch.randn),
            "sin": tensor(rope_shape, torch.float32, creator=torch.randn),
        },
        outputs={
            "q_rot": tensor(q_shape, torch.bfloat16),
            "k_rot": tensor(k_shape, torch.bfloat16),
        },
        args=("q", "k", "cos", "sin", True),
    )


@mojo_perf(
    name="mojo_apply_rope_function_backward",
    target=MojoApplyRoPEFunction,
    cases=CASES,
    phase="backward",
)
def apply_rope_backward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    q_shape, k_shape, rope_shape = _shapes(case)
    return PerfWorkload(
        inputs={
            "q": tensor(q_shape, torch.bfloat16, creator=torch.randn),
            "k": tensor(k_shape, torch.bfloat16, creator=torch.randn),
            "cos": tensor(rope_shape, torch.float32, creator=torch.randn),
            "sin": tensor(rope_shape, torch.float32, creator=torch.randn),
            "dq_out": tensor(q_shape, torch.bfloat16, creator=torch.randn),
            "dk_out": tensor(k_shape, torch.bfloat16, creator=torch.randn),
        },
        outputs={
            "dq": tensor(q_shape, torch.bfloat16),
            "dk": tensor(k_shape, torch.bfloat16),
        },
        forward_args=("q", "k", "cos", "sin", True),
    )
