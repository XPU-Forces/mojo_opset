"""Mojo Function normalization performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoRMSNormFunction
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor


CASES = (
    perf_case("smoke_32x1024", tags=("smoke",), rows=32, hidden=1024),
    perf_case("full_64x8192", tags=("full",), rows=64, hidden=8192),
)


@mojo_perf(
    name="mojo_rmsnorm_function_forward",
    target=MojoRMSNormFunction,
    cases=CASES,
)
def rmsnorm_forward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["hidden"]))
    hidden = shape[-1]
    return PerfWorkload(
        inputs={
            "x": tensor(shape, torch.bfloat16, creator=torch.randn),
            "weight": tensor((hidden,), torch.float32, creator=torch.randn),
        },
        outputs={"y": tensor(shape, torch.bfloat16)},
        args=("x", "weight", 1e-6),
        flops=5 * shape[0] * hidden,
    )


@mojo_perf(
    name="mojo_rmsnorm_function_backward",
    target=MojoRMSNormFunction,
    cases=CASES,
    phase="backward",
)
def rmsnorm_backward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["hidden"]))
    hidden = shape[-1]
    return PerfWorkload(
        inputs={
            "x": tensor(shape, torch.bfloat16, creator=torch.randn),
            "weight": tensor((hidden,), torch.float32, creator=torch.randn),
            "dy": tensor(shape, torch.bfloat16, creator=torch.randn),
        },
        outputs={
            "dx": tensor(shape, torch.bfloat16),
            "dweight": tensor((hidden,), torch.float32),
        },
        forward_args=("x", "weight", 1e-6),
        flops=10 * shape[0] * hidden,
    )
