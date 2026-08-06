"""Mojo Function activation performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoSiluFunction
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor


CASES = (
    perf_case("smoke_1024x1024", tags=("smoke",), rows=1024, cols=1024),
    perf_case("full_4096x4096", tags=("full",), rows=4096, cols=4096),
)


@mojo_perf(
    name="mojo_silu_function_forward",
    target=MojoSiluFunction,
    cases=CASES,
)
def silu_forward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["cols"]))
    return PerfWorkload(
        inputs={"x": tensor(shape, torch.float16, creator=torch.randn)},
        outputs={"y": tensor(shape, torch.float16)},
        flops=4 * shape[0] * shape[1],
    )


@mojo_perf(
    name="mojo_silu_function_backward",
    target=MojoSiluFunction,
    cases=CASES,
    phase="backward",
)
def silu_backward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["cols"]))
    return PerfWorkload(
        inputs={
            "x": tensor(shape, torch.float16, creator=torch.randn),
            "dy": tensor(shape, torch.float16, creator=torch.randn),
        },
        outputs={"dx": tensor(shape, torch.float16)},
        forward_args=("x",),
        flops=7 * shape[0] * shape[1],
    )
