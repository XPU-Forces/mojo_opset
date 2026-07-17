"""Activation performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSiluFunction
from mojo_opset import MojoSwiGLU
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor



GELU_CASES = (
    perf_case("128x128_f32", tags=("smoke", "full"), rows=128, cols=128, dtype=torch.float32),
    perf_case("1024x10240_f32", tags=("full",), rows=1024, cols=10240, dtype=torch.float32),
)


@mojo_perf(name="mojo_gelu", target=MojoGelu, cases=GELU_CASES)
def gelu_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["cols"]))
    value_dtype = case["dtype"]
    return PerfWorkload(
        inputs={"x": tensor(shape, value_dtype, creator=torch.rand)},
        outputs={"output": tensor(shape, value_dtype)},
        flops=8 * shape[0] * shape[1],
    )


SILU_CASES = (
    perf_case("128x128_f32", tags=("smoke", "full"), rows=128, cols=128, dtype=torch.float32),
)


@mojo_perf(name="mojo_silu", target=MojoSilu, cases=SILU_CASES)
def silu_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["cols"]))
    value_dtype = case["dtype"]
    return PerfWorkload(
        inputs={"x": tensor(shape, value_dtype, creator=torch.rand)},
        outputs={"output": tensor(shape, value_dtype)},
        flops=4 * shape[0] * shape[1],
    )


SILU_FUNCTION_CASES = (
    perf_case("smoke_1024x1024", tags=("smoke",), rows=1024, cols=1024),
    perf_case("full_4096x4096", tags=("full",), rows=4096, cols=4096),
)


@mojo_perf(
    name="mojo_silu_function",
    target=MojoSiluFunction,
    cases=SILU_FUNCTION_CASES,
)
def silu_function_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["cols"]))
    return PerfWorkload(
        inputs={"input": tensor(shape, torch.float16, creator=torch.randn)},
        outputs={"output": tensor(shape, torch.float16)},
        flops=4 * shape[0] * shape[1],
    )


SWIGLU_CASES = (
    perf_case("256x128_f32", tags=("smoke", "full"), rows=256, cols=128, dtype=torch.float32),
)


@mojo_perf(name="mojo_swiglu", target=MojoSwiGLU, cases=SWIGLU_CASES)
def swiglu_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["cols"]))
    value_dtype = case["dtype"]
    return PerfWorkload(
        inputs={
            "gate_out": tensor(shape, value_dtype, creator=torch.rand),
            "up_out": tensor(shape, value_dtype, creator=torch.rand),
        },
        outputs={"output": tensor(shape, value_dtype)},
        flops=5 * shape[0] * shape[1],
    )
