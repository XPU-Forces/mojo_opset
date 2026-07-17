"""Normalization performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoLayerNorm
from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset import MojoResidualAddRMSNorm
from mojo_opset import MojoRMSNorm
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor



_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_NORM_POSITIONS = ("pre", "post")

RESIDUAL_CASES = tuple(
    perf_case(
        f"128x128_{str(value_dtype).removeprefix('torch.')}_{norm_pos}",
        tags=(("smoke", "full") if value_dtype is torch.float16 and norm_pos == "pre" else ("full",)),
        rows=128,
        hidden=128,
        dtype=value_dtype,
        norm_pos=norm_pos,
        eps=1e-5,
    )
    for value_dtype in _DTYPES
    for norm_pos in _NORM_POSITIONS
)


def _residual_workload(case: Mapping[str, Any], *, with_bias: bool) -> PerfWorkload:
    rows = int(case["rows"])
    hidden = int(case["hidden"])
    value_dtype = case["dtype"]
    inputs = {
        "x": tensor((rows, hidden), value_dtype, creator=torch.randn),
        "residual": tensor((rows, hidden), value_dtype, creator=torch.randn),
        "weight": tensor((hidden,), value_dtype, creator=torch.randn),
    }
    state = {"weight": "weight"}
    if with_bias:
        inputs["bias"] = tensor((hidden,), value_dtype, creator=torch.randn)
        state["bias"] = "bias"
    return PerfWorkload(
        op_kwargs={
            "norm_size": hidden,
            "eps": float(case["eps"]),
            "norm_pos": case["norm_pos"],
        },
        inputs=inputs,
        outputs={"output": tensor((rows, hidden), value_dtype)},
        state=state,
        flops=(6 if with_bias else 5) * rows * hidden,
    )


@mojo_perf(
    name="mojo_residual_add_rmsnorm",
    target=MojoResidualAddRMSNorm,
    cases=RESIDUAL_CASES,
)
def residual_add_rmsnorm_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _residual_workload(case, with_bias=False)


@mojo_perf(
    name="mojo_residual_add_layernorm",
    target=MojoResidualAddLayerNorm,
    cases=RESIDUAL_CASES,
)
def residual_add_layernorm_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _residual_workload(case, with_bias=True)


RMSNORM_CASES = tuple(
    perf_case(
        f"1x32x2048_{str(value_dtype).removeprefix('torch.')}",
        tags=(("smoke", "full") if value_dtype is torch.float16 else ("full",)),
        batch=1,
        tokens=32,
        hidden=2048,
        dtype=value_dtype,
        eps=1e-5,
    )
    for value_dtype in _DTYPES
)


@mojo_perf(name="mojo_rmsnorm", target=MojoRMSNorm, cases=RMSNORM_CASES)
def rmsnorm_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["batch"]), int(case["tokens"]), int(case["hidden"]))
    hidden = shape[-1]
    value_dtype = case["dtype"]
    return PerfWorkload(
        op_kwargs={"norm_size": hidden, "eps": float(case["eps"])},
        inputs={
            "x": tensor(shape, value_dtype, creator=torch.randn),
            "weight": tensor((hidden,), torch.float32, creator=torch.randn),
        },
        outputs={"output": tensor(shape, value_dtype)},
        state={"weight": "weight"},
        flops=5 * shape[0] * shape[1] * shape[2],
    )


LAYERNORM_CASES = tuple(
    perf_case(
        f"256x128_{str(value_dtype).removeprefix('torch.')}",
        tags=(("smoke", "full") if value_dtype is torch.float16 else ("full",)),
        rows=256,
        hidden=128,
        dtype=value_dtype,
        eps=1e-5,
    )
    for value_dtype in _DTYPES
)


@mojo_perf(name="mojo_layernorm", target=MojoLayerNorm, cases=LAYERNORM_CASES)
def layernorm_workload(case: Mapping[str, Any]) -> PerfWorkload:
    shape = (int(case["rows"]), int(case["hidden"]))
    hidden = shape[-1]
    value_dtype = case["dtype"]
    return PerfWorkload(
        op_kwargs={"norm_size": hidden, "eps": float(case["eps"])},
        inputs={
            "x": tensor(shape, value_dtype, creator=torch.randn),
            "weight": tensor((hidden,), torch.float32, creator=torch.randn),
            "bias": tensor((hidden,), torch.float32, creator=torch.randn),
        },
        outputs={"output": tensor(shape, value_dtype)},
        state={"weight": "weight", "bias": "bias"},
        flops=6 * shape[0] * shape[1],
    )
