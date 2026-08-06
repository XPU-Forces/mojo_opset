"""Mojo Function causal-convolution performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoCausalConv1dFunction
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import literal
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor


CASES = (
    perf_case(
        "smoke_b2_t64_d1024_w3",
        tags=("smoke",),
        batch=2,
        tokens=64,
        hidden=1024,
        width=3,
        has_bias=True,
        has_residual=True,
        activation="swish",
    ),
    perf_case(
        "full_b2_t128_d4096_w4",
        tags=("full",),
        batch=2,
        tokens=128,
        hidden=4096,
        width=4,
        has_bias=False,
        has_residual=False,
        activation="swish",
    ),
)


def _workload(case: Mapping[str, Any], *, backward: bool) -> PerfWorkload:
    batch = int(case["batch"])
    tokens = int(case["tokens"])
    hidden = int(case["hidden"])
    width = int(case["width"])
    has_bias = bool(case["has_bias"])
    has_residual = bool(case["has_residual"])
    shape = (batch, tokens, hidden)
    weight_shape = (hidden, width)

    inputs = {
        "x": tensor(shape, torch.float16, creator=torch.rand),
        "weight": tensor(weight_shape, torch.float16, creator=torch.rand),
    }
    if has_bias:
        inputs["bias"] = tensor((hidden,), torch.float16, creator=torch.rand)
    if has_residual:
        inputs["residual"] = tensor(shape, torch.float16, creator=torch.rand)

    forward_args = (
        "x",
        "weight",
        "bias" if has_bias else None,
        "residual" if has_residual else None,
        None,
        False,
        literal(case["activation"]),
    )
    flops = 2 * batch * tokens * hidden * width
    if not backward:
        return PerfWorkload(
            inputs=inputs,
            outputs={"y": tensor(shape, torch.float16)},
            args=forward_args,
            flops=flops,
        )

    inputs["dy"] = tensor(shape, torch.float16, creator=torch.rand)
    outputs = {
        "dx": tensor(shape, torch.float16),
        "dweight": tensor(weight_shape, torch.float16),
    }
    if has_bias:
        outputs["dbias"] = tensor((hidden,), torch.float16)
    if has_residual:
        outputs["dresidual"] = tensor(shape, torch.float16)
    return PerfWorkload(
        inputs=inputs,
        outputs=outputs,
        forward_args=forward_args,
        flops=2 * flops,
    )


@mojo_perf(
    name="mojo_causal_conv1d_function_forward",
    target=MojoCausalConv1dFunction,
    cases=CASES,
)
def causal_conv1d_forward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _workload(case, backward=False)


@mojo_perf(
    name="mojo_causal_conv1d_function_backward",
    target=MojoCausalConv1dFunction,
    cases=CASES,
    phase="backward",
)
def causal_conv1d_backward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _workload(case, backward=True)
