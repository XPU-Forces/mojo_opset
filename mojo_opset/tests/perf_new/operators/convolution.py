"""Causal-convolution update-state performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoCausalConv1dUpdateState
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import literal
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor



_CONFIGS = (
    (1, 12291, 8192, 4, "swish"),
    (1, 5000, 2048, 4, "swish"),
    (2, 64, 128, 3, "swish"),
    (2, 128, 128, 4, "swish"),
    (2, 64, 128, 3, None),
    (3, 1446, 256, 4, None),
    (1, 32, 32, 4, None),
)

CASES = tuple(
    perf_case(
        f"b{batch}_t{tokens}_d{channels}_w{width}_{activation or 'none'}",
        tags=(("smoke", "full") if (batch, tokens, channels) == (1, 32, 32) else ("full",)),
        batch=batch,
        tokens=tokens,
        channels=channels,
        width=width,
        activation=activation,
    )
    for batch, tokens, channels, width, activation in _CONFIGS
)


@mojo_perf(
    name="mojo_causal_conv1d_update_state",
    target=MojoCausalConv1dUpdateState,
    cases=CASES,
)
def causal_conv_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    tokens = int(case["tokens"])
    channels = int(case["channels"])
    width = int(case["width"])
    activation = case.get("activation")
    hidden_shape = (batch, channels, tokens)
    return PerfWorkload(
        inputs={
            "hidden_states": tensor(hidden_shape, torch.float16, creator=torch.randn),
            "conv_state": tensor((batch, channels, width), torch.float16, creator=torch.randn),
            "weight": tensor((channels, width), torch.float16, creator=torch.randn),
        },
        outputs={"output": tensor(hidden_shape, torch.float16)},
        args=(
            "hidden_states",
            "conv_state",
            "weight",
            None,
            literal(activation) if activation is not None else None,
        ),
        flops=2 * batch * channels * tokens * width,
    )
