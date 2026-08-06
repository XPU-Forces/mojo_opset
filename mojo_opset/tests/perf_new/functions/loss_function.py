"""Mojo Function fused-linear-cross-entropy performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoFusedLinearCrossEntropyFunction
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor


CASES = (
    perf_case(
        "smoke_b128_h256_v1024",
        tags=("smoke",),
        batch=128,
        hidden=256,
        vocab=1024,
    ),
    perf_case(
        "full_b2048_h1024_v4096",
        tags=("full",),
        batch=2048,
        hidden=1024,
        vocab=4096,
    ),
)


def _target_creator(vocab: int):
    def creator(*, size, dtype, device):
        return torch.randint(0, vocab, size, dtype=dtype, device=device)

    return creator


def _inputs(case: Mapping[str, Any]):
    batch = int(case["batch"])
    hidden = int(case["hidden"])
    vocab = int(case["vocab"])
    return {
        "input": tensor((batch, hidden), torch.bfloat16, creator=torch.randn),
        "weight": tensor((vocab, hidden), torch.bfloat16, creator=torch.randn),
        "target": tensor(
            (batch,),
            torch.int64,
            creator=_target_creator(vocab),
        ),
    }


@mojo_perf(
    name="mojo_fused_linear_cross_entropy_function_forward",
    target=MojoFusedLinearCrossEntropyFunction,
    cases=CASES,
)
def fused_ce_forward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    hidden = int(case["hidden"])
    vocab = int(case["vocab"])
    return PerfWorkload(
        inputs=_inputs(case),
        outputs={"loss": tensor((), torch.float32)},
        flops=2 * batch * hidden * vocab,
    )


@mojo_perf(
    name="mojo_fused_linear_cross_entropy_function_backward",
    target=MojoFusedLinearCrossEntropyFunction,
    cases=CASES,
    phase="backward",
)
def fused_ce_backward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    hidden = int(case["hidden"])
    vocab = int(case["vocab"])
    inputs = _inputs(case)
    inputs["grad_loss"] = tensor((), torch.float32, creator=torch.rand)
    return PerfWorkload(
        inputs=inputs,
        outputs={
            "grad_input": tensor((batch, hidden), torch.bfloat16),
            "grad_weight": tensor((vocab, hidden), torch.bfloat16),
        },
        forward_args=("input", "weight", "target"),
    )
