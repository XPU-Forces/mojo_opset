"""Mojo Function sliding-window-attention performance cases."""

import math

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoSWAFunction
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor


CASES = (
    perf_case(
        "smoke_b1_qh8_kvh2_d128_s128",
        tags=("smoke",),
        batch=1,
        q_heads=8,
        kv_heads=2,
        head_dim=128,
        seq_len=128,
        local_window=31,
        global_window=4,
        gqa_interleave=True,
    ),
    perf_case(
        "full_b2_qh16_kvh4_d128_s4096",
        tags=("full",),
        batch=2,
        q_heads=16,
        kv_heads=4,
        head_dim=128,
        seq_len=4096,
        local_window=1023,
        global_window=4,
        gqa_interleave=False,
    ),
)


def _uniform_cu_lens(batch: int, seq_len: int):
    def creator(*, size, dtype, device):
        del size
        return torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            dtype=dtype,
            device=device,
        )

    return creator


def _workload(case: Mapping[str, Any], *, backward: bool) -> PerfWorkload:
    batch = int(case["batch"])
    q_heads = int(case["q_heads"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    seq_len = int(case["seq_len"])
    total_tokens = batch * seq_len
    q_shape = (total_tokens, q_heads, head_dim)
    kv_shape = (total_tokens, kv_heads, head_dim)
    cu_shape = (batch + 1,)
    inputs = {
        "q": tensor(q_shape, torch.bfloat16, creator=torch.randn),
        "k": tensor(kv_shape, torch.bfloat16, creator=torch.randn),
        "v": tensor(kv_shape, torch.bfloat16, creator=torch.randn),
        "cu_q_lens": tensor(
            cu_shape,
            torch.int32,
            creator=_uniform_cu_lens(batch, seq_len),
        ),
        "cu_total_seq_lens": tensor(
            cu_shape,
            torch.int32,
            creator=_uniform_cu_lens(batch, seq_len),
        ),
    }
    forward_args = (
        "q",
        "k",
        "v",
        "cu_q_lens",
        "cu_total_seq_lens",
        True,
        int(case["local_window"]),
        int(case["global_window"]),
        1.0 / math.sqrt(head_dim),
        bool(case["gqa_interleave"]),
        False,
    )
    if not backward:
        return PerfWorkload(
            inputs=inputs,
            outputs={"o": tensor(q_shape, torch.bfloat16)},
            args=forward_args,
        )

    inputs["do"] = tensor(q_shape, torch.bfloat16, creator=torch.randn)
    return PerfWorkload(
        inputs=inputs,
        outputs={
            "dq": tensor(q_shape, torch.bfloat16),
            "dk": tensor(kv_shape, torch.bfloat16),
            "dv": tensor(kv_shape, torch.bfloat16),
        },
        forward_args=forward_args,
    )


@mojo_perf(
    name="mojo_swa_function_forward",
    target=MojoSWAFunction,
    cases=CASES,
)
def swa_forward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _workload(case, backward=False)


@mojo_perf(
    name="mojo_swa_function_backward",
    target=MojoSWAFunction,
    cases=CASES,
    phase="backward",
)
def swa_backward_workload(case: Mapping[str, Any]) -> PerfWorkload:
    return _workload(case, backward=True)
