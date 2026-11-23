import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training

# Production shapes from tests/functions/test_shared_expert_gate.py; all cases
# are kept above the 20us measurement floor.
CASES = [
    ((8192, 2048), torch.bfloat16),
    ((8192, 2048), torch.float16),
    ((16384, 2048), torch.bfloat16),
    ((16384, 2048), torch.float16),
    ((32768, 2048), torch.bfloat16),
    ((32768, 2048), torch.float16),
    ((4096, 3584), torch.bfloat16),
    ((4096, 3584), torch.float16),
    ((8192, 3584), torch.bfloat16),
    ((8192, 3584), torch.float16),
    ((4096, 4096), torch.bfloat16),
    ((4096, 4096), torch.float16),
    ((8192, 4096), torch.bfloat16),
    ((8192, 4096), torch.float16),
    ((4096, 5120), torch.bfloat16),
    ((4096, 5120), torch.float16),
    ((8192, 5120), torch.bfloat16),
    ((8192, 5120), torch.float16),
    ((8192, 7168), torch.bfloat16),
    ((8192, 7168), torch.float16),
]

_DTYPE_NAME = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}


def _case_id(case):
    shape, dtype = case
    return f"{shape[0]}x{shape[1]}_{_DTYPE_NAME[dtype]}"


@pytest.mark.api("functions.shared_expert_gate", ops=["shared_expert_gate"])
@pytest.mark.parametrize("shape,dtype", CASES, ids=[_case_id(case) for case in CASES])
def test_shared_expert_gate(benchmark, perf_environment, shape, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, dtype=dtype, device=device)
        w = torch.randn(1, shape[-1], dtype=dtype, device=device)
        s = torch.randn(shape, dtype=dtype, device=device)
        return training(
            lambda: functions.shared_expert_gate(
                x,
                w,
                s,
                implementation=implementation,
            ),
            (x, w, s),
            "forward",
        )

    benchmark(
        factory=factory,
        op="shared_expert_gate",
        shape=list(shape),
        dtype=str(dtype),
    )
