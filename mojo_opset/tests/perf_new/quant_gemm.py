"""xpu-perf description for :class:`MojoQuantGemm`."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoQuantGemm
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor


CASES = (
    perf_case(
        "smoke_m16_k1024_n1024",
        tags=("smoke",),
        m=16,
        k=1024,
        n=1024,
        output_dtype=torch.bfloat16,
        trans_weight=False,
    ),
    perf_case(
        "full_m4096_k4096_n4096",
        tags=("full",),
        m=4096,
        k=4096,
        n=4096,
        output_dtype=torch.bfloat16,
        trans_weight=False,
    ),
    perf_case(
        "full_m8192_k8192_n8192",
        tags=("full",),
        m=8192,
        k=8192,
        n=8192,
        output_dtype=torch.bfloat16,
        trans_weight=False,
    ),
    *(
        perf_case(
            f"m{m}_k{k}_n{n}",
            tags=("full",),
            m=m,
            k=k,
            n=n,
            output_dtype=torch.bfloat16,
            trans_weight=False,
        )
        for m, k, n in (
            (1, 4096, 4096),
            (32, 4096, 4096),
            (128, 4096, 4096),
            (256, 4096, 4096),
            (512, 4096, 4096),
            (1024, 4096, 4096),
            (2048, 4096, 4096),
            (128, 4096, 11008),
            (1024, 8192, 4096),
            (4096, 8192, 4096),
        )
    ),
)


@mojo_perf(
    name="mojo_quant_gemm",
    target=MojoQuantGemm,
    cases=CASES,
)
def quant_gemm_workload(case: Mapping[str, Any]) -> PerfWorkload:
    """Return construction metadata; xpu-perf creates the actual tensors."""

    m = int(case["m"])
    k = int(case["k"])
    n = int(case["n"])
    sp_size = int(case.get("sp_size", 1))
    m = ((m + sp_size - 1) // sp_size) * sp_size

    output_dtype = case.get("output_dtype", torch.bfloat16)
    quant_dtype = case.get("quant_dtype", torch.int8)
    weight_dtype = case.get("weight_dtype", torch.int8)
    trans_weight = bool(case.get("trans_weight", False))
    if quant_dtype != torch.int8 or weight_dtype != torch.int8:
        raise ValueError("MojoQuantGemm performance cases currently require int8 input and weight")

    weight_shape = (n, k) if trans_weight else (k, n)
    return PerfWorkload(
        op_kwargs={
            "in_features": k,
            "out_features": n,
            "output_dtype": output_dtype,
            "trans_weight": trans_weight,
            "quant_dtype": quant_dtype,
            "weight_dtype": weight_dtype,
        },
        inputs={
            "input": tensor((m, k), torch.int8, creator=torch.zeros),
            "input_scale": tensor((m,), torch.float32, creator=torch.ones),
            "weight": tensor(weight_shape, torch.int8, creator=torch.zeros),
            "weight_scale": tensor((n,), torch.bfloat16, creator=torch.ones),
        },
        outputs={
            "y": tensor((m, n), output_dtype),
        },
        state={
            "weight": "weight",
            "weight_scale": "weight_scale",
        },
        flops=2 * m * k * n,
    )
