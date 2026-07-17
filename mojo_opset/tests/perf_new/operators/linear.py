"""Grouped-linear performance cases."""

from functools import partial
from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoGroupGemm
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor
from mojo_opset.experimental import MojoQuantBatchGemmReduceSum



GROUP_GEMM_CASES = (
    perf_case(
        "smoke_g2_t32_k128_n128_f16",
        tags=("smoke",),
        groups=2,
        tokens_per_group=32,
        k=128,
        n=128,
        dtype=torch.float16,
    ),
    *(
        perf_case(
            f"g8_t2560_k4096_n4096_{str(value_dtype).removeprefix('torch.')}",
            tags=("full",),
            groups=8,
            tokens_per_group=2560,
            k=4096,
            n=4096,
            dtype=value_dtype,
        )
        for value_dtype in (torch.float16, torch.bfloat16)
    ),
)


@mojo_perf(name="mojo_group_gemm", target=MojoGroupGemm, cases=GROUP_GEMM_CASES)
def group_gemm_workload(case: Mapping[str, Any]) -> PerfWorkload:
    groups = int(case["groups"])
    tokens_per_group = int(case["tokens_per_group"])
    k = int(case["k"])
    n = int(case["n"])
    total_tokens = groups * tokens_per_group
    value_dtype = case["dtype"]
    weight_shape = (groups, k, n)

    def tensor_factory(device: str):
        return {
            "input": torch.randn((total_tokens, k), dtype=value_dtype, device=device),
            "weight": torch.randn(weight_shape, dtype=value_dtype, device=device),
            "group_list": torch.full(
                (groups,),
                tokens_per_group,
                dtype=torch.int32,
                device=device,
            ).cumsum(0, dtype=torch.int32),
        }

    def target_factory(target_cls: type, device: str):
        placeholder = torch.empty(weight_shape, dtype=value_dtype, device=device)
        return target_cls(weight=placeholder, trans_weight=False)

    return PerfWorkload(
        inputs={
            "input": tensor((total_tokens, k), value_dtype),
            "weight": tensor(weight_shape, value_dtype),
            "group_list": tensor((groups,), torch.int32),
        },
        outputs={"output": tensor((total_tokens, n), value_dtype)},
        state={"weight": "weight"},
        tensor_factory=tensor_factory,
        target_factory=target_factory,
        flops=2 * total_tokens * k * n,
    )


QUANT_REDUCE_CASES = tuple(
    perf_case(
        f"b{batch}_m{m}_k{k}_n{n}",
        tags=(("smoke", "full") if batch == 8 else ("full",)),
        batch=batch,
        m=m,
        k=k,
        n=n,
    )
    for batch, m, k, n in ((8, 512, 128, 256), (4, 1024, 128, 512))
)


@mojo_perf(
    name="mojo_quant_batch_gemm_reduce_sum",
    target=MojoQuantBatchGemmReduceSum,
    cases=QUANT_REDUCE_CASES,
)
def quant_batch_gemm_reduce_sum_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    m = int(case["m"])
    k = int(case["k"])
    n = int(case["n"])
    weight_shape = (batch, k, n)

    def tensor_factory(device: str):
        weight = torch.randint(
            -128,
            128,
            weight_shape,
            dtype=torch.int8,
            device=device,
        )
        if str(device).split(":", 1)[0] == "npu":
            # The NPU kernel requires a private FRACTAL_NZ weight. Prepare the
            # static weight before xpu-perf starts probe/warmup/timing loops.
            import torch_npu

            torch.npu.config.allow_internal_format = True
            weight = torch_npu.npu_format_cast(weight.contiguous(), 29)
            if torch_npu.get_npu_format(weight) != 29:
                raise RuntimeError(
                    "QuantBatchGemmReduceSum benchmark requires FRACTAL_NZ weight, "
                    "but the ND-to-NZ format cast failed"
                )
        return {
            "x1": torch.randint(-128, 128, (batch, m, k), dtype=torch.int8, device=device),
            "weight": weight,
            "x1_scale": torch.rand((batch, m), dtype=torch.float32, device=device),
            "x2_scale": torch.rand((n,), dtype=torch.bfloat16, device=device),
        }

    def target_factory(target_cls: type, device: str):
        placeholder = torch.empty(weight_shape, dtype=torch.int8, device=device)
        return target_cls(weight=placeholder, trans_weight=False)

    return PerfWorkload(
        inputs={
            "x1": tensor(
                (batch, m, k),
                torch.int8,
                creator=partial(torch.randint, low=-128, high=128),
            ),
            "weight": tensor(
                weight_shape,
                torch.int8,
                creator=partial(torch.randint, low=-128, high=128),
            ),
            "x1_scale": tensor((batch, m), torch.float32, creator=torch.rand),
            "x2_scale": tensor((n,), torch.bfloat16, creator=torch.rand),
        },
        outputs={"output": tensor((m, n), torch.bfloat16)},
        state={"weight": "weight"},
        tensor_factory=tensor_factory,
        target_factory=target_factory,
        flops=2 * batch * m * k * n,
    )
