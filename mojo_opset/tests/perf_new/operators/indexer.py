"""Lightning-indexer performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor
from mojo_opset.experimental import MojoLightningIndexer



_SHAPES = ((128, 256, 256, 64, 128), (24, 1024, 1024, 128, 128), (24, 1, 16384, 128, 128))

CASES = tuple(
    perf_case(
        f"b{batch}_m{m}_n{n}_h{heads}_k{k}_{str(value_dtype).removeprefix('torch.')}",
        tags=(
            ("smoke", "full")
            if shape_index == 0 and value_dtype is torch.bfloat16
            else ("full",)
        ),
        batch=batch,
        m=m,
        n=n,
        heads=heads,
        k=k,
        dtype=value_dtype,
    )
    for shape_index, (batch, m, n, heads, k) in enumerate(_SHAPES)
    for value_dtype in (torch.bfloat16, torch.float16, torch.float32)
)


@mojo_perf(name="mojo_lightning_indexer", target=MojoLightningIndexer, cases=CASES)
def lightning_indexer_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    m = int(case["m"])
    n = int(case["n"])
    heads = int(case["heads"])
    k = int(case["k"])
    value_dtype = case["dtype"]
    return PerfWorkload(
        inputs={
            "query": tensor((batch, m, heads, k), value_dtype, creator=torch.randn),
            "query_scale": tensor((batch, m, heads), torch.float32, creator=torch.randn),
            "key": tensor((batch, n, k), value_dtype, creator=torch.randn),
            "key_scale": tensor((batch, n), torch.float32, creator=torch.randn),
        },
        outputs={},
        flops=2 * batch * m * n * heads * k,
    )
