"""Low-rank-cache store performance cases."""

from functools import partial
from typing import Any
from typing import Mapping

import torch

from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor
from mojo_opset.experimental import MojoStoreLowrank



CASES = tuple(
    perf_case(
        f"blocks256_h{heads}_d128_kv{kv_len}",
        tags=(("smoke", "full") if heads == 1 and kv_len == 1024 else ("full",)),
        blocks=256,
        heads=heads,
        block_size=512,
        head_dim=128,
        kv_len=kv_len,
    )
    for heads in (1, 8)
    for kv_len in (1024, 2048, 4096, 8192, 13312)
)


@mojo_perf(name="mojo_store_lowrank", target=MojoStoreLowrank, cases=CASES)
def store_lowrank_workload(case: Mapping[str, Any]) -> PerfWorkload:
    blocks = int(case["blocks"])
    heads = int(case["heads"])
    block_size = int(case["block_size"])
    head_dim = int(case["head_dim"])
    kv_len = int(case["kv_len"])
    cache_shape = (blocks, heads, block_size, head_dim)
    key_shape = (kv_len, heads, head_dim)
    return PerfWorkload(
        inputs={
            "label_cache": tensor(cache_shape, torch.bfloat16, creator=torch.zeros),
            "key_lr": tensor(key_shape, torch.bfloat16, creator=torch.randn),
            "block_idxs": tensor(
                (kv_len,),
                torch.int32,
                creator=partial(torch.randint, low=0, high=blocks),
            ),
            "token_idxs": tensor(
                (kv_len,),
                torch.int32,
                creator=partial(torch.randint, low=0, high=block_size),
            ),
        },
        outputs={"label_cache_out": tensor(cache_shape, torch.bfloat16)},
        args=("label_cache", "key_lr", "block_idxs", "token_idxs", kv_len),
        read_bytes=kv_len * heads * head_dim * 2 + kv_len * 8,
        write_bytes=kv_len * heads * head_dim * 2,
    )
