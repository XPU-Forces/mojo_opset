"""Paged-KV-cache store performance cases."""

from typing import Any
from typing import Mapping

import torch

from mojo_opset import MojoStorePagedKVCache
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata



_CONFIGS = (
    (2, 2, 128, 128, [0, 0], [130, 33]),
    (2, 2, 128, 128, [32, 35], [1, 1]),
    (2, 2, 128, 128, [15, 40], [788, 126]),
    (2, 2, 128, 256, [15, 40], [788, 126]),
    (1, 1, 128, 128, [0], [5]),
    (1, 1, 128, 128, [5], [1]),
    (8, 2, 128, 128, [224, 542, 34, 41, 54, 57, 65, 0], [432, 84, 977, 93, 23, 89, 31, 555]),
    (8, 2, 128, 128, [772, 974, 3232, 43, 77, 7633, 888, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
)

CASES = tuple(
    perf_case(
        f"b{batch}_h{kv_heads}_d{head_dim}_blk{block_size}_{index}",
        tags=(("smoke", "full") if index == 4 else ("full",)),
        batch=batch,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        context_kv_lens=context_lens,
        q_lens=q_lens,
    )
    for index, (batch, kv_heads, head_dim, block_size, context_lens, q_lens) in enumerate(_CONFIGS)
)


def _metadata_shape(
    batch: int,
    block_size: int,
    context_lens: list[int],
    q_lens_values: list[int],
) -> tuple[int, ...]:
    context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32)
    q_lens = torch.tensor(q_lens_values, dtype=torch.int32)
    cu_q_lens = None
    if not all(value == 1 for value in q_lens_values):
        cu_q_lens = torch.cat(
            (torch.zeros(1, dtype=torch.int32), q_lens.cumsum(0, dtype=torch.int32))
        )
    max_kv_len = max(context + query for context, query in zip(context_lens, q_lens_values))
    max_blocks = (max_kv_len + block_size - 1) // block_size + 2
    block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32)
    current = 0
    for index, (context, query) in enumerate(zip(context_lens, q_lens_values)):
        needed = (context + query + block_size - 1) // block_size
        block_table[index, :needed] = torch.arange(current, current + needed, dtype=torch.int32)
        current += needed
    metadata = build_paged_kv_chunk_metadata(
        block_table,
        cu_q_lens,
        context_lens_tensor,
        block_size,
    )
    return tuple(metadata.shape)


@mojo_perf(name="mojo_store_paged_kv_cache", target=MojoStorePagedKVCache, cases=CASES)
def store_paged_kv_workload(case: Mapping[str, Any]) -> PerfWorkload:
    batch = int(case["batch"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    block_size = int(case["block_size"])
    context_values = [int(value) for value in case["context_kv_lens"]]
    q_values = [int(value) for value in case["q_lens"]]
    is_decode = all(value == 1 for value in q_values)
    total_tokens = batch if is_decode else sum(q_values)
    max_kv_len = max(context + query for context, query in zip(context_values, q_values))
    max_blocks = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks = sum(
        (context + query + block_size - 1) // block_size
        for context, query in zip(context_values, q_values)
    ) + 10
    cache_shape = (total_blocks, kv_heads, block_size, head_dim)
    state_shape = (total_tokens, kv_heads, head_dim)
    metadata_shape = _metadata_shape(batch, block_size, context_values, q_values)

    def tensor_factory(device: str):
        context_lens = torch.tensor(context_values, dtype=torch.int32, device=device)
        q_lens = torch.tensor(q_values, dtype=torch.int32, device=device)
        cu_q_lens = None
        if not is_decode:
            cu_q_lens = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device=device),
                    q_lens.cumsum(0, dtype=torch.int32),
                )
            )
        block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=device)
        current = 0
        for index, (context, query) in enumerate(zip(context_values, q_values)):
            needed = (context + query + block_size - 1) // block_size
            block_table[index, :needed] = torch.arange(
                current,
                current + needed,
                dtype=torch.int32,
                device=device,
            )
            current += needed
        metadata = build_paged_kv_chunk_metadata(
            block_table,
            cu_q_lens,
            context_lens,
            block_size,
        )
        return {
            "key_states": torch.randn(state_shape, dtype=torch.bfloat16, device=device),
            "value_states": torch.randn(state_shape, dtype=torch.bfloat16, device=device),
            "key_cache": torch.zeros(cache_shape, dtype=torch.bfloat16, device=device),
            "value_cache": torch.zeros(cache_shape, dtype=torch.bfloat16, device=device),
            "chunk_metadata": metadata,
        }

    return PerfWorkload(
        inputs={
            "key_states": tensor(state_shape, torch.bfloat16),
            "value_states": tensor(state_shape, torch.bfloat16),
            "key_cache": tensor(cache_shape, torch.bfloat16),
            "value_cache": tensor(cache_shape, torch.bfloat16),
            "chunk_metadata": tensor(metadata_shape, torch.int32),
        },
        outputs={
            "key_cache_out": tensor(cache_shape, torch.bfloat16),
            "value_cache_out": tensor(cache_shape, torch.bfloat16),
        },
        kwargs={"chunk_metadata": "chunk_metadata"},
        tensor_factory=tensor_factory,
        read_bytes=2 * total_tokens * kv_heads * head_dim * 2 + metadata_shape[0] * metadata_shape[1] * 4,
        write_bytes=2 * total_tokens * kv_heads * head_dim * 2,
    )
