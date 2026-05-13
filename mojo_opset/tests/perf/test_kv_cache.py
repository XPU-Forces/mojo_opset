import pytest
import torch

from mojo_opset import MojoStorePagedKVCache
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


@pytest.mark.parametrize(
    "batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val",
    [
        (2, 2, 128, 128, [0, 0], [130, 33]),
        (2, 2, 128, 128, [32, 35], [1, 1]),
        (2, 2, 128, 128, [15, 40], [788, 126]),
        (2, 2, 128, 256, [15, 40], [788, 126]),
        (1, 1, 128, 128, [0], [5]),
        (1, 1, 128, 128, [5], [1]),
        (8, 2, 128, 128, [224, 542, 34, 41, 54, 57, 65, 0], [432, 84, 977, 93, 23, 89, 31, 555]),
        (8, 2, 128, 128, [772, 974, 3232, 43, 77, 7633, 888, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    ],
)
@bypass_not_implemented
@auto_switch_platform(set_perf=True)
def test_store_paged_kv(batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val):
    device = get_torch_device()

    context_kv_lens = torch.tensor(context_kv_lens_val, dtype=torch.int32, device=device)
    q_lens = torch.tensor(q_lens_val, dtype=torch.int32, device=device)

    is_decode = torch.all(q_lens == 1).item()
    cu_q_lens = (
        torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(q_lens, dim=0, dtype=torch.int32)])
        if not is_decode
        else None
    )
    total_tokens = int(q_lens.sum().item()) if not is_decode else batch_size

    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device=device)

    max_kv_len = (context_kv_lens + q_lens).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2

    total_blocks_needed = sum(
        (context_kv_len + q_len + block_size - 1) // block_size
        for context_kv_len, q_len in zip(context_kv_lens_val, q_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)

    k_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    v_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    curr = 0
    for i in range(batch_size):
        needed = (context_kv_lens_val[i] + q_lens_val[i] + block_size - 1) // block_size
        ids = torch.arange(curr, curr + needed, device=device, dtype=torch.int32)
        block_table[i, :needed] = ids
        curr += needed

    chunk_metadata = build_paged_kv_chunk_metadata(block_table, cu_q_lens, context_kv_lens, block_size)

    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()

    store_paged_kv = MojoStorePagedKVCache()

    perf(  # noqa: F821
        lambda: store_paged_kv(
            key_states,
            value_states,
            k_cache,
            v_cache,
            chunk_metadata=chunk_metadata,
        )
    )
