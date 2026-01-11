import pytest
import torch

from tests.utils import assert_close
from tests.utils import bypass_not_implemented

from mojo_opset import MojoStorePagedKVCache
from mojo_opset.utils.platform import get_platform


def generate_inputs(batch_size, num_heads, head_dim, new_seq_len, context_len_val, block_size, device):
    total_len = context_len_val + new_seq_len

    num_logical_blocks = (total_len + block_size - 1) // block_size

    total_pool_blocks = 1000

    block_tables = torch.zeros((batch_size, num_logical_blocks + 10), dtype=torch.long, device=device)

    all_indices = torch.randperm(total_pool_blocks, device=device)
    cursor = 0
    for i in range(batch_size):
        needed_ids = all_indices[cursor : cursor + num_logical_blocks]
        block_tables[i, :num_logical_blocks] = needed_ids
        cursor += num_logical_blocks

    k_cache = torch.zeros(
        (total_pool_blocks, num_heads, block_size, head_dim),
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.zeros(
        (total_pool_blocks, num_heads, block_size, head_dim),
        dtype=torch.float16,
        device=device,
    )

    key_states = torch.randn(
        (batch_size, num_heads, new_seq_len, head_dim),
        dtype=torch.float16,
        device=device,
    )
    value_states = torch.randn(
        (batch_size, num_heads, new_seq_len, head_dim),
        dtype=torch.float16,
        device=device,
    )

    context_lens = torch.full((batch_size,), context_len_val, dtype=torch.long, device=device)

    return key_states, value_states, k_cache, v_cache, block_tables, context_lens


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("batch_size, num_heads, head_dim", [(1, 4, 64), (4, 8, 64), (16, 16, 128), (1, 8, 128)])
@pytest.mark.parametrize(
    "context_len, new_seq_len",
    [
        (0, 32),
        (0, 35),
        (31, 1),
        (32, 1),
        (10, 20),
    ],
)
@bypass_not_implemented
def test_paged_update_kernel(block_size, batch_size, num_heads, head_dim, context_len, new_seq_len):
    device = get_platform()
    key_states, value_states, k_cache_ref, v_cache_ref, block_tables, context_lens = generate_inputs(
        batch_size,
        num_heads,
        head_dim,
        new_seq_len,
        context_len,
        block_size,
        device,
    )

    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()

    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")(kv_layout="NPU_ND", block_size=block_size)
    store_paged_kv = MojoStorePagedKVCache(kv_layout="NPU_ND", block_size=block_size)

    store_paged_kv_ref(
        key_states,
        value_states,
        k_cache_ref,
        v_cache_ref,
        block_tables,
        context_lens,
    )

    store_paged_kv(
        key_states,
        value_states,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
    )

    assert_close(k_cache, k_cache_ref)
    assert_close(v_cache, v_cache_ref)
