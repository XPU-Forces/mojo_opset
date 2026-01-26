import pytest
import torch

from tests.utils import assert_close
from tests.utils import bypass_not_implemented

from mojo_opset import MojoStorePagedKVCache
from mojo_opset.utils.platform import get_platform


@pytest.mark.parametrize(
    "batch_size, kv_heads, head_dim, block_size, kv_lens_val, seq_lens_val",
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
def test_store_paged_kv(batch_size, kv_heads, head_dim, block_size, kv_lens_val, seq_lens_val):
    device = get_platform()

    kv_lens = torch.tensor(kv_lens_val, dtype=torch.long, device=device)
    seq_lens = torch.tensor(seq_lens_val, dtype=torch.long, device=device)

    cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)]
    ).to(torch.long)

    total_tokens = cu_seqlens[-1].item()

    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device=device)

    max_kv_len = (kv_lens + seq_lens).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2

    total_blocks_needed = sum([(k + s + block_size - 1) // block_size for k, s in zip(kv_lens_val, seq_lens_val)])
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)

    k_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    v_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.long, device=device)
    curr = 0
    for i in range(batch_size):
        needed = (kv_lens_val[i] + seq_lens_val[i] + block_size - 1) // block_size
        ids = torch.arange(curr, curr + needed, device=device)
        block_table[i, :needed] = ids
        curr += needed

    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()

    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")()
    store_paged_kv = MojoStorePagedKVCache()

    store_paged_kv_ref(
        key_states,
        value_states,
        k_cache_ref,
        v_cache_ref,
        block_table,
        cu_seqlens,
        kv_lens,
    )

    store_paged_kv(
        key_states,
        value_states,
        k_cache,
        v_cache,
        block_table,
        cu_seqlens,
        kv_lens,
    )

    assert_close(k_cache, k_cache_ref)
    assert_close(v_cache, v_cache_ref)
