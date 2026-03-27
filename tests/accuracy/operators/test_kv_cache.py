import pytest
import torch

from tests.utils import assert_close
from tests.utils import bypass_not_implemented

from mojo_opset import MojoStorePagedKVCache
from mojo_opset import MojoStorePagedMLAKVCache


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
    kv_lens = torch.tensor(kv_lens_val, dtype=torch.long)
    seq_lens = torch.tensor(seq_lens_val, dtype=torch.long)

    is_decode = torch.all(seq_lens == 1)

    cu_seqlens = (
        torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)]
        ).to(torch.long)
        if not is_decode
        else None
    )

    total_tokens = cu_seqlens[-1].item() if not is_decode else len(kv_lens_val)

    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)

    max_kv_len = (kv_lens + seq_lens).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2

    total_blocks_needed = sum([(k + s + block_size - 1) // block_size for k, s in zip(kv_lens_val, seq_lens_val)])
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)

    k_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16)

    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.long)
    curr = 0
    for i in range(batch_size):
        needed = (kv_lens_val[i] + seq_lens_val[i] + block_size - 1) // block_size
        ids = torch.arange(curr, curr + needed)
        block_table[i, :needed] = ids
        curr += needed

    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()

    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")()
    store_paged_kv = MojoStorePagedKVCache()

    if type(store_paged_kv_ref) is type(store_paged_kv):
        raise NotImplementedError(
            f"both operands resolve to the same implementation, skipping comparison."
        )

    k_cache_ref, v_cache_ref = store_paged_kv_ref(
        key_states,
        value_states,
        k_cache_ref,
        v_cache_ref,
        block_table,
        cu_seqlens,
        kv_lens,
    )

    k_cache, v_cache = store_paged_kv(
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


# ===========================================================================
# MojoStorePagedMLAKVCache
# ===========================================================================

@pytest.mark.parametrize(
    "batch_size, kv_lora_rank, qk_rope_head_dim, block_size, kv_lens_val, seq_lens_val",
    [
        (2, 64, 32, 128, [0, 0], [130, 33]),
        (2, 64, 32, 128, [32, 35], [1, 1]),
        (2, 128, 64, 128, [15, 40], [788, 126]),
        (1, 64, 32, 128, [0], [5]),
        (1, 64, 32, 128, [5], [1]),
        (4, 64, 32, 256, [224, 0, 34, 41], [432, 84, 977, 93]),
        (4, 64, 32, 128, [772, 974, 43, 77], [1, 1, 1, 1]),
    ],
)
@bypass_not_implemented
def test_store_paged_mla_kv(batch_size, kv_lora_rank, qk_rope_head_dim, block_size,
                            kv_lens_val, seq_lens_val):
    kv_lens = torch.tensor(kv_lens_val, dtype=torch.long)
    seq_lens = torch.tensor(seq_lens_val, dtype=torch.long)

    is_decode = torch.all(seq_lens == 1)
    cu_seqlens = (
        torch.cat([
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]).to(torch.long)
        if not is_decode
        else None
    )

    total_tokens = cu_seqlens[-1].item() if not is_decode else len(kv_lens_val)

    ckv_states = torch.randn(total_tokens, kv_lora_rank, dtype=torch.bfloat16)
    kpe_states = torch.randn(total_tokens, qk_rope_head_dim, dtype=torch.bfloat16)

    max_kv_len = (kv_lens + seq_lens).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks_needed = sum(
        (k + s + block_size - 1) // block_size for k, s in zip(kv_lens_val, seq_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    ckv_cache_ref = torch.zeros(total_phys_blocks, 1, block_size, kv_lora_rank, dtype=torch.bfloat16)
    kpe_cache_ref = torch.zeros(total_phys_blocks, 1, block_size, qk_rope_head_dim, dtype=torch.bfloat16)
    ckv_cache = ckv_cache_ref.clone()
    kpe_cache = kpe_cache_ref.clone()

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.long)
    curr = 0
    for i in range(batch_size):
        needed = (kv_lens_val[i] + seq_lens_val[i] + block_size - 1) // block_size
        block_table[i, :needed] = torch.arange(curr, curr + needed)
        curr += needed

    op_ref = MojoStorePagedMLAKVCache._registry.get("torch")()
    op = MojoStorePagedMLAKVCache()

    ckv_cache_ref, kpe_cache_ref = op_ref(
        ckv_states, kpe_states, ckv_cache_ref, kpe_cache_ref,
        block_table, cu_seqlens, kv_lens,
    )
    ckv_cache, kpe_cache = op(
        ckv_states, kpe_states, ckv_cache, kpe_cache,
        block_table, cu_seqlens, kv_lens,
    )

    assert_close(ckv_cache, ckv_cache_ref)
    assert_close(kpe_cache, kpe_cache_ref)
