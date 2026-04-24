import pytest
import torch

from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import synchronize_current_device
from mojo_opset.utils.platform import get_torch_device

from mojo_opset import MojoStorePagedKVCache
from mojo_opset import MojoStorePagedMLAKVCache
from mojo_opset.utils.acc import check_tol_diff


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
    kv_lens = torch.tensor(kv_lens_val, dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens_val, dtype=torch.int32)

    is_decode = torch.all(seq_lens == 1)

    cu_seqlens = (
        torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)]
        )
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

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
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


@auto_switch_platform()
@bypass_not_implemented
def test_store_paged_kv_bucket_padded_varlen():
    real_batch_size = 4
    bucket_batch_size = 6
    token_bucket_size = 8
    kv_heads = 2
    head_dim = 128
    block_size = 8

    cu_seqlens = torch.tensor([0, 1, 2, 3, 4, 4, 4], dtype=torch.int32)
    kv_lens = torch.tensor([0, 2, 7, 9, 0, 0], dtype=torch.int32)

    key_states = torch.randn((token_bucket_size, kv_heads, head_dim), dtype=torch.bfloat16)
    value_states = torch.randn((token_bucket_size, kv_heads, head_dim), dtype=torch.bfloat16)

    max_kv_len = (kv_lens[:real_batch_size] + 1).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 1
    total_phys_blocks = bucket_batch_size * max_blocks_per_seq + 4

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16)
    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)

    block_table = torch.full((bucket_batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
    next_block = 0
    for batch_id in range(real_batch_size):
        needed = (kv_lens[batch_id].item() + 1 + block_size - 1) // block_size
        block_table[batch_id, :needed] = torch.arange(next_block, next_block + needed, dtype=torch.int32)
        next_block += needed

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

    for batch_id in range(real_batch_size):
        write_pos = kv_lens[batch_id].item()
        block_idx = write_pos // block_size
        block_offset = write_pos % block_size
        phys_block = block_table[batch_id, block_idx].item()
        assert_close(
            k_cache[phys_block, :, block_offset : block_offset + 1, :],
            k_cache_ref[phys_block, :, block_offset : block_offset + 1, :],
        )
        assert_close(
            v_cache[phys_block, :, block_offset : block_offset + 1, :],
            v_cache_ref[phys_block, :, block_offset : block_offset + 1, :],
        )

