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


# ===========================================================================
# MojoStorePagedKVCache + CUDA Graph
# ===========================================================================


def generate_store_paged_kv_data(batch_size, kv_heads, head_dim, block_size, kv_lens_val, seq_lens_val):
    kv_lens = torch.tensor(kv_lens_val, dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens_val, dtype=torch.int32)

    is_decode = torch.all(seq_lens == 1)
    cu_seqlens = (
        torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)]
        ).to(torch.int32)
        if not is_decode
        else None
    )

    total_tokens = cu_seqlens[-1].item() if not is_decode else len(kv_lens_val)

    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)

    max_kv_len = (kv_lens + seq_lens).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2

    total_blocks_needed = sum(
        (k + s + block_size - 1) // block_size for k, s in zip(kv_lens_val, seq_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)

    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
    curr = 0
    for i in range(batch_size):
        needed = (kv_lens_val[i] + seq_lens_val[i] + block_size - 1) // block_size
        ids = torch.arange(curr, curr + needed)
        block_table[i, :needed] = ids
        curr += needed

    return key_states, value_states, k_cache, v_cache, block_table, cu_seqlens, kv_lens


def generate_store_paged_kv_data_with_graph(
    batch_size, kv_heads, head_dim, block_size, max_kv_len, max_seq_len
):
    """Generate fixed-shape data for CUDA Graph capture.

    Unlike the non-graph variant, cu_seqlens is ALWAYS created (never None),
    because CUDA Graph requires all kernel arguments to be static buffers.
    """
    kv_lens = torch.tensor([max_kv_len] * batch_size, dtype=torch.int32)
    seq_lens = torch.tensor([max_seq_len] * batch_size, dtype=torch.int32)

    cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)]
    ).to(torch.int32)

    total_tokens = int(cu_seqlens[-1].item())

    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)

    max_kv_total = (kv_lens + seq_lens).max().item()
    max_blocks_per_seq = (max_kv_total + block_size - 1) // block_size + 2

    total_blocks_needed = sum(
        (max_kv_len + max_seq_len + block_size - 1) // block_size for _ in range(batch_size)
    )
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)

    block_table = torch.full((batch_size, max_blocks_per_seq), 0, dtype=torch.int32)
    curr = 0
    for i in range(batch_size):
        needed = (max_kv_len + max_seq_len + block_size - 1) // block_size
        ids = torch.arange(curr, curr + needed)
        block_table[i, :needed] = ids
        curr += needed

    return key_states, value_states, k_cache, v_cache, block_table, cu_seqlens, kv_lens


test_configs_store_paged_kv_with_graph = [
    (4, 2, 128, 128, 32, 1, "DECODE"),
    (4, 2, 128, 128, 0, 130, "PREFILL_NO_CACHE"),
    (4, 2, 128, 128, 40, 126, "PREFILL_WITH_CACHE"),
    (8, 2, 128, 128, 64, 1, "DECODE_8B"),
]


@pytest.mark.parametrize(
    "max_batch_size, kv_heads, head_dim, block_size, max_kv_len, max_seq_len",
    [
        pytest.param(B, KV_H, D, BLK, KV, SL, id=ID)
        for B, KV_H, D, BLK, KV, SL, ID in test_configs_store_paged_kv_with_graph
    ],
)
@bypass_not_implemented
def test_store_paged_kv_with_graph(
    max_batch_size, kv_heads, head_dim, block_size, max_kv_len, max_seq_len
):
    key_states, value_states, k_cache, v_cache, block_table, cu_seqlens, kv_lens = (
        generate_store_paged_kv_data_with_graph(
            max_batch_size, kv_heads, head_dim, block_size, max_kv_len, max_seq_len
        )
    )

    store_paged_kv = MojoStorePagedKVCache()
    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")()

    if type(store_paged_kv_ref) is type(store_paged_kv):
        raise NotImplementedError(
            "both operands resolve to the same implementation, skipping comparison."
        )

    with torch.no_grad():
        # Warm-up
        store_paged_kv(key_states, value_states, k_cache, v_cache, block_table, cu_seqlens, kv_lens)
        synchronize_current_device()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                k_out, v_out = store_paged_kv(
                    key_states, value_states, k_cache, v_cache, block_table, cu_seqlens, kv_lens
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"CUDA graph capture failed: {e}.")
            torch.cuda.empty_cache()
            return

    # Replay with max-shape data and verify
    k_cache.zero_()
    v_cache.zero_()
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()

    graph.replay()
    synchronize_current_device()

    k_cache_ref, v_cache_ref = store_paged_kv_ref(
        key_states, value_states, k_cache_ref, v_cache_ref, block_table, cu_seqlens, kv_lens
    )

    assert_close(k_out, k_cache_ref)
    assert_close(v_out, v_cache_ref)

    # Replay with varying smaller batches
    for _ in range(5):
        current_batch_size = torch.randint(1, max_batch_size + 1, ()).item()

        cur_kv_lens_val = [
            torch.randint(0, max_kv_len + 1, ()).item() for _ in range(current_batch_size)
        ]
        cur_seq_lens_val = [
            torch.randint(1, max_seq_len + 1, ()).item() for _ in range(current_batch_size)
        ]

        cur_kv_lens_t = torch.tensor(cur_kv_lens_val, dtype=torch.int32)
        cur_seq_lens_t = torch.tensor(cur_seq_lens_val, dtype=torch.int32)

        cur_cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(cur_seq_lens_t, dim=0, dtype=torch.int32)]
        ).to(torch.int32)

        cur_total_tokens = int(cur_cu_seqlens[-1].item())

        cur_key_states = torch.randn((cur_total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)
        cur_value_states = torch.randn((cur_total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)

        cur_max_kv_total = (cur_kv_lens_t + cur_seq_lens_t).max().item()
        cur_max_blocks_per_seq = (cur_max_kv_total + block_size - 1) // block_size + 2

        cur_total_blocks_needed = sum(
            (k + s + block_size - 1) // block_size for k, s in zip(cur_kv_lens_val, cur_seq_lens_val)
        )
        cur_total_phys_blocks = cur_total_blocks_needed + 10

        cur_k_cache = torch.zeros((cur_total_phys_blocks, kv_heads, block_size, head_dim), dtype=torch.bfloat16)
        cur_v_cache = torch.zeros((cur_total_phys_blocks, kv_heads, block_size, head_dim), dtype=torch.bfloat16)

        cur_block_table = torch.full((current_batch_size, cur_max_blocks_per_seq), 0, dtype=torch.int32)
        curr = 0
        for i in range(current_batch_size):
            needed = (cur_kv_lens_val[i] + cur_seq_lens_val[i] + block_size - 1) // block_size
            cur_block_table[i, :needed] = torch.arange(curr, curr + needed)
            curr += needed

        # In-place update static buffers
        key_states[:cur_total_tokens].copy_(cur_key_states)
        value_states[:cur_total_tokens].copy_(cur_value_states)

        cur_num_blocks = cur_k_cache.shape[0]
        k_cache.zero_()
        v_cache.zero_()
        k_cache[:cur_num_blocks].copy_(cur_k_cache)
        v_cache[:cur_num_blocks].copy_(cur_v_cache)

        # kv_lens: valid batches copied, padded batches set to 0
        kv_lens[:current_batch_size].copy_(cur_kv_lens_t)
        kv_lens[current_batch_size:] = 0

        # block_table: valid entries copied; padded rows/cols get safe block id (0)
        block_table.fill_(0)
        block_table[:current_batch_size, :cur_block_table.shape[1]].copy_(cur_block_table)

        # cu_seqlens: valid prefix copied, padded batches keep final cumsum value
        cu_seqlens[:current_batch_size + 1].copy_(cur_cu_seqlens)
        cu_seqlens[current_batch_size + 1:] = cur_cu_seqlens[-1]

        # Reference computation (always pass cu_seqlens, never None)
        k_cache_ref = cur_k_cache.clone()
        v_cache_ref = cur_v_cache.clone()
        k_cache_ref, v_cache_ref = store_paged_kv_ref(
            cur_key_states, cur_value_states, k_cache_ref, v_cache_ref,
            cur_block_table, cur_cu_seqlens, cur_kv_lens_t,
        )

        graph.replay()
        synchronize_current_device()

        # Verify that the valid cache blocks match reference
        check_tol_diff(k_out[:cur_num_blocks], k_cache_ref, atol=0, rtol=0)
        check_tol_diff(v_out[:cur_num_blocks], v_cache_ref, atol=0, rtol=0)


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
    kv_lens = torch.tensor(kv_lens_val, dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens_val, dtype=torch.int32)

    is_decode = torch.all(seq_lens == 1)
    cu_seqlens = (
        torch.cat([
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ])
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

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
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
