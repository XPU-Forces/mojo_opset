import pytest
import torch

from mojo_opset import MojoStorePagedKVCache
from mojo_opset import MojoStorePagedMLAKVCache
from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import host_perf
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata


def _build_store_paged_kv_case(
    batch_size,
    kv_heads,
    head_dim,
    block_size,
    context_kv_lens_val,
    q_lens_val,
    *,
    device,
):
    context_kv_lens = torch.tensor(context_kv_lens_val, dtype=torch.int32, device=device)
    q_lens = torch.tensor(q_lens_val, dtype=torch.int32, device=device)

    is_decode = torch.all(q_lens == 1).item()
    cu_q_lens = (
        torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(q_lens, dim=0, dtype=torch.int32)]
        )
        if not is_decode
        else None
    )

    total_tokens = int(q_lens.sum().item()) if not is_decode else batch_size
    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device=device)

    max_kv_len = torch.clamp(context_kv_lens + q_lens, min=0).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks_needed = sum(
        max(0, context_kv_len + q_len + block_size - 1) // block_size
        for context_kv_len, q_len in zip(context_kv_lens_val, q_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    next_block = 0
    for batch_id in range(batch_size):
        needed = max(0, context_kv_lens_val[batch_id] + q_lens_val[batch_id] + block_size - 1) // block_size
        if needed > 0:
            block_table[batch_id, :needed] = torch.arange(
                next_block,
                next_block + needed,
                dtype=torch.int32,
                device=device,
            )
        next_block += needed

    chunk_metadata = build_paged_kv_chunk_metadata(
        block_table,
        cu_q_lens,
        context_kv_lens,
        block_size,
    )

    return {
        "context_kv_lens": context_kv_lens,
        "q_lens": q_lens,
        "cu_q_lens": cu_q_lens,
        "key_states": key_states,
        "value_states": value_states,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "chunk_metadata": chunk_metadata,
    }


@pytest.mark.parametrize(
    "batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val",
    [
        (2, 2, 128, 128, [0, 0], [130, 33]),
        (2, 2, 128, 128, [32, 35], [1, 1]),
        (2, 2, 128, 128, [15, 40], [788, 126]),
        (2, 2, 128, 256, [15, 40], [788, 126]),
        (2, 2, 128, 512, [255, 511], [300, 257]),
        (2, 2, 128, 1024, [511, 1023], [600, 513]),
        (2, 2, 128, 2048, [1023, 2047], [900, 1025]),
        (1, 1, 128, 128, [0], [5]),
        (1, 1, 128, 128, [5], [1]),
        (1, 1, 128, 512, [510], [3]),
        (1, 1, 128, 1024, [1022], [2]),
        (1, 1, 128, 2048, [2046], [2]),
        (3, 2, 128, 128, [32, -1, 35], [1, 1, 1]),
        (3, 2, 128, 128, [0, -1, 5], [4, 0, 2]),
        (3, 2, 128, 512, [510, -1, 700], [4, 1, 300]),
        (3, 2, 128, 1024, [1020, -1, 1530], [8, 1, 520]),
        (3, 2, 128, 2048, [2040, -1, 3000], [16, 1, 900]),
        (8, 2, 128, 128, [224, 542, 34, 41, 54, 57, 65, 0], [432, 84, 977, 93, 23, 89, 31, 555]),
        (8, 2, 128, 128, [772, 974, 3232, 43, 77, 7633, 888, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
        (
            8,
            2,
            128,
            512,
            [224, 542, 34, 41, 54, 57, 65, 0],
            [432, 84, 977, 93, 23, 89, 31, 555],
        ),
        (
            8,
            2,
            128,
            1024,
            [900, 1500, 34, 41, 54, 57, 65, 0],
            [700, 600, 977, 93, 23, 89, 31, 555],
        ),
        (
            8,
            2,
            128,
            2048,
            [1800, 2500, 34, 41, 54, 57, 65, 0],
            [900, 1200, 977, 93, 23, 89, 31, 555],
        ),
        (
            8,
            2,
            128,
            512,
            [772, 974, 3232, 43, 77, 7633, 888, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ),
        (
            8,
            2,
            128,
            1024,
            [1023, 1024, 3232, 43, 77, 7633, 888, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ),
        (
            8,
            2,
            128,
            2048,
            [2047, 2048, 3232, 43, 77, 7633, 888, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ),
    ],
)
@bypass_not_implemented
def test_store_paged_kv(batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val):
    case = _build_store_paged_kv_case(
        batch_size,
        kv_heads,
        head_dim,
        block_size,
        context_kv_lens_val,
        q_lens_val,
        device=get_torch_device(),
    )

    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")()
    store_paged_kv = MojoStorePagedKVCache()
    if type(store_paged_kv_ref) is type(store_paged_kv):
        raise NotImplementedError("both operands resolve to the same implementation, skipping comparison.")

    k_cache_ref, v_cache_ref = store_paged_kv_ref(
        case["key_states"],
        case["value_states"],
        case["k_cache"].clone(),
        case["v_cache"].clone(),
        chunk_metadata=case["chunk_metadata"],
    )
    k_cache, v_cache = store_paged_kv(
        case["key_states"],
        case["value_states"],
        case["k_cache"].clone(),
        case["v_cache"].clone(),
        chunk_metadata=case["chunk_metadata"],
    )

    assert_close(k_cache, k_cache_ref)
    assert_close(v_cache, v_cache_ref)


@auto_switch_platform()
@bypass_not_implemented
def test_store_paged_kv_bucket_padded_varlen():
    real_batch_size = 4
    bucket_batch_size = 6
    kv_heads = 2
    head_dim = 128
    block_size = 8
    device = get_torch_device()

    cu_q_lens = torch.tensor([0, 1, 2, 3, 4, 4, 4], dtype=torch.int32, device=device)
    context_kv_lens = torch.tensor([0, 2, 7, 9, -1, -1], dtype=torch.int32, device=device)
    total_tokens = int(cu_q_lens[-1].item())

    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16, device=device)

    max_kv_len = (context_kv_lens[:real_batch_size] + 1).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 1
    total_phys_blocks = bucket_batch_size * max_blocks_per_seq + 4

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    v_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    block_table = torch.full((bucket_batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    next_block = 0
    for batch_id in range(real_batch_size):
        needed = (context_kv_lens[batch_id].item() + 1 + block_size - 1) // block_size
        block_table[batch_id, :needed] = torch.arange(
            next_block,
            next_block + needed,
            dtype=torch.int32,
            device=device,
        )
        next_block += needed

    chunk_metadata = build_paged_kv_chunk_metadata(block_table, cu_q_lens, context_kv_lens, block_size)

    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")()
    store_paged_kv = MojoStorePagedKVCache()
    if type(store_paged_kv_ref) is type(store_paged_kv):
        raise NotImplementedError("both operands resolve to the same implementation, skipping comparison.")

    k_cache_ref, v_cache_ref = store_paged_kv_ref(
        key_states,
        value_states,
        k_cache_ref,
        v_cache_ref,
        chunk_metadata=chunk_metadata,
    )
    k_cache, v_cache = store_paged_kv(
        key_states,
        value_states,
        k_cache,
        v_cache,
        chunk_metadata=chunk_metadata,
    )

    for batch_id in range(real_batch_size):
        write_pos = context_kv_lens[batch_id].item()
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


@bypass_not_implemented
def test_store_paged_kv_chunk_metadata_perf_and_accuracy():
    if get_platform() != "npu":
        pytest.skip("chunk metadata performance comparison is NPU-only")

    from mojo_opset.backends.ttx.kernels.npu.kv_cache import store_paged_kv_impl_legacy

    real_batch_size = 8
    batch_size = 16384
    kv_heads = 32
    head_dim = 128
    block_size = 128
    context_kv_lens_val = [0, 17, 255, 1023, 4095, 63, 8191, 511] + [-1] * (batch_size - real_batch_size)
    q_lens_val = [1] * real_batch_size + [0] * (batch_size - real_batch_size)
    device = get_torch_device()

    case = _build_store_paged_kv_case(
        batch_size,
        kv_heads,
        head_dim,
        block_size,
        context_kv_lens_val,
        q_lens_val,
        device=device,
    )

    store_paged_kv = MojoStorePagedKVCache()
    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")()
    if type(store_paged_kv_ref) is type(store_paged_kv):
        raise NotImplementedError("both operands resolve to the same implementation, skipping comparison.")

    k_cache_new, v_cache_new = store_paged_kv(
        case["key_states"],
        case["value_states"],
        case["k_cache"].clone(),
        case["v_cache"].clone(),
        chunk_metadata=case["chunk_metadata"],
    )
    k_cache_legacy, v_cache_legacy = store_paged_kv_impl_legacy(
        case["key_states"],
        case["value_states"],
        case["k_cache"].clone(),
        case["v_cache"].clone(),
        case["block_table"],
        case["cu_q_lens"],
        case["context_kv_lens"],
    )

    assert_close(k_cache_new, k_cache_legacy)
    assert_close(v_cache_new, v_cache_legacy)

    k_cache_bench_new = case["k_cache"].clone()
    v_cache_bench_new = case["v_cache"].clone()
    k_cache_bench_legacy = case["k_cache"].clone()
    v_cache_bench_legacy = case["v_cache"].clone()

    new_latency_ms = host_perf(
        lambda: store_paged_kv(
            case["key_states"],
            case["value_states"],
            k_cache_bench_new,
            v_cache_bench_new,
            chunk_metadata=case["chunk_metadata"],
        ),
        "npu",
        warmup=10,
        repeat=50,
    )
    legacy_latency_ms = host_perf(
        lambda: store_paged_kv_impl_legacy(
            case["key_states"],
            case["value_states"],
            k_cache_bench_legacy,
            v_cache_bench_legacy,
            case["block_table"],
            case["cu_q_lens"],
            case["context_kv_lens"],
        ),
        "npu",
        warmup=10,
        repeat=50,
    )

    assert new_latency_ms < legacy_latency_ms, (
        f"chunk-metadata kernel should outperform legacy sequence-chunk kernel on bucket-padded sparse batches. "
        f"new={new_latency_ms:.3f} ms, legacy={legacy_latency_ms:.3f} ms"
    )


# ===========================================================================
# MojoStorePagedMLAKVCache
# ===========================================================================

@pytest.mark.parametrize(
    "batch_size, kv_lora_rank, qk_rope_head_dim, block_size, context_kv_lens_val, q_lens_val",
    [
        (2, 64, 32, 128, [0, 0], [130, 33]),
        (2, 64, 32, 128, [32, 35], [1, 1]),
        (2, 128, 64, 128, [15, 40], [788, 126]),
        (1, 64, 32, 128, [0], [5]),
        (1, 64, 32, 128, [5], [1]),
        (3, 64, 32, 128, [32, -1, 35], [1, 1, 1]),
        (3, 64, 32, 128, [0, -1, 5], [4, 0, 2]),
        (4, 64, 32, 256, [224, 0, 34, 41], [432, 84, 977, 93]),
        (4, 64, 32, 128, [772, 974, 43, 77], [1, 1, 1, 1]),
    ],
)
@bypass_not_implemented
def test_store_paged_mla_kv(
    batch_size,
    kv_lora_rank,
    qk_rope_head_dim,
    block_size,
    context_kv_lens_val,
    q_lens_val,
):
    context_kv_lens = torch.tensor(context_kv_lens_val, dtype=torch.int32)
    q_lens = torch.tensor(q_lens_val, dtype=torch.int32)

    is_decode = torch.all(q_lens == 1)
    cu_q_lens = (
        torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(q_lens, dim=0, dtype=torch.int32),
            ]
        )
        if not is_decode
        else None
    )

    total_tokens = cu_q_lens[-1].item() if not is_decode else len(context_kv_lens_val)
    ckv_states = torch.randn(total_tokens, kv_lora_rank, dtype=torch.bfloat16)
    kpe_states = torch.randn(total_tokens, qk_rope_head_dim, dtype=torch.bfloat16)

    max_kv_len = torch.clamp(context_kv_lens + q_lens, min=0).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks_needed = sum(
        max(0, context_kv_len + q_len + block_size - 1) // block_size
        for context_kv_len, q_len in zip(context_kv_lens_val, q_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    ckv_cache_ref = torch.zeros(total_phys_blocks, 1, block_size, kv_lora_rank, dtype=torch.bfloat16)
    kpe_cache_ref = torch.zeros(total_phys_blocks, 1, block_size, qk_rope_head_dim, dtype=torch.bfloat16)
    ckv_cache = ckv_cache_ref.clone()
    kpe_cache = kpe_cache_ref.clone()

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
    curr = 0
    for i in range(batch_size):
        needed = max(0, context_kv_lens_val[i] + q_lens_val[i] + block_size - 1) // block_size
        block_table[i, :needed] = torch.arange(curr, curr + needed)
        curr += needed

    op_ref = MojoStorePagedMLAKVCache._registry.get("torch")()
    op = MojoStorePagedMLAKVCache()
    ckv_cache_ref, kpe_cache_ref = op_ref(
        ckv_states,
        kpe_states,
        ckv_cache_ref,
        kpe_cache_ref,
        block_table,
        cu_q_lens,
        context_kv_lens,
    )
    ckv_cache, kpe_cache = op(
        ckv_states,
        kpe_states,
        ckv_cache,
        kpe_cache,
        block_table,
        cu_q_lens,
        context_kv_lens,
    )

    assert_close(ckv_cache, ckv_cache_ref)
    assert_close(kpe_cache, kpe_cache_ref)
