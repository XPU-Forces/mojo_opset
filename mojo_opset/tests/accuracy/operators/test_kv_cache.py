import pytest
import torch
import math
import random

from mojo_opset import MojoStorePagedKVCache
from mojo_opset.experimental import MojoStorePagedMLAKVCache
from mojo_opset.experimental import MojoStorePagedKVCacheC8
from mojo_opset.experimental import MojoDequantFromPagedKVCache
from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import host_perf
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata


def _assert_int8_cache_close(result, ref, mismatch_ratio=1e-5):
    assert result.shape == ref.shape
    assert result.dtype == ref.dtype

    diff = torch.abs(result.cpu().to(torch.int16) - ref.cpu().to(torch.int16))
    assert diff.max().item() <= 1
    assert torch.count_nonzero(diff).item() / diff.numel() <= mismatch_ratio


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
def test_store_paged_kv_c8(batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val):
    case = _build_store_paged_kv_case(
        batch_size,
        kv_heads,
        head_dim,
        block_size,
        context_kv_lens_val,
        q_lens_val,
        device=get_torch_device(),
    )
    cache_scale = torch.randn((2, kv_heads, head_dim), dtype=torch.float, device="mlu")
    key_scale = cache_scale[0]
    value_scale = cache_scale[1]

    store_paged_kv_c8_ref = MojoStorePagedKVCacheC8._registry.get("torch")()
    store_paged_kv_c8 = MojoStorePagedKVCacheC8()
    if type(store_paged_kv_c8_ref) is type(store_paged_kv_c8):
        raise NotImplementedError("both operands resolve to the same implementation, skipping comparison.")

    k_cache_ref, v_cache_ref = store_paged_kv_c8_ref(
        case["key_states"],
        case["value_states"],
        case["k_cache"].clone().to(torch.int8),
        case["v_cache"].clone().to(torch.int8),
        key_scale,
        value_scale,
        chunk_metadata=case["chunk_metadata"],
    )
    k_cache, v_cache = store_paged_kv_c8(
        case["key_states"],
        case["value_states"],
        case["k_cache"].clone().to(torch.int8),
        case["v_cache"].clone().to(torch.int8),
        key_scale,
        value_scale,
        chunk_metadata=case["chunk_metadata"],
    )

    _assert_int8_cache_close(k_cache, k_cache_ref)
    _assert_int8_cache_close(v_cache, v_cache_ref)


@pytest.mark.parametrize(
    "batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val",
    [
        (1, 2, 128, 16, [0], [3]),
        (1, 2, 128, 128, [127], [1]),
        (2, 4, 128, 32, [5, 33], [7, 19]),
        (2, 4, 128, 256, [255, 511], [1, 1]),
        (3, 8, 128, 64, [0, 11, 95], [5, 17, 29]),
        (3, 8, 128, 128, [17, -1, 63], [1, 1, 1]),
        (4, 16, 128, 128, [0, 3, 127, 255], [9, 17, 33, 65]),
        (4, 16, 128, 512, [511, 1025, 7, 63], [1, 1, 1, 1]),
        (5, 24, 128, 64, [13, 97, 0, 255, 511], [31, 65, 7, 19, 127]),
        (5, 24, 128, 256, [255, 511, -1, 33, 777], [1, 1, 1, 1, 1]),
        (6, 24, 128, 1024, [1023, 17, 2047, 0, 4097, 63], [1, 1, 1, 1, 1, 1]),
        (6, 24, 128, 128, [31, 511, 1023, 7, 95, 1535], [129, 257, 513, 5, 17, 65]),
    ],
)
@bypass_not_implemented
def test_store_paged_kv_without_chunk_metadata(
    batch_size,
    kv_heads,
    head_dim,
    block_size,
    context_kv_lens_val,
    q_lens_val,
):
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
        case["block_table"],
        case["cu_q_lens"],
        case["context_kv_lens"],
    )
    k_cache, v_cache = store_paged_kv(
        case["key_states"],
        case["value_states"],
        case["k_cache"].clone(),
        case["v_cache"].clone(),
        case["block_table"],
        case["cu_q_lens"],
        case["context_kv_lens"],
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


def gen_args(
    batch_size,
    max_context_len,
    head_num_q,
    head_num_kv,
    cache_mem_len,
    head_size,
    group_size,
    block_size,
    use_seq_offset,
    dtype,
    quant_mode,
    quant_bit,
    pad_head_size=0,
    has_value=True,
    context_strided=False,
):
    # Preprocess arguments
    assert cache_mem_len >= max_context_len, "cache_mem_len should greater then or equal to max_context_len."
    assert head_size % group_size == 0, "head_size should be a multiply of groupwise."
    total_heads = head_num_q + head_num_kv * 2
    max_seq_offset = cache_mem_len - max_context_len
    max_block_num = int(math.ceil(max_context_len / block_size))
    total_blocks = int(math.ceil(cache_mem_len / block_size)) * batch_size // 4 * 4
    block_tables = random.sample(range(0, total_blocks), batch_size * max_block_num)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="mlu").view(batch_size, max_block_num)
    # Generates key and cache from context
    context_lens = torch.randint(size=[batch_size], low=1, high=max_context_len + 1, dtype=torch.int32, device="mlu")
    if use_seq_offset:
        context_paddings = torch.randint(size=[batch_size], low=0, high=max_seq_offset, dtype=torch.int32, device="mlu")
    else:
        context_paddings = torch.zeros_like(context_lens)
    cu_context_lens = torch.cumsum(context_lens + context_paddings, dim=-1)
    total_seqlen = cu_context_lens[-1]
    context_seq_offset = torch.zeros([batch_size], dtype=torch.int32, device="mlu")
    context_seq_offset[1:] = cu_context_lens[:-1]
    if context_strided:
        context = torch.randn([total_heads, total_seqlen, max(pad_head_size, head_size)], dtype=dtype, device="mlu")
        context = context.transpose(0, 1)
    else:
        context = torch.randn([total_seqlen, total_heads, max(pad_head_size, head_size)], dtype=dtype, device="mlu")
    key = context[..., head_num_q : head_num_q + head_num_kv, :head_size]
    value = None
    dim = 2 if has_value else 1
    cache = (
        torch.randint(
            size=(dim, total_blocks // 4, head_num_kv, block_size, head_size),
            low=-128,
            high=127,
            dtype=torch.int32,
            device="mlu",
        )
        .view(torch.int8)
        .view(dim, total_blocks, head_num_kv, block_size, head_size)
    )

    # Generates key_cache_scale and value_cache_scale
    if quant_mode == 0:  # quant_mode == 0 is per channel
        cache_scale = torch.randn((dim, head_num_kv, head_size), dtype=torch.float, device="mlu")
    else:  # quant_mode != 1 (== 1 for extend) is per head
        cache_scale = torch.randn((dim, total_blocks, head_num_kv, block_size), dtype=torch.float, device="mlu")
    key_cache = cache[0]
    value_cache = None
    key_cache_scale = cache_scale[0]
    value_cache_scale = None
    # Prepare arguments
    if has_value:
        value = context[..., head_num_q + head_num_kv : head_num_q + 2 * head_num_kv, :head_size]
        value_cache = cache[1]
        value_cache_scale = cache_scale[1]
    args = [key, value, key_cache, value_cache, key_cache_scale, value_cache_scale]
    args += [context_lens, max_context_len, context_seq_offset if use_seq_offset else None, block_tables]
    args += [quant_mode, quant_bit]
    return args

@pytest.mark.ci
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("head_num_q", [4])
@pytest.mark.parametrize("max_context_len", [512])
@pytest.mark.parametrize("head_num_kv", [32])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("pad_head_size", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("cache_mem_len", [768, 1024])
@pytest.mark.parametrize("quant_mode", [0])
@pytest.mark.parametrize("quant_bit", [8])
@pytest.mark.parametrize("use_seq_offset", [False, True])
@pytest.mark.parametrize("has_value", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("context_strided", [False, True])
@bypass_not_implemented
def test_dequant_from_paged_cache(
    batch_size,
    head_num_q,
    max_context_len,
    head_num_kv,
    head_size,
    pad_head_size,
    block_size,
    cache_mem_len,
    quant_mode,
    quant_bit,
    use_seq_offset,
    has_value,
    dtype,
    context_strided,
):
    print(
        "batch_size={}, head_num={}, head_size={}, max_context_len={}, quant_mode={}, "
        "quant_bit={}, dtype={},context_strided={} testing...".format(
            batch_size, head_num_kv, head_size, max_context_len, quant_mode, quant_bit, dtype, context_strided
        )
    )

    args = gen_args(
        batch_size,
        max_context_len,
        head_num_q,
        head_num_kv,
        cache_mem_len,
        head_size,
        head_size,
        block_size,
        use_seq_offset,
        dtype,
        quant_mode,
        quant_bit,
        pad_head_size,
        has_value,
        context_strided,
    )
    (
        key,
        value,
        key_cache,
        value_cache,
        key_cache_scale,
        value_cache_scale,
        context_lengths,
        max_context_len,
        context_seq_offset,
        block_tables,
        quant_mode,
        quant_bit,
    ) = args
    key_ref = key.clone()
    value_ref = value.clone()

    dequant_from_paged_kv_cache_ref = MojoDequantFromPagedKVCache._registry.get("torch")()
    dequant_from_paged_kv_cache = MojoDequantFromPagedKVCache()
    if type(dequant_from_paged_kv_cache_ref) is type(dequant_from_paged_kv_cache):
        raise NotImplementedError("both operands resolve to the same implementation, skipping comparison.")

    key, value = dequant_from_paged_kv_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        key_cache_scale=key_cache_scale,
        value_cache=value_cache,
        value_cache_scale=value_cache_scale,
        context_lengths=context_lengths,
        max_context_len=max_context_len,
        context_seq_offset=context_seq_offset,
        block_tables=block_tables,
    )

    key_ref, value_ref = dequant_from_paged_kv_cache_ref(
        key=key_ref,
        value=value_ref,
        key_cache=key_cache,
        key_cache_scale=key_cache_scale,
        value_cache=value_cache,
        value_cache_scale=value_cache_scale,
        context_lengths=context_lengths,
        max_context_len=max_context_len,
        context_seq_offset=context_seq_offset,
        block_tables=block_tables,
    )

    assert_close(key, key_ref)
    assert_close(value, value_ref)
