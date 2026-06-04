import pytest
import torch
import torch_npu

from mojo_opset import MojoStorePagedKVCache
from mojo_opset import MojoStorePagedMLAKVCache
from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.platform import get_torch_device


def _available_float8_dtypes():
    return [
        dtype
        for name in ("float8_e4m3fn", "float8_e5m2")
        if (dtype := getattr(torch, name, None)) is not None
    ]


def _rand_float8(shape, dtype, device):
    return torch.randint(-2, 3, shape, dtype=torch.int8, device=device).to(torch.float16).to(dtype)


def _assert_exact_float8_cache(actual, expected):
    torch.testing.assert_close(actual.to(torch.float32), expected.to(torch.float32), atol=0, rtol=0)


@pytest.mark.parametrize(
    "batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val",
    [
        (2, 2, 128, 128, [0, 0], [130, 33]),
        (2, 2, 128, 128, [32, 35], [1, 1]),
        (2, 2, 128, 128, [15, 40], [788, 126]),
        (2, 2, 128, 256, [15, 40], [788, 126]),
        (1, 1, 128, 128, [0], [5]),
        (1, 1, 128, 128, [5], [1]),
        (3, 2, 128, 128, [32, -1, 35], [1, 1, 1]),
        (3, 2, 128, 128, [0, -1, 5], [4, 0, 2]),
        (8, 2, 128, 128, [224, 542, 34, 41, 54, 57, 65, 0], [432, 84, 977, 93, 23, 89, 31, 555]),
        (8, 2, 128, 128, [772, 974, 3232, 43, 77, 7633, 888, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    ],
)
@bypass_not_implemented
def test_store_paged_kv(batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val):
    context_kv_lens = torch.tensor(context_kv_lens_val, dtype=torch.int32)
    q_lens = torch.tensor(q_lens_val, dtype=torch.int32)

    is_decode = torch.all(q_lens == 1)
    cu_q_lens = (
        torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, dim=0, dtype=torch.int32)]
        )
        if not is_decode
        else None
    )

    total_tokens = cu_q_lens[-1].item() if not is_decode else len(context_kv_lens_val)
    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=torch.bfloat16)

    max_kv_len = torch.clamp(context_kv_lens + q_lens, min=0).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks_needed = sum(
        max(0, context_kv_len + q_len + block_size - 1) // block_size
        for context_kv_len, q_len in zip(context_kv_lens_val, q_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16)
    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
    curr = 0
    for i in range(batch_size):
        needed = max(0, context_kv_lens_val[i] + q_lens_val[i] + block_size - 1) // block_size
        block_table[i, :needed] = torch.arange(curr, curr + needed)
        curr += needed

    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")()
    store_paged_kv = MojoStorePagedKVCache()
    if type(store_paged_kv_ref) is type(store_paged_kv):
        raise NotImplementedError("both operands resolve to the same implementation, skipping comparison.")

    k_cache_ref, v_cache_ref = store_paged_kv_ref(
        key_states,
        value_states,
        k_cache_ref,
        v_cache_ref,
        block_table,
        cu_q_lens,
        context_kv_lens,
    )
    k_cache, v_cache = store_paged_kv(
        key_states,
        value_states,
        k_cache,
        v_cache,
        block_table,
        cu_q_lens,
        context_kv_lens,
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

    cu_q_lens = torch.tensor([0, 1, 2, 3, 4, 4, 4], dtype=torch.int32)
    context_kv_lens = torch.tensor([0, 2, 7, 9, -1, -1], dtype=torch.int32)

    key_states = torch.randn((token_bucket_size, kv_heads, head_dim), dtype=torch.bfloat16)
    value_states = torch.randn((token_bucket_size, kv_heads, head_dim), dtype=torch.bfloat16)

    max_kv_len = (context_kv_lens[:real_batch_size] + 1).max().item()
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
        needed = (context_kv_lens[batch_id].item() + 1 + block_size - 1) // block_size
        block_table[batch_id, :needed] = torch.arange(next_block, next_block + needed, dtype=torch.int32)
        next_block += needed

    store_paged_kv_ref = MojoStorePagedKVCache._registry.get("torch")()
    store_paged_kv = MojoStorePagedKVCache()
    if type(store_paged_kv_ref) is type(store_paged_kv):
        raise NotImplementedError("both operands resolve to the same implementation, skipping comparison.")

    k_cache_ref, v_cache_ref = store_paged_kv_ref(
        key_states,
        value_states,
        k_cache_ref,
        v_cache_ref,
        block_table,
        cu_q_lens,
        context_kv_lens,
    )
    k_cache, v_cache = store_paged_kv(
        key_states,
        value_states,
        k_cache,
        v_cache,
        block_table,
        cu_q_lens,
        context_kv_lens,
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_decode", [False, True])
@auto_switch_platform()
@bypass_not_implemented
def test_store_paged_kv_scatter_kernel_called(monkeypatch, dtype, is_decode):
    if get_platform() != "npu":
        pytest.skip("npu_scatter_pa_kv_cache kernel call check requires NPU.")

    device = get_torch_device()
    batch_size, kv_heads, head_dim, block_size = 2, 2, 128, 128
    context_kv_lens = torch.tensor([0, 3], dtype=torch.int32, device=device)
    if is_decode:
        cu_q_lens = None
        total_tokens = batch_size
    else:
        q_lens = torch.tensor([4, 2], dtype=torch.int32, device=device)
        cu_q_lens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(q_lens, dim=0, dtype=torch.int32),
            ]
        )
        total_tokens = int(cu_q_lens[-1].item())

    key_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=dtype, device=device)
    value_states = torch.randn((total_tokens, kv_heads, head_dim), dtype=dtype, device=device)
    cache_shape = (4, kv_heads, block_size, head_dim)
    k_cache_ref = torch.zeros(cache_shape, dtype=dtype, device=device)
    v_cache_ref = torch.zeros(cache_shape, dtype=dtype, device=device)
    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()
    block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)

    calls = {"count": 0}
    original_scatter = torch_npu.npu_scatter_pa_kv_cache

    def _scatter_spy(*args, **kwargs):
        result = original_scatter(*args, **kwargs)
        calls["count"] += 1
        return result

    monkeypatch.setattr(torch_npu, "npu_scatter_pa_kv_cache", _scatter_spy)

    op_ref = MojoStorePagedKVCache._registry.get("torch")()
    op = MojoStorePagedKVCache()
    k_cache_ref, v_cache_ref = op_ref(
        key_states,
        value_states,
        k_cache_ref,
        v_cache_ref,
        block_table,
        cu_q_lens,
        context_kv_lens,
    )
    k_cache, v_cache = op(
        key_states,
        value_states,
        k_cache,
        v_cache,
        block_table,
        cu_q_lens,
        context_kv_lens,
    )

    assert calls["count"] == 1
    assert_close(k_cache, k_cache_ref)
    assert_close(v_cache, v_cache_ref)


@pytest.mark.parametrize("dtype", _available_float8_dtypes())
@auto_switch_platform()
@bypass_not_implemented
def test_store_paged_kv_mxfp8(dtype):
    if get_platform() != "npu":
        pytest.skip("MXFP8 paged KV cache scatter is a torch_npu-only scenario.")

    device = get_torch_device()
    batch_size, kv_heads, head_dim, block_size = 2, 2, 128, 128
    context_kv_lens = torch.tensor([0, 3], dtype=torch.int32, device=device)
    q_lens = torch.tensor([4, 2], dtype=torch.int32, device=device)
    cu_q_lens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(q_lens, dim=0, dtype=torch.int32),
        ]
    )
    total_tokens = int(cu_q_lens[-1].item())

    key_states = _rand_float8((total_tokens, kv_heads, head_dim), dtype, device)
    value_states = _rand_float8((total_tokens, kv_heads, head_dim), dtype, device)
    cache_shape = (4, kv_heads, block_size, head_dim)
    k_cache_ref = torch.zeros(cache_shape, dtype=dtype, device=device)
    v_cache_ref = torch.zeros(cache_shape, dtype=dtype, device=device)
    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()
    block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)

    op_ref = MojoStorePagedKVCache._registry.get("torch")()
    op = MojoStorePagedKVCache()

    k_cache_ref, v_cache_ref = op_ref(
        key_states,
        value_states,
        k_cache_ref,
        v_cache_ref,
        block_table,
        cu_q_lens,
        context_kv_lens,
    )
    k_cache, v_cache = op(
        key_states,
        value_states,
        k_cache,
        v_cache,
        block_table,
        cu_q_lens,
        context_kv_lens,
    )

    _assert_exact_float8_cache(k_cache, k_cache_ref)
    _assert_exact_float8_cache(v_cache, v_cache_ref)


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
