import pytest
import torch

from mojo_opset import MojoStorePagedKVCache
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform


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
    if get_platform() =="npu":
        from  mojo_opset.backends.torch_npu.operators.kv_cache import TorchNpuStorePagedKVCache
        if isinstance(store_paged_kv,TorchNpuStorePagedKVCache):
            perf(  
            lambda: store_paged_kv(
                key_states,
                value_states,
                k_cache,
                v_cache,
                block_table,
                cu_q_lens,
                context_kv_lens,
                chunk_metadata=None,)
           )
        else:
            perf(  
                    lambda: store_paged_kv(
                        key_states,
                        value_states,
                        k_cache,
                        v_cache,
                        chunk_metadata=chunk_metadata,
                    )
            )
    else:
        perf(  
        lambda: store_paged_kv(
            key_states,
            value_states,
            k_cache,
            v_cache,
            chunk_metadata=chunk_metadata,
        )
    )
        
    
    

@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_store_paged_kv_bucket_padded_varlen():
    device = get_torch_device()

    real_batch_size = 4
    bucket_batch_size = 6
    token_bucket_size = 8
    kv_heads = 2
    head_dim = 128
    block_size = 8

    cu_q_lens = torch.tensor([0, 1, 2, 3, 4, 4, 4], dtype=torch.int32, device=device)
    context_kv_lens = torch.tensor([0, 2, 7, 9, -1, -1], dtype=torch.int32, device=device)

    key_states = torch.randn((token_bucket_size, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
    value_states = torch.randn((token_bucket_size, kv_heads, head_dim), dtype=torch.bfloat16, device=device)

    max_kv_len = (context_kv_lens[:real_batch_size] + 1).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 1
    total_phys_blocks = bucket_batch_size * max_blocks_per_seq + 4

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)
    v_cache_ref = torch.zeros(cache_shape, dtype=torch.bfloat16, device=device)

    k_cache = k_cache_ref.clone()
    v_cache = v_cache_ref.clone()

    block_table = torch.full((bucket_batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    next_block = 0
    for batch_id in range(real_batch_size):
        needed = (context_kv_lens[batch_id].item() + 1 + block_size - 1) // block_size
        ids = torch.arange(next_block, next_block + needed, device=device, dtype=torch.int32)
        block_table[batch_id, :needed] = ids
        next_block += needed

    store_paged_kv = MojoStorePagedKVCache()

    perf(  # noqa: F821
        lambda: store_paged_kv(
            key_states,
            value_states,
            k_cache,
            v_cache,
            block_table,
            cu_q_lens,
            context_kv_lens,
        )
    )
