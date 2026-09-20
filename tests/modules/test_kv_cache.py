import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules as M
from mojo_opset.utils.kv_cache_metadata import build_paged_kv_chunk_metadata
from tests._checks import assert_close

KV_CASES = [
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
    (8, 2, 128, 512, [224, 542, 34, 41, 54, 57, 65, 0], [432, 84, 977, 93, 23, 89, 31, 555]),
    (8, 2, 128, 1024, [900, 1500, 34, 41, 54, 57, 65, 0], [700, 600, 977, 93, 23, 89, 31, 555]),
    (8, 2, 128, 2048, [1800, 2500, 34, 41, 54, 57, 65, 0], [900, 1200, 977, 93, 23, 89, 31, 555]),
    (8, 2, 128, 512, [772, 974, 3232, 43, 77, 7633, 888, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    (8, 2, 128, 1024, [1023, 1024, 3232, 43, 77, 7633, 888, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
    (8, 2, 128, 2048, [2047, 2048, 3232, 43, 77, 7633, 888, 1], [1, 1, 1, 1, 1, 1, 1, 1]),
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
]


def _build_store_paged_kv_case(batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val, *, device):
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
    max_kv_len = torch.clamp(context_kv_lens + q_lens, min=0).max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks_needed = sum(
        (
            max(0, context_kv_len + q_len + block_size - 1) // block_size
            for context_kv_len, q_len in zip(context_kv_lens_val, q_lens_val)
        )
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
                next_block, next_block + needed, dtype=torch.int32, device=device
            )
        next_block += needed
    chunk_metadata = build_paged_kv_chunk_metadata(block_table, cu_q_lens, context_kv_lens, block_size)
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


@pytest.mark.api("modules.StorePagedKVCache", ops=["store_paged_kv_cache"])
@pytest.mark.accuracy
@pytest.mark.parametrize("case", KV_CASES + [(6, 2, 128, 128, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0])])
@pytest.mark.parametrize("precompute", [False, True])
def test_store(accuracy_backend, case, precompute):
    impl, _, device = accuracy_backend
    if impl == "torch_npu":
        pytest.skip("Inherited master exclusion: torch_npu paged KV scatter causes CI coredump")
    data = _build_store_paged_kv_case(*case, device=device)
    metadata = (
        {"chunk_metadata": data["chunk_metadata"]}
        if precompute
        else {
            "block_table": data["block_table"],
            "cu_q_lens": data["cu_q_lens"],
            "context_kv_lens": data["context_kv_lens"],
        }
    )
    op = M.StorePagedKVCache(implementation=impl)
    inputs = (data["key_states"], data["value_states"])

    def run(k, v):
        kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
        actual = op(k, v, kc, vc, **metadata)
        assert actual[0] is kc and actual[1] is vc
        return actual

    actual = run(*inputs)
    expected = F.store_paged_kv_cache(
        *inputs, data["k_cache"].clone(), data["v_cache"].clone(), implementation="torch_reference", **metadata
    )
    for a, e in zip(actual, expected):
        assert_close(a, e, rtol=0, atol=0)
