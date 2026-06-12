import pytest
import torch
import torch.nn as nn

from mojo_opset.experimental import MojoFusedNormRoPEQuantStore
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import get_torch_device

torch.manual_seed(42)

CONFIGS = [
    # (num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim)
    (8, 2, 32, 4, 128, 128),
    (8, 2, 32, 4, 128, 64),
    (16, 4, 48, 8, 96, 96),
    (4, 1, 16, 2, 128, 128),
    (8, 2, 32, 4, 64, 64),
    (32, 8, 64, 16, 128, 128),
    (4, 2, 8, 4, 96, 48),
]

SEQ_CONFIGS = [
    # (batch_size, q_lens_list, context_kv_lens_list)
    (1, [1], [0]),
    (1, [1], [15]),
    (1, [1], [127]),
    (1, [1], [1023]),
    (2, [1, 1], [0, 7]),
    (4, [1, 1, 1, 1], [0, 15, 31, 63]),
    (1, [32], [0]),
    (1, [64], [0]),
    (1, [128], [0]),
    (1, [256], [0]),
    (2, [16, 8], [5, 10]),
    (2, [64, 32], [0, 128]),
    (3, [1, 1, 1], [10, 20, 30]),
    (1, [512], [0]),
    (2, [128, 128], [64, 64]),
]

BLOCK_SIZE = 128


def _build_kv_case(batch_size, kv_heads, head_dim, block_size, context_kv_lens_val, q_lens_val, device):
    context_kv_lens = torch.tensor(context_kv_lens_val, dtype=torch.int32, device=device)
    q_lens = torch.tensor(q_lens_val, dtype=torch.int32, device=device)

    is_decode = all(q == 1 for q in q_lens_val)
    cu_q_lens = (
        torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(q_lens, dim=0, dtype=torch.int32),
        ])
        if not is_decode
        else None
    )

    total_tokens = int(q_lens.sum().item()) if not is_decode else batch_size

    max_kv_len = int(torch.clamp(context_kv_lens + q_lens, min=0).max().item())
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size + 2
    total_blocks_needed = sum(
        max(0, ckv + ql + block_size - 1) // block_size
        for ckv, ql in zip(context_kv_lens_val, q_lens_val)
    )
    total_phys_blocks = total_blocks_needed + 10

    cache_shape = (total_phys_blocks, kv_heads, block_size, head_dim)
    k_cache = torch.zeros(cache_shape, dtype=torch.int8, device=device)
    v_cache = torch.zeros(cache_shape, dtype=torch.int8, device=device)

    block_table = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    next_block = 0
    for b in range(batch_size):
        needed = max(0, context_kv_lens_val[b] + q_lens_val[b] + block_size - 1) // block_size
        if needed > 0:
            block_table[b, :needed] = torch.arange(next_block, next_block + needed, dtype=torch.int32, device=device)
        next_block += needed

    return {
        "total_tokens": total_tokens,
        "cu_q_lens": cu_q_lens,
        "context_kv_lens": context_kv_lens,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
    }


@pytest.mark.parametrize("num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim", CONFIGS)
@pytest.mark.parametrize("batch_size, q_lens_val, context_kv_lens_val", SEQ_CONFIGS)
@pytest.mark.parametrize("update_kv", [True, False])
@bypass_not_implemented
def test_fused_norm_rope_quant_store(
    num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k, head_dim, rope_dim,
    batch_size, q_lens_val, context_kv_lens_val,
    update_kv,
):
    """forward_diff_with: op vs torch reference backend."""
    torch.manual_seed(42)

    device = get_torch_device()

    op = MojoFusedNormRoPEQuantStore(
        num_heads_swa_q=num_heads_swa_q,
        num_heads_swa_k=num_heads_swa_k,
        num_heads_full_q=num_heads_full_q,
        num_heads_full_k=num_heads_full_k,
        head_dim=head_dim,
        norm_eps=1e-5,
        use_query_norm=True,
        use_key_norm=True,
        quant_dtype=torch.int8,
    ).to(device)

    op_ref = MojoFusedNormRoPEQuantStore._registry.get("torch")(
        num_heads_swa_q=num_heads_swa_q,
        num_heads_swa_k=num_heads_swa_k,
        num_heads_full_q=num_heads_full_q,
        num_heads_full_k=num_heads_full_k,
        head_dim=head_dim,
        norm_eps=1e-5,
        use_query_norm=True,
        use_key_norm=True,
        quant_dtype=torch.int8,
    ).to(device)

    for p in op_ref.parameters():
        nn.init.normal_(p, mean=1.0, std=0.1)
    op.load_state_dict(op_ref.state_dict())

    full_kv_case = _build_kv_case(batch_size, num_heads_full_k, head_dim, BLOCK_SIZE, context_kv_lens_val, q_lens_val, device)
    swa_kv_case = _build_kv_case(batch_size, num_heads_swa_k, head_dim, BLOCK_SIZE, context_kv_lens_val, q_lens_val, device)

    T = full_kv_case["total_tokens"]
    swa_query = torch.randn(T, num_heads_swa_q, head_dim, dtype=torch.bfloat16, device=device)
    swa_key = torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16, device=device)
    swa_value = torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16, device=device)
    full_query = torch.randn(T, num_heads_full_q, head_dim, dtype=torch.bfloat16, device=device)
    full_key = torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16, device=device)
    full_value = torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16, device=device)

    cos = torch.randn(T, rope_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(T, rope_dim, dtype=torch.bfloat16, device=device)

    op.forward_diff_with(
        op_ref,
        swa_query, swa_key, swa_value,
        full_query, full_key, full_value,
        cos, sin,
        full_kv_case["k_cache"].clone(), full_kv_case["v_cache"].clone(),
        swa_kv_case["k_cache"].clone(), swa_kv_case["v_cache"].clone(),
        full_kv_case["block_table"], full_kv_case["cu_q_lens"], full_kv_case["context_kv_lens"],
        swa_kv_case["block_table"], swa_kv_case["cu_q_lens"], swa_kv_case["context_kv_lens"],
        update_kv=update_kv,
        atol=1, rtol=0.05, ptol=0.999,
    )
