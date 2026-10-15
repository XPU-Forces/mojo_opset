import pytest
import torch
import torch.nn as nn

from mojo_opset.experimental import MojoFusedNormRoPEFp8KVStore
from mojo_opset.experimental import MojoFusedNormRoPEQuantStore
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import get_torch_device
from mojo_opset.tests.utils import requires_platform_backend

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


# ---------------------------------------------------------------------------
# fp8 (e4m3) KV store, at the prefill shapes the op is deployed at
# ---------------------------------------------------------------------------

# Lifted from prefill.ag_rs.all_cases.workloads.jsonl, the rows whose backend_op
# is MojoFusedNormRoPEQuantStoreOp: four head groups, each at three batch sizes
# and three KV lengths. GQA_* in the workload is the full-attention stream,
# SWA_* the sliding-window one; every row has GQA_HC = SWA_HC = 128, Q_LEN =
# 8192 and NO_KV = 0.
FP8_HEAD_CONFIGS = [
    # (num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k)
    (6, 1, 2, 1),
    (4, 1, 4, 1),
    (12, 2, 4, 1),
    (8, 2, 8, 2),
]

FP8_SEQ_CONFIGS = [
    # (batch_size, q_len, kv_len)
    (batch_size, 8192, kv_len)
    for batch_size in (1, 2, 4)
    for kv_len in (8192, 24576, 57344)
]

FP8_HEAD_DIM = 128


def _fp8_kernel_available():
    try:
        from ixformer import functions as ixf_f
    except ImportError:
        return False
    return hasattr(ixf_f, "rms_norm_qk_rotary_embedding_fp8_kv_store")


def _build_fp8_prefill_case(batch_size, q_len, kv_len, kv_heads, head_dim, block_size, device):
    """Paged fp8 cache and store metadata for a batch of equal-length prefills."""
    context_len = max(0, kv_len - q_len)
    context_kv_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
    cu_q_lens = torch.arange(0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device)

    blocks_per_seq = (context_len + q_len + block_size - 1) // block_size
    block_table = torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device=device
    ).reshape(batch_size, blocks_per_seq)

    cache_shape = (batch_size * blocks_per_seq, kv_heads, block_size, head_dim)
    return {
        "cu_q_lens": cu_q_lens,
        "context_kv_lens": context_kv_lens,
        "block_table": block_table,
        "k_cache": torch.zeros(cache_shape, dtype=torch.float8_e4m3fn, device=device),
        "v_cache": torch.zeros(cache_shape, dtype=torch.float8_e4m3fn, device=device),
    }


@requires_platform_backend(
    platforms="ilu", backends="ixformer", reason="fp8 KV store is an ixformer kernel"
)
@pytest.mark.parametrize(
    "num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k", FP8_HEAD_CONFIGS
)
@pytest.mark.parametrize("batch_size, q_len, kv_len", FP8_SEQ_CONFIGS)
@bypass_not_implemented
def test_fused_norm_rope_fp8_kv_store(
    num_heads_swa_q, num_heads_swa_k, num_heads_full_q, num_heads_full_k,
    batch_size, q_len, kv_len,
):
    """op vs a direct call of the kernel it wraps, per attention stream.

    The reference is the kernel rather than this operator's torch backend:
    MojoStaticQuant rounds to an integer before casting, which is what int8
    wants and not what fp8 does, so the generated torch backend does not model
    the fp8 cache. What is checked here is the operator's own work — splitting
    qk_norm into four per-stream weights, handing the kernel the per-(batch,
    head) scale layout it demands, and running both streams against their own
    cache and block table. The kernel's numerics are covered by ixformer's
    test_rms_norm_qk_rotary_embedding_fp8_kv_store.
    """
    if not _fp8_kernel_available():
        pytest.skip("rms_norm_qk_rotary_embedding_fp8_kv_store is not in this ixformer build")

    from ixformer import functions as ixf_f

    torch.manual_seed(42)
    device = get_torch_device()
    head_dim = FP8_HEAD_DIM
    block_size = BLOCK_SIZE

    op = MojoFusedNormRoPEFp8KVStore(
        num_heads_swa_q=num_heads_swa_q,
        num_heads_swa_k=num_heads_swa_k,
        num_heads_full_q=num_heads_full_q,
        num_heads_full_k=num_heads_full_k,
        head_dim=head_dim,
        norm_eps=1e-5,
        use_query_norm=True,
        use_key_norm=True,
        batch_size=batch_size,
    ).to(device)

    # Positive scales, and load through load_state_dict so the backend's
    # post-hook casts qk_norm to the input dtype the kernel requires.
    op.load_state_dict({k: torch.rand_like(v) * 0.2 + 0.9 for k, v in op.state_dict().items()})

    full_case = _build_fp8_prefill_case(
        batch_size, q_len, kv_len, num_heads_full_k, head_dim, block_size, device
    )
    swa_case = _build_fp8_prefill_case(
        batch_size, q_len, kv_len, num_heads_swa_k, head_dim, block_size, device
    )

    T = batch_size * q_len
    swa_query = torch.randn(T, num_heads_swa_q, head_dim, dtype=torch.bfloat16, device=device)
    swa_key = torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16, device=device)
    swa_value = torch.randn(T, num_heads_swa_k, head_dim, dtype=torch.bfloat16, device=device)
    full_query = torch.randn(T, num_heads_full_q, head_dim, dtype=torch.bfloat16, device=device)
    full_key = torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16, device=device)
    full_value = torch.randn(T, num_heads_full_k, head_dim, dtype=torch.bfloat16, device=device)

    cos = torch.randn(T, head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.randn(T, head_dim, dtype=torch.bfloat16, device=device)

    op_full_k_cache = full_case["k_cache"].clone()
    op_full_v_cache = full_case["v_cache"].clone()
    op_swa_k_cache = swa_case["k_cache"].clone()
    op_swa_v_cache = swa_case["v_cache"].clone()

    (
        swa_q_out, full_q_out,
        full_key_q, full_k_scale,
        swa_key_q, swa_k_scale,
        full_val_q, full_v_scale,
        swa_val_q, swa_v_scale,
    ) = op(
        swa_query, swa_key, swa_value,
        full_query, full_key, full_value,
        cos, sin,
        op_full_k_cache, op_full_v_cache,
        op_swa_k_cache, op_swa_v_cache,
        full_case["block_table"], full_case["cu_q_lens"], full_case["context_kv_lens"],
        swa_case["block_table"], swa_case["cu_q_lens"], swa_case["context_kv_lens"],
        update_kv=True,
    )

    # The kernel leaves the quantized K/V in the paged cache and nowhere else.
    assert (full_key_q, swa_key_q, full_val_q, swa_val_q) == (None, None, None, None)
    # Per-(batch, head) scales, handed to the kernel exactly as the parameters
    # hold them; a per-channel scale is what the kernel rejects.
    for scale, kv_heads in (
        (full_k_scale, num_heads_full_k), (full_v_scale, num_heads_full_k),
        (swa_k_scale, num_heads_swa_k), (swa_v_scale, num_heads_swa_k),
    ):
        assert scale.shape == (batch_size, kv_heads)
        assert scale.is_contiguous()

    w = op.qk_norm.weight
    eps = op.qk_norm.variance_epsilon

    def run_reference(query, key, value, weight_q, weight_k, scale_k, scale_v, case):
        k_cache = case["k_cache"].clone()
        v_cache = case["v_cache"].clone()
        output_q = ixf_f.rms_norm_qk_rotary_embedding_fp8_kv_store(
            cos, sin,
            query, key, value,
            weight_q, weight_k,
            head_dim,
            scale_k, scale_v,
            k_cache, v_cache,
            case["block_table"], case["cu_q_lens"], case["context_kv_lens"],
            eps=eps, is_neox_style=True, rotary_dim=head_dim,
        )
        return output_q, k_cache, v_cache

    ref_swa_q, ref_swa_k_cache, ref_swa_v_cache = run_reference(
        swa_query, swa_key, swa_value, w[0], w[1],
        op.swa_k_quantize.scale, op.swa_v_quantize.scale, swa_case,
    )
    ref_full_q, ref_full_k_cache, ref_full_v_cache = run_reference(
        full_query, full_key, full_value, w[2], w[3],
        op.full_k_quantize.scale, op.full_v_quantize.scale, full_case,
    )

    torch.testing.assert_close(swa_q_out, ref_swa_q, atol=0, rtol=0)
    torch.testing.assert_close(full_q_out, ref_full_q, atol=0, rtol=0)
    for got, want in (
        (op_swa_k_cache, ref_swa_k_cache), (op_swa_v_cache, ref_swa_v_cache),
        (op_full_k_cache, ref_full_k_cache), (op_full_v_cache, ref_full_v_cache),
    ):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
