import functools
import math
from typing import Optional

import pytest
import torch

from mojo_opset import MojoDecodeGQA
from mojo_opset import MojoDecodeMLA
from mojo_opset import MojoDecodeNSA
from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedDecodeMLA
from mojo_opset import MojoPagedDecodeNSA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoPagedPrefillMLA
from mojo_opset import MojoPagedPrefillNSA
from mojo_opset import MojoPrefillGQA
from mojo_opset import MojoPrefillMLA
from mojo_opset import MojoPrefillNSA
from mojo_opset import MojoSdpa
from mojo_opset import MojoPagedPrefillSWA
from mojo_opset import MojoPagedDecodeSWA
from mojo_opset import MojoSWA
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented


def generate_paged_decode_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype)

    seqlens = torch.randint(1, max_seq_len, (batch_size,), dtype=torch.int32)

    max_num_blocks_per_seq = (seqlens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(seqlens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.long)
    free_blocks = torch.randperm(num_total_blocks)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = seqlens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")

        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    return query, k_cache, v_cache, seqlens, block_tables


test_configs_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 8192, 128, torch.bfloat16, "M_BF16_LONG"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, seqlens, block_tables",
    [
        pytest.param(
            *generate_paged_decode_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
):
    from mojo_opset.utils.platform import get_platform
    if get_platform() == "npu":
        head_dim = query.shape[-1]
        if head_dim % 128 != 0:
            pytest.skip(f"NPU kernel npu_fused_infer_attention_score currently produces incorrect results for head_dim={head_dim} (not a multiple of 128)")

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    paged_decode_attn = MojoPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )
    paged_decode_attn_ref = MojoPagedDecodeGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    atol = 2e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 2e-2 if query.dtype != torch.float32 else 1e-6

    paged_decode_attn.forward_diff_with(
        paged_decode_attn_ref,
        query,
        k_cache,
        v_cache,
        seqlens,
        block_tables,
        softmax_scale=softmax_scale,
        atol=atol,
        rtol=rtol,
    )


def generate_paged_prefill_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_seqlens_kv = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0)])

    total_q_tokens = cu_seqlens_q[-1].item()
    total_kv_tokens = cu_seqlens_kv[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    k_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    v_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)

    max_num_blocks_per_seq = (kv_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(kv_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.long)
    free_blocks = torch.randperm(num_total_blocks)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        start_loc = cu_seqlens_kv[i].item()

        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

        k_seq = k_unpadded[start_loc : start_loc + seq_len]
        v_seq = v_unpadded[start_loc : start_loc + seq_len]
        for j in range(num_blocks_for_seq):
            physical_block_id = assigned_blocks[j]
            start_pos_in_seq = j * block_size
            tokens_in_block = min(block_size, seq_len - start_pos_in_seq)

            k_slice = k_seq[start_pos_in_seq : start_pos_in_seq + tokens_in_block].permute(1, 0, 2)
            v_slice = v_seq[start_pos_in_seq : start_pos_in_seq + tokens_in_block].permute(1, 0, 2)

            k_cache[physical_block_id, :, :tokens_in_block, :] = k_slice
            v_cache[physical_block_id, :, :tokens_in_block, :] = v_slice

    return query, k_cache, v_cache, cu_seqlens_q, block_tables, None if kv_cache_lens is None else kv_lens


test_configs_prefill = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 4096, 8192, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, block_tables, seqlens_kv",
    [
        pytest.param(
            *generate_paged_prefill_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    seqlens_kv: Optional[torch.Tensor],
):
    from mojo_opset.utils.platform import get_platform
    if get_platform() == "npu":
        head_dim = query.shape[-1]
        if head_dim % 128 != 0:
            pytest.skip(f"NPU kernel npu_fused_infer_attention_score currently produces incorrect results for head_dim={head_dim} (not a multiple of 128)")
        if seqlens_kv is not None:
            pytest.skip("NPU kernel npu_fused_infer_attention_score currently does not support TND layout with sparse_mode=3 (Page Attention), raising RuntimeError: call aclnnFusedInferAttentionScoreV3 failed.")

    paged_prefill_attn = MojoPagedPrefillGQA(
        is_causal=True,
        gqa_layout=gqa_layout
    )

    paged_prefill_attn_ref = MojoPagedPrefillGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    paged_prefill_attn.forward_diff_with(
        paged_prefill_attn_ref,
        query,
        k_cache,
        v_cache,
        cu_seqlens_q,
        block_tables,
        softmax_scale=softmax_scale,
        seqlens_kv=seqlens_kv,
        atol=2e-2 if query.dtype != torch.float32 else 1e-5,
        rtol=2e-2 if query.dtype != torch.float32 else 1e-6,
    )


@functools.lru_cache()
def generate_diffusion_attention_mask(
    seq_length: int,
    block_size: int,
) -> torch.Tensor:
    total_length = seq_length * 2
    attn_mask = torch.zeros(total_length, total_length, dtype=torch.int8)

    for i in range(total_length):
        for j in range(total_length):
            block_i = i // block_size
            block_j = j // block_size
            if block_i == block_j:
                attn_mask[i, j] = 1

            if j >= seq_length and i < seq_length and ((j - seq_length) // block_size) < block_i:
                attn_mask[i, j] = 1

            if i >= seq_length and j >= seq_length and block_j < block_i:
                attn_mask[i, j] = 1

    return attn_mask.to(torch.bool)


def generate_diffusion_attn_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
    block_size: int,
):
    query = torch.randn(bsz, q_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    key = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    value = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    blockwise_diffusion_attn_mask = generate_diffusion_attention_mask(seq_length, block_size)
    # blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, key, value, blockwise_diffusion_attn_mask, q_head_num != kv_head_num


@pytest.mark.parametrize(
    "bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size",
    [(1, 5, 1, 128, 2048, 32,)],
)
def test_sdpa(
    bsz,
    q_head_num,
    kv_head_num,
    head_dim,
    seq_length,
    block_size,
):
    query, key, value, blockwise_diffusion_attn_mask, enable_gqa = generate_diffusion_attn_test_data(
        bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size
    )
    diffusion_attn_ref = MojoSdpa._registry.get("torch")(
        scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    diffusion_attn = MojoSdpa(
        scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    diffusion_attn_ref.forward_diff_with(diffusion_attn, query, key, value, blockwise_diffusion_attn_mask)


# ===========================================================================
# MojoDecodeGQA (non-paged)
# ===========================================================================

@pytest.mark.parametrize(
    "B, Hq, Hkv, D, S",
    [(4, 16, 4, 128, 256), (2, 8, 1, 64, 512)],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@bypass_not_implemented
def test_decode_gqa(B, Hq, Hkv, D, S, gqa_layout):
    query = torch.randn(B, Hq, D, dtype=torch.bfloat16)
    key = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)
    value = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)
    seqlens = torch.randint(S // 2, S + 1, (B,), dtype=torch.int32)

    op = MojoDecodeGQA(gqa_layout=gqa_layout)
    op_ref = MojoDecodeGQA._registry.get("torch")(gqa_layout=gqa_layout)
    op.forward_diff_with(
        op_ref, query, key, value, seqlens,
        softmax_scale=1.0 / math.sqrt(D),
        atol=0, rtol=0,
    )


@pytest.mark.parametrize("window_size", [64, 128])
@bypass_not_implemented
def test_decode_gqa_sliding_window(window_size):
    B, Hq, Hkv, D, S = 2, 8, 2, 128, 256
    query = torch.randn(B, Hq, D, dtype=torch.bfloat16)
    key = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)
    value = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)
    seqlens = torch.full((B,), S, dtype=torch.int32)

    op = MojoDecodeGQA(gqa_layout="AABB", window_size=window_size)
    op_ref = MojoDecodeGQA._registry.get("torch")(gqa_layout="AABB", window_size=window_size)
    op.forward_diff_with(
        op_ref, query, key, value, seqlens,
        softmax_scale=1.0 / math.sqrt(D),
        atol=0, rtol=0,
    )


# ===========================================================================
# MojoDecodeMLA
# ===========================================================================

@pytest.mark.parametrize(
    "B, H, d_nope, d_rope, d_v, d_c, S",
    [(4, 16, 96, 32, 128, 64, 256)],
)
@bypass_not_implemented
def test_decode_mla(B, H, d_nope, d_rope, d_v, d_c, S):
    query = torch.randn(B, H, d_nope + d_rope, dtype=torch.bfloat16)
    compressed_kv = torch.randn(B, S, d_c, dtype=torch.bfloat16)
    k_pe = torch.randn(B, S, 1, d_rope, dtype=torch.bfloat16)
    seqlens = torch.randint(S // 2, S + 1, (B,), dtype=torch.int32)

    op = MojoDecodeMLA(H, d_nope, d_rope, d_v, d_c)
    op_ref = MojoDecodeMLA._registry.get("torch")(H, d_nope, d_rope, d_v, d_c)
    with torch.no_grad():
        w = torch.randn_like(op.kv_b_proj)
        op.kv_b_proj.copy_(w)
        op_ref.kv_b_proj.copy_(w)

    op.forward_diff_with(
        op_ref, query, compressed_kv, k_pe, seqlens,
        atol=1e-2, rtol=1e-2,
    )


# ===========================================================================
# MojoPrefillMLA
# ===========================================================================

@pytest.mark.parametrize(
    "H, d_nope, d_rope, d_v, d_c",
    [(8, 64, 32, 64, 32)],
)
@bypass_not_implemented
def test_prefill_mla(H, d_nope, d_rope, d_v, d_c):
    seqlens = torch.tensor([32, 48], dtype=torch.int32)
    cu = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens.cumsum(0)])
    T = cu[-1].item()

    query = torch.randn(T, H, d_nope + d_rope, dtype=torch.bfloat16)
    compressed_kv = torch.randn(T, d_c, dtype=torch.bfloat16)
    k_pe = torch.randn(T, 1, d_rope, dtype=torch.bfloat16)

    op = MojoPrefillMLA(H, d_nope, d_rope, d_v, d_c, is_causal=True)
    op_ref = MojoPrefillMLA._registry.get("torch")(H, d_nope, d_rope, d_v, d_c, is_causal=True)
    with torch.no_grad():
        w = torch.randn_like(op.kv_b_proj)
        op.kv_b_proj.copy_(w)
        op_ref.kv_b_proj.copy_(w)

    op.forward_diff_with(
        op_ref, query, compressed_kv, k_pe, cu,
        atol=1e-2, rtol=1e-2,
    )


# ===========================================================================
# MojoDecodeNSA
# ===========================================================================

@pytest.mark.parametrize(
    "B, H, D, S",
    [(2, 8, 64, 256)],
)
@bypass_not_implemented
def test_decode_nsa(B, H, D, S):
    query = torch.randn(B, H, D, dtype=torch.bfloat16)
    key = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    value = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    seqlens = torch.full((B,), S, dtype=torch.int32)

    op = MojoDecodeNSA(H, D, compress_ratio=4, num_selected_blocks=4, window_size=64)
    op_ref = MojoDecodeNSA._registry.get("torch")(H, D, compress_ratio=4, num_selected_blocks=4, window_size=64)
    with torch.no_grad():
        g = torch.randn_like(op.gate_proj)
        op.gate_proj.copy_(g)
        op_ref.gate_proj.copy_(g)

    op.forward_diff_with(
        op_ref, query, key, value, seqlens,
        atol=1e-2, rtol=1e-2,
    )


# ===========================================================================
# MojoPrefillGQA (non-paged)
# ===========================================================================

@pytest.mark.parametrize(
    "B, Hq, Hkv, D, S",
    [(2, 16, 4, 128, 64), (1, 8, 1, 64, 128)],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@auto_switch_platform()
@bypass_not_implemented
def test_prefill_gqa(B, Hq, Hkv, D, S, gqa_layout):
    """Non-paged prefill GQA — query/key/value are batched 4-D tensors."""
    from mojo_opset.utils.platform import get_platform
    if get_platform() == "npu" and D % 128 != 0:
        pytest.skip(f"NPU kernel requires head_dim % 128 == 0, got {D}")

    query = torch.randn(B, Hq, S, D, dtype=torch.bfloat16)
    key = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)
    value = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16)
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32)

    op = MojoPrefillGQA(is_causal=True, gqa_layout=gqa_layout)
    op_ref = MojoPrefillGQA._registry.get("torch")(is_causal=True, gqa_layout=gqa_layout)
    op.forward_diff_with(
        op_ref, query, key, value, cu,
        softmax_scale=1.0 / math.sqrt(D),
        atol=2e-2, rtol=2e-2,
    )


# ===========================================================================
# MojoPagedDecodeMLA
# ===========================================================================

def _generate_paged_mla_decode_data(batch_size, num_heads, d_nope, d_rope, d_v,
                                     kv_lora_rank, max_seq_len, block_size, dtype):
    query = torch.randn(batch_size, num_heads, d_nope + d_rope, dtype=dtype)
    seqlens = torch.randint(max_seq_len // 2, max_seq_len, (batch_size,), dtype=torch.int32).clamp(min=1)

    max_nb = (seqlens.max().item() + block_size - 1) // block_size
    total_blocks = int(torch.div(seqlens + block_size - 1, block_size, rounding_mode="floor").sum().item()) + 10

    ckv_cache = torch.randn(total_blocks, 1, block_size, kv_lora_rank, dtype=dtype)
    kpe_cache = torch.randn(total_blocks, 1, block_size, d_rope, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_nb, dtype=torch.long)
    free = torch.randperm(total_blocks)
    off = 0
    for i in range(batch_size):
        n = (seqlens[i].item() + block_size - 1) // block_size
        block_tables[i, :n] = free[off:off + n]
        off += n

    return query, ckv_cache, kpe_cache, seqlens, block_tables


@pytest.mark.parametrize(
    "B, H, d_nope, d_rope, d_v, d_c, S, blk",
    [
        (4, 16, 96, 32, 128, 64, 256, 64),
        (2, 8, 64, 32, 64, 32, 128, 32),
    ],
)
@bypass_not_implemented
def test_paged_decode_mla(B, H, d_nope, d_rope, d_v, d_c, S, blk):
    query, ckv_cache, kpe_cache, seqlens, bt = _generate_paged_mla_decode_data(
        B, H, d_nope, d_rope, d_v, d_c, S, blk, torch.bfloat16,
    )
    op = MojoPagedDecodeMLA(H, d_nope, d_rope, d_v, d_c)
    op_ref = MojoPagedDecodeMLA._registry.get("torch")(H, d_nope, d_rope, d_v, d_c)
    with torch.no_grad():
        w = torch.randn_like(op.kv_b_proj)
        op.kv_b_proj.copy_(w)
        op_ref.kv_b_proj.copy_(w)

    op.forward_diff_with(
        op_ref, query, ckv_cache, kpe_cache, seqlens, bt,
        atol=1e-2, rtol=1e-2,
    )


# ===========================================================================
# MojoPagedPrefillMLA
# ===========================================================================

def _generate_paged_mla_prefill_data(batch_size, num_heads, d_nope, d_rope, d_v,
                                      kv_lora_rank, max_q_len, block_size, dtype):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32).clamp(min=1)
    cu = torch.cat([torch.tensor([0], dtype=torch.int32), q_lens.cumsum(0)])
    T = cu[-1].item()

    query = torch.randn(T, num_heads, d_nope + d_rope, dtype=dtype)

    kv_lens = q_lens
    max_nb = (kv_lens.max().item() + block_size - 1) // block_size
    total_blocks = int(torch.div(kv_lens + block_size - 1, block_size, rounding_mode="floor").sum().item()) + 10

    ckv_cache = torch.zeros(total_blocks, 1, block_size, kv_lora_rank, dtype=dtype)
    kpe_cache = torch.zeros(total_blocks, 1, block_size, d_rope, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_nb, dtype=torch.long)
    free = torch.randperm(total_blocks)
    off = 0

    for i in range(batch_size):
        kl = kv_lens[i].item()
        nb = (kl + block_size - 1) // block_size
        blocks = free[off:off + nb]
        block_tables[i, :nb] = blocks
        off += nb

        ckv_data = torch.randn(kl, kv_lora_rank, dtype=dtype)
        kpe_data = torch.randn(kl, d_rope, dtype=dtype)
        for j in range(nb):
            bid = blocks[j].item()
            s = j * block_size
            e = min(s + block_size, kl)
            ckv_cache[bid, 0, : e - s] = ckv_data[s:e]
            kpe_cache[bid, 0, : e - s] = kpe_data[s:e]

    return query, ckv_cache, kpe_cache, cu, block_tables


@pytest.mark.parametrize(
    "B, H, d_nope, d_rope, d_v, d_c, max_q, blk",
    [
        (2, 8, 64, 32, 64, 32, 48, 32),
    ],
)
@bypass_not_implemented
def test_paged_prefill_mla(B, H, d_nope, d_rope, d_v, d_c, max_q, blk):
    query, ckv_cache, kpe_cache, cu, bt = _generate_paged_mla_prefill_data(
        B, H, d_nope, d_rope, d_v, d_c, max_q, blk, torch.bfloat16,
    )
    op = MojoPagedPrefillMLA(H, d_nope, d_rope, d_v, d_c, is_causal=True)
    op_ref = MojoPagedPrefillMLA._registry.get("torch")(H, d_nope, d_rope, d_v, d_c, is_causal=True)
    with torch.no_grad():
        w = torch.randn_like(op.kv_b_proj)
        op.kv_b_proj.copy_(w)
        op_ref.kv_b_proj.copy_(w)

    op.forward_diff_with(
        op_ref, query, ckv_cache, kpe_cache, cu, bt,
        atol=1e-2, rtol=1e-2,
    )


# ===========================================================================
# MojoPagedDecodeNSA
# ===========================================================================

@pytest.mark.parametrize(
    "B, H, D, S, blk",
    [(2, 8, 64, 256, 64)],
)
@bypass_not_implemented
def test_paged_decode_nsa(B, H, D, S, blk):
    query, k_cache, v_cache, seqlens, bt = generate_paged_decode_data(
        batch_size=B, num_q_heads=H, num_kv_heads=H,
        head_dim=D, max_seq_len=S, block_size=blk, dtype=torch.bfloat16,
    )
    cr, nsb, ws = 4, 4, 64
    op = MojoPagedDecodeNSA(H, D, compress_ratio=cr, num_selected_blocks=nsb, window_size=ws)
    op_ref = MojoPagedDecodeNSA._registry.get("torch")(H, D, compress_ratio=cr, num_selected_blocks=nsb, window_size=ws)
    with torch.no_grad():
        g = torch.randn_like(op.gate_proj)
        op.gate_proj.copy_(g)
        op_ref.gate_proj.copy_(g)

    op.forward_diff_with(
        op_ref, query, k_cache, v_cache, seqlens, bt,
        atol=1e-2, rtol=1e-2,
    )


# ===========================================================================
# MojoPrefillNSA (non-paged) — small seqlens to keep runtime manageable
# ===========================================================================

@pytest.mark.parametrize(
    "H, D",
    [(4, 64)],
)
@bypass_not_implemented
def test_prefill_nsa(H, D):
    seqlens = torch.tensor([32, 24], dtype=torch.int32)
    cu = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens.cumsum(0)])
    T = cu[-1].item()

    query = torch.randn(T, H, D, dtype=torch.bfloat16)
    key = torch.randn(T, H, D, dtype=torch.bfloat16)
    value = torch.randn(T, H, D, dtype=torch.bfloat16)

    cr, nsb, ws = 4, 2, 16
    op = MojoPrefillNSA(H, D, compress_ratio=cr, num_selected_blocks=nsb, window_size=ws, is_causal=True)
    op_ref = MojoPrefillNSA._registry.get("torch")(H, D, compress_ratio=cr, num_selected_blocks=nsb, window_size=ws, is_causal=True)
    with torch.no_grad():
        g = torch.randn_like(op.gate_proj)
        op.gate_proj.copy_(g)
        op_ref.gate_proj.copy_(g)

    op.forward_diff_with(
        op_ref, query, key, value, cu,
        atol=1e-2, rtol=1e-2,
    )


# ===========================================================================
# MojoPagedPrefillNSA — small seqlens to keep runtime manageable
# ===========================================================================

@pytest.mark.parametrize(
    "H, D, blk",
    [(4, 64, 32)],
)
@bypass_not_implemented
def test_paged_prefill_nsa(H, D, blk):
    B = 2
    query, k_cache, v_cache, cu, bt, _ = generate_paged_prefill_data(
        batch_size=B, num_q_heads=H, num_kv_heads=H,
        head_dim=D, max_q_len=32, max_kv_computed_len=0,
        block_size=blk, dtype=torch.bfloat16,
    )
    cr, nsb, ws = 4, 2, 16
    op = MojoPagedPrefillNSA(H, D, compress_ratio=cr, num_selected_blocks=nsb, window_size=ws, is_causal=True)
    op_ref = MojoPagedPrefillNSA._registry.get("torch")(H, D, compress_ratio=cr, num_selected_blocks=nsb, window_size=ws, is_causal=True)
    with torch.no_grad():
        g = torch.randn_like(op.gate_proj)
        op.gate_proj.copy_(g)
        op_ref.gate_proj.copy_(g)

    op.forward_diff_with(
        op_ref, query, k_cache, v_cache, cu, bt,
        atol=1e-2, rtol=1e-2,
    )



# ===========================================================================
# MojoSWA
# ===========================================================================


test_configs_swa_prefill = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 2048, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 256, 1024, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, block_tables, seqlens_kv",
    [
        pytest.param(
            *generate_paged_prefill_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_swa_prefill
    ],
)
@pytest.mark.parametrize("gqa_layout, global_window, local_window", [
    ("ABAB", 4, 255), 
    ("AABB", 4, 1023),
])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_swa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    seqlens_kv: Optional[torch.Tensor],
    global_window: int,
    local_window: int,
):

    paged_prefill_swa = MojoPagedPrefillSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        local_window_size=local_window,
        global_window_size=global_window,
    )

    paged_prefill_swa_ref = MojoPagedPrefillSWA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        local_window_size=local_window,
        global_window_size=global_window,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    paged_prefill_swa.forward_diff_with(
        paged_prefill_swa_ref,
        query,
        k_cache,
        v_cache,
        cu_seqlens_q,
        block_tables,
        softmax_scale=softmax_scale,
        seqlens_kv=seqlens_kv,
        atol=2e-2 if query.dtype != torch.float32 else 1e-5,
        rtol=2e-2 if query.dtype != torch.float32 else 1e-6,
    )


test_configs_swa_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 2048, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 4096, 128, torch.bfloat16, "M_BF16_LONG"),
]

@pytest.mark.parametrize(
    "query, k_cache, v_cache, seqlens, block_tables",
    [
        pytest.param(
            *generate_paged_decode_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_swa_decode
    ],
)
@pytest.mark.parametrize("gqa_layout, global_window, local_window", [
    ("ABAB", 4, 255), 
    ("AABB", 4, 1023),
])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_swa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    global_window: int,
    local_window: int,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    paged_decode_swa = MojoPagedDecodeSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )
    paged_decode_swa_ref = MojoPagedDecodeSWA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    atol = 2e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 2e-2 if query.dtype != torch.float32 else 1e-6

    paged_decode_swa.forward_diff_with(
        paged_decode_swa_ref,
        query,
        k_cache,
        v_cache,
        seqlens,
        block_tables,
        softmax_scale=softmax_scale,
        atol=atol,
        rtol=rtol,
    )


def generate_sdpa_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_seqlens_kv = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0)])

    total_q_tokens = cu_seqlens_q[-1].item()
    total_kv_tokens = cu_seqlens_kv[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    key = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)

    
    return query, key, value, cu_seqlens_q, cu_seqlens_kv


test_configs_swa_infer = [
    (2, 16, 4, 128, 1024, 0, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 1024, 2048, torch.bfloat16, "M_BF16_WITH_CACHE"),
]

@pytest.mark.parametrize(
    "query, key, value, cu_seqlens_q, cu_seqlens_kv",
    [
        pytest.param(
            *generate_sdpa_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, dtype, ID in test_configs_swa_infer
    ],
)
@pytest.mark.parametrize("gqa_layout, global_window, local_window", [
    ("ABAB", 4, 255), 
    ("AABB", 4, 1023),
])
@auto_switch_platform()
@bypass_not_implemented
def test_swa_infer(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    gqa_layout: str,
    global_window: int,
    local_window: int,
):
    swa = MojoSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        local_window_size=local_window,
        global_window_size=global_window,
    )

    swa_ref = MojoSWA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        local_window_size=local_window,
        global_window_size=global_window,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    swa.forward_diff_with(
        swa_ref,
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_kv,
        softmax_scale=softmax_scale,
        atol=2e-2 if query.dtype != torch.float32 else 1e-5,
        rtol=2e-2 if query.dtype != torch.float32 else 1e-6,
    )
