import pytest
import torch

from mojo_opset.tests.utils import bypass_not_implemented

from mojo_opset import MojoApplyRoPE
from mojo_opset import MojoGridRoPE
from mojo_opset import MojoMRoPE
from mojo_opset import MojoRelativeEmbedding
from mojo_opset import MojoRotaryEmbedding
from mojo_opset.utils.platform import get_torch_device

torch.random.manual_seed(42)


def compute_cos_sin_cache(head_dim, rotary_dim, max_position, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim // 2, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_position, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = freqs.repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()



def prepare_mrope_test_inputs(num_tokens, n_qh, n_kh, head_dim, mrope_section, device, dtype=torch.float32):
    rotary_dim = sum(mrope_section) * 2
    
    positions = torch.randint(0, 1000, (3, num_tokens), device=device, dtype=torch.long)
    cos_cache, sin_cache = compute_cos_sin_cache(head_dim, rotary_dim, 4000, base=10000.0)
    
    half_rotary_dim = rotary_dim // 2
    cos_3d = torch.zeros(3, num_tokens, half_rotary_dim, device=device, dtype=torch.float32)
    sin_3d = torch.zeros(3, num_tokens, half_rotary_dim, device=device, dtype=torch.float32)
    
    for dim_idx in range(3):
        pos = positions[dim_idx]
        cos_3d[dim_idx] = cos_cache[pos][:, :half_rotary_dim]
        sin_3d[dim_idx] = sin_cache[pos][:, :half_rotary_dim]
    
    query = torch.randn(num_tokens, n_qh * head_dim, device=device, dtype=dtype)
    key = torch.randn(num_tokens, n_kh * head_dim, device=device, dtype=dtype)
    
    return query, key, cos_3d, sin_3d, mrope_section


@pytest.mark.parametrize("bs", [1, 6])
@pytest.mark.parametrize("seqlen", [2048])
@pytest.mark.parametrize(
    "rope_dim", [32, 48, 64, 88, 96, 128],
)
@pytest.mark.parametrize("mode", ["padding_prefill", "varlen_prefill", "decode"])
@bypass_not_implemented
def test_rotary_embedding(bs, seqlen, rope_dim, mode):
    """Test MojoRotaryEmbedding (position embedding extraction) independently."""
    device = get_torch_device()
    max_seq_len = 32768
    hidden_size = 256

    rot_pos_emb_ref_nocache = MojoRotaryEmbedding._registry.get("torch")(rope_theta=10000.0, rope_dim=rope_dim).to(device)
    rot_pos_emb_ref = MojoRotaryEmbedding._registry.get("torch")(rope_theta=10000.0, rope_dim=rope_dim, init_max_length=max_seq_len).to(device)
    rot_pos_emb = MojoRotaryEmbedding(rope_theta=10000.0, rope_dim=rope_dim, init_max_length=max_seq_len).to(device)

    if mode == "padding_prefill":
        x = torch.randn(bs, seqlen, hidden_size, device=device, dtype=torch.float32)
        torch.testing.assert_close(
            rot_pos_emb_ref(x), 
            rot_pos_emb_ref_nocache(x), 
            atol=1e-5, 
            rtol=1e-5,
        )

        rot_pos_emb.forward_diff_with(
            rot_pos_emb_ref,
            x,
            atol=1e-5,
            rtol=1e-5,
        )
    elif mode == "decode":
        x = torch.randn(bs, hidden_size, device=device, dtype=torch.float32)
        position_ids = torch.randint(0, max_seq_len, (bs,), dtype=torch.int32, device=device)
        torch.testing.assert_close(
            rot_pos_emb_ref(x, position_ids=position_ids), 
            rot_pos_emb_ref_nocache(x, position_ids=position_ids), 
            atol=1e-5, 
            rtol=1e-5,
        )

        rot_pos_emb.forward_diff_with(
            rot_pos_emb_ref,
            x,
            position_ids=position_ids,
            atol=1e-5,
            rtol=1e-5,
        )
    else:
        seq_lens = torch.randint((seqlen+1) // 2, seqlen + 1, (bs,), device=device, dtype=torch.int32)
        cu_seqlens = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
        kv_lens = torch.randint(0, max_seq_len - seqlen, (bs,), device=device, dtype=torch.int32) + seq_lens
        x = torch.randn(cu_seqlens[-1].item(), hidden_size, device=device, dtype=torch.float32)

        torch.testing.assert_close(
            rot_pos_emb_ref(x, cu_q_lens=cu_seqlens, total_seq_lens=kv_lens),
            rot_pos_emb(x, cu_q_lens=cu_seqlens, total_seq_lens=kv_lens),
            atol=1e-5,
            rtol=1e-5,
        )
        rot_pos_emb.forward_diff_with(
            rot_pos_emb_ref,
            x,
            cu_q_lens=cu_seqlens,
            total_seq_lens=kv_lens,
            atol=1e-5,
            rtol=1e-5,
    )


@pytest.mark.parametrize(
    "bs, seqlen",
    [
        (1, 124),
        (6, 555),
        (2, 2048),
    ],
)
@pytest.mark.parametrize(
    "dtype, q_heads, k_heads, head_first, head_dim, rope_percentage", 
    [
        (torch.float16, 32, 8, True, 96, 1.0),
        (torch.bfloat16, 8, 2, False, 96, 0.3333333333333333333333),
        (torch.float16, 16, 8, True, 128, 1.0),
        (torch.bfloat16, 64, 8, False, 88, 1.0),
        (torch.float16, 64, 4, True, 128, 0.375),
    ],
)
@pytest.mark.parametrize("mode", ["padding_prefill_pos2d", "padding_prefill_pos3d", "varlen_prefill", "decode"])
@bypass_not_implemented
def test_apply_rope(bs, seqlen, q_heads, k_heads, head_first, head_dim, rope_percentage, mode, dtype):
    """Test MojoApplyRoPE (apply rotary position embedding) with pre-extracted cos/sin."""
    device = get_torch_device()
    max_seq_len = 32768

    rope_dim = int(head_dim * rope_percentage)
    hidden_size = q_heads * head_dim

    rot_pos_emb = MojoRotaryEmbedding._registry.get("torch")(rope_theta=10000.0, rope_dim=rope_dim, init_max_length=max_seq_len).to(device)

    if mode == "padding_prefill_pos3d":
        offsets = torch.randint(0, max_seq_len - seqlen - 1, (bs,), device=device, dtype=torch.int32)
        position_ids = torch.arange(seqlen, dtype=torch.int32, device=device)
        position_ids = position_ids[None, :] + offsets[:, None]
        x = torch.randn(bs, seqlen, hidden_size, device=device, dtype=dtype)
        cos, sin = rot_pos_emb(x, position_ids=position_ids)
        q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype)
        if head_first:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

    elif mode == "padding_prefill_pos2d":
        x = torch.randn(bs, seqlen, hidden_size, device=device, dtype=dtype)
        cos, sin = rot_pos_emb(x)
        q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype)
        if head_first:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

    elif mode == "varlen_prefill":
        seq_lens = torch.randint(1, seqlen + 1, (bs,), device=device, dtype=torch.int32)
        total_seq_len = seq_lens.sum().item()
        cu_seqlens = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)

        x = torch.randn(total_seq_len, hidden_size, device=device, dtype=dtype)
        q = torch.randn(total_seq_len, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_seq_len, k_heads, head_dim, device=device, dtype=dtype)

        kv_lens = torch.randint(0, max_seq_len - seqlen, (bs,), device=device, dtype=torch.int32)
        cos, sin = rot_pos_emb(x, cu_q_lens=cu_seqlens, total_seq_lens=kv_lens)
        if head_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
    elif mode == "decode":
        x = torch.randn(bs, hidden_size, device=device, dtype=dtype)
        q = torch.randn(bs, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(bs, k_heads, head_dim, device=device, dtype=dtype)
        kv_lens = torch.randint(0, max_seq_len - 1, (bs,), device=device, dtype=torch.int32)
        cos, sin = rot_pos_emb(x, position_ids=kv_lens)
        if head_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)

    rope = MojoApplyRoPE()
    rope_ref = MojoApplyRoPE._registry.get("torch")()

    rope.forward_diff_with(
        rope_ref,
        q,
        k,
        cos,
        sin,
        head_first,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "bs, grid, heads, head_dim, pad",
    [
        (4, (2, 4, 8), 8, 64, 10),
        (2, (1, 8, 8), 16, 128, 5),
        (3, (4, 4, 4), 4, 64, 3),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_grid_pos_emb(bs, grid, heads, head_dim, pad, dtype):
    device = get_torch_device()
    f, h, w = grid
    seq_len = f * h * w
    L = seq_len + pad

    x = torch.randn(bs, L, heads, head_dim, device=device, dtype=dtype)

    grid_sizes = torch.tensor([grid] * bs, device=device, dtype=torch.int32)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs_scalar = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, head_dim/2]
    cos = freqs_scalar.cos()[:, None, :]  # [seq_len, 1, head_dim/2]
    sin = freqs_scalar.sin()[:, None, :]  # [seq_len, 1, head_dim/2]
    freqs = torch.complex(cos, sin)  # complex64
    freqs_list = [freqs for _ in range(bs)]

    rope = MojoGridRoPE()
    rope_ref = MojoGridRoPE._registry.get("torch")()

    rope.forward_diff_with(rope_ref, x, grid_sizes, freqs_list, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("n_qh, n_kh, mrope_section, is_interleaved, model_name", [
    (28, 4, [16, 24, 24], False, "Qwen2-VL-7B"),      # Qwen2-VL / Qwen2.5-VL-7B
    (40, 8, [16, 24, 24], False, "Qwen2.5-VL-32B"),   # Qwen2.5-VL-32B
    (16, 8, [24, 20, 20], True, "Qwen3-VL-2B"),       # Qwen3-VL-2B (interleaved)
    (32, 8, [24, 20, 20], True, "Qwen3-VL-8B"),       # Qwen3-VL-8B (interleaved)
])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_mrope_qwen_models(
    num_tokens,
    n_qh,
    n_kh,
    head_dim,
    mrope_section,
    is_interleaved,
    model_name,
    dtype,
):
    """
    Test MRoPE with actual Qwen2-VL, Qwen2.5-VL, and Qwen3-VL model configurations.
    
    Key differences:
    - Qwen2/2.5-VL: mrope_section=[16, 24, 24], is_interleaved=False
    - Qwen3-VL: mrope_section=[24, 20, 20], is_interleaved=True
    
    All models use head_dim=128, rotary_dim=128 (full rotation).
    """
    device = get_torch_device()
    query, key, cos_table, sin_table, mrope_section_out = prepare_mrope_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device, dtype=dtype
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(mrope_ref, query, key, cos_table, sin_table, mrope_section_out, is_interleaved, head_dim=head_dim)


@pytest.mark.parametrize("num_tokens", [16, 64])
@pytest.mark.parametrize("n_qh, n_kh", [
    (32, 4),   # Typical GQA ratio 8:1
    (64, 8),   # Typical GQA ratio 8:1
])
@pytest.mark.parametrize("head_dim, mrope_section, description", [
    (128, [8, 12, 12], "partial_rotation_50pct"),   # rotary_dim=64, 50% rotation
    (128, [12, 18, 18], "partial_rotation_75pct"),  # rotary_dim=96, 75% rotation
    (96, [8, 12, 12], "small_head_full_rotation"),  # rotary_dim=64, full rotation
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_interleaved", [False])
@bypass_not_implemented
def test_mrope_partial_rotation(
    num_tokens,
    n_qh,
    n_kh,
    head_dim,
    mrope_section,
    description,
    dtype,
    is_interleaved,
):
    """
    Test MRoPE with partial rotation (head_dim > rotary_dim) and various configurations.
    This tests scenarios beyond Qwen VL's standard full rotation.
    """
    device = get_torch_device()
    query, key, cos_table, sin_table, mrope_section_out = prepare_mrope_test_inputs(
        num_tokens, n_qh, n_kh, head_dim, mrope_section, device, dtype=dtype
    )

    mrope = MojoMRoPE()
    mrope_ref = MojoMRoPE._registry.get("torch")()
    mrope.forward_diff_with(mrope_ref, query, key, cos_table, sin_table, mrope_section_out, is_interleaved, head_dim=head_dim)


@pytest.mark.parametrize("num_buckets", [32, 64])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize(
    "lq, lk",
    [
        (64, 64),
        (128, 512),
        (33, 97),
    ],
)
@bypass_not_implemented
def test_relative_embedding(num_buckets, num_heads, bidirectional, lq, lk):
    emb = MojoRelativeEmbedding(num_buckets=num_buckets, num_heads=num_heads, bidirectional=bidirectional)
    emb_ref = MojoRelativeEmbedding._registry.get("torch")(
        num_buckets=num_buckets, num_heads=num_heads, bidirectional=bidirectional
    )

    with torch.no_grad():
        weight = torch.randn(num_buckets, num_heads, dtype=torch.float32)
        emb.embedding.weight.copy_(weight)
        emb_ref.embedding.weight.copy_(weight)

    emb.forward_diff_with(emb_ref, lq, lk, atol=1e-5, rtol=1e-6)
