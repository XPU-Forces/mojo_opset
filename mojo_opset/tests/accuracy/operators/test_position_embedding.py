import pytest
import torch

from mojo_opset.tests.accuracy.operators._vision_rope_2d_golden import (
    _vision_rope_2d_golden,
)
from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import bypass_not_implemented

from mojo_opset import MojoRotaryEmbedding
from mojo_opset import MojoApplyRoPE
from mojo_opset import MojoGridRoPE
from mojo_opset import MojoRelativeEmbedding
from mojo_opset import MojoVisionRoPE2D
from mojo_opset.utils.platform import get_torch_device

torch.random.manual_seed(42)

VISION_VIT_CONFIG = {
    "img_size": 448,
    "patch_size": 14,
    "embed_dim": 1280,
    "depth": 27,
    "num_heads": 20,
    "head_dim": 64,
    "mlp_ratio": 4,
    "mlp_intermediate_dim": 5120,
    "rope_theta": 10000.0,
    "adapooling_factor": 2,
}


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



@pytest.mark.parametrize(
    "grid",
    [
        ((4, 4),),
        ((8, 6),),
        ((8, 8), (4, 6)),
    ],
)
@pytest.mark.parametrize(
    "vision_config",
    [
        pytest.param(VISION_VIT_CONFIG, id="vision_448_27l_20h_h64"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_vision_rope_2d_packed_against_golden(
    grid, vision_config, dtype
):
    device = get_torch_device()
    rope_theta = vision_config["rope_theta"]
    head_dim = vision_config["head_dim"]
    q_heads = vision_config["num_heads"]
    k_heads = vision_config["num_heads"]
    adapooling_factor = vision_config["adapooling_factor"]

    grid_hw = torch.tensor(grid, device=device, dtype=torch.int32)
    total_tokens = int(grid_hw.to(dtype=torch.int64).prod(dim=-1).sum().item())

    q = torch.randn(total_tokens, q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, k_heads, head_dim, device=device, dtype=dtype)

    rope = MojoVisionRoPE2D(rope_theta=rope_theta, rope_dim=head_dim, adapooling_factor=adapooling_factor)
    q_rot, k_rot = rope(q, k, grid_hw)
    q_ref, k_ref = _vision_rope_2d_golden(
        q, k, grid_hw, rope_theta, head_dim, adapooling_factor=adapooling_factor
    )

    assert_close(q_rot, q_ref)
    assert_close(k_rot, k_ref)


@pytest.mark.parametrize("invalid_adapooling_factor", [0, -1])
def test_vision_rope_2d_rejects_non_positive_adapooling_factor(invalid_adapooling_factor):
    with pytest.raises(AssertionError, match="adapooling_factor must be >= 1"):
        MojoVisionRoPE2D(adapooling_factor=invalid_adapooling_factor)


@pytest.mark.parametrize(
    "vision_config, batch_size, seq_len, grid_hw_data",
    [
        pytest.param(VISION_VIT_CONFIG, 2, 8, [[2, 4], [2, 4]], id="native_4d_layout"),
    ],
)
def test_vision_rope_2d_rejects_non_native_layout_extensions(
    vision_config, batch_size, seq_len, grid_hw_data
):
    q_heads = vision_config["num_heads"]
    k_heads = vision_config["num_heads"]
    head_dim = vision_config["head_dim"]
    rope = MojoVisionRoPE2D(rope_dim=head_dim)
    q = torch.randn(batch_size, seq_len, q_heads, head_dim)
    k = torch.randn(batch_size, seq_len, k_heads, head_dim)
    grid_hw = torch.tensor(grid_hw_data, dtype=torch.int32)
    with pytest.raises(AssertionError, match="packed token-first"):
        rope(q, k, grid_hw)


@pytest.mark.parametrize(
    "vision_config, rope_dim, total_tokens, grid_hw_data",
    [
        pytest.param(VISION_VIT_CONFIG, 32, 8, [[2, 4]], id="partial_rope_dim"),
    ],
)
def test_vision_rope_2d_rejects_partial_rope_dim(
    vision_config, rope_dim, total_tokens, grid_hw_data
):
    q_heads = vision_config["num_heads"]
    k_heads = vision_config["num_heads"]
    head_dim = vision_config["head_dim"]
    rope = MojoVisionRoPE2D(rope_dim=rope_dim)
    q = torch.randn(total_tokens, q_heads, head_dim)
    k = torch.randn(total_tokens, k_heads, head_dim)
    grid_hw = torch.tensor(grid_hw_data, dtype=torch.int32)
    with pytest.raises(AssertionError, match="full head_dim"):
        rope(q, k, grid_hw)
