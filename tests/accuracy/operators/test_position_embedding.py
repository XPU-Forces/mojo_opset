import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset import MojoRoPE
from mojo_opset.utils.platform import get_platform

torch.random.manual_seed(42)


@pytest.mark.parametrize("bs", [1, 6])
@pytest.mark.parametrize("seqlen", [124, 555, 2048])
@pytest.mark.parametrize(
    "q_heads, k_heads",
    [
        (32, 8),
        (8, 2),
        (16, 8),
        (64, 8),
        (64, 4),
    ],
)
@pytest.mark.parametrize("head_dim", [96])
@pytest.mark.parametrize("rope_percentage", [1.0, 0.3333333333333333333333])
@pytest.mark.parametrize("mode", ["padding_prefill", "varlen_prefill", "decode"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_pos_emb(bs, seqlen, q_heads, k_heads, head_dim, rope_percentage, mode, dtype):
    device = get_platform()
    max_seq_len = 32768

    rope_dim = int(head_dim * rope_percentage)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, :, :]
    sin = emb.sin()[None, :, :]

    head_first = True

    if mode == "padding_prefill":
        q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        cos = cos[:, :seqlen, :]
        sin = sin[:, :seqlen, :]

        cu_seqlens = None
        kv_lens = None

    elif mode == "varlen_prefill":
        seq_lens = torch.randint(1, seqlen + 1, (bs,), device=device, dtype=torch.int32)
        total_seq_len = seq_lens.sum().item()
        cu_seqlens = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)

        q = torch.randn(total_seq_len, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_seq_len, k_heads, head_dim, device=device, dtype=dtype)

        kv_lens = torch.randint(0, max_seq_len - seqlen, (bs,), device=device, dtype=torch.int32)
        head_first = False

    elif mode == "decode":
        q = torch.randn(bs, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(bs, k_heads, head_dim, device=device, dtype=dtype)
        cu_seqlens = None
        kv_lens = torch.randint(0, max_seq_len - 1, (bs,), device=device, dtype=torch.int32)
        head_first = False

    rope = MojoRoPE()
    rope_ref = MojoRoPE._registry.get("torch")()

    rope.forward_diff_with(rope_ref, q, k, cos, sin, cu_seqlens, kv_lens, head_first, rope_percentage)
