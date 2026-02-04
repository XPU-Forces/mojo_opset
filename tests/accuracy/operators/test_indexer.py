import random

import pytest
import torch


from mojo_opset import MojoLightningIndexer
from mojo_opset.experimental import MojoIndexer
from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform, bypass_not_implemented

TEST_SHAPES = [
    (128, 256, 256, 64, 128),
    (24, 1024, 1024, 128, 128),
    (24, 1, 16384, 128, 128),
]
TEST_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


@pytest.mark.parametrize(
    "query, query_scale, key, key_scale",
    [
        (
            torch.randn(B, M, H, K, dtype=dtype),
            torch.randn(B, M, H, dtype=torch.float32),
            torch.randn(B, N, K, dtype=dtype),
            torch.randn(B, N, K, dtype=torch.float32),
        )
        for (B, M, N, H, K) in TEST_SHAPES
        for dtype in TEST_DTYPES
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_lightning_indexer(query, query_scale, key, key_scale):
    indexer = MojoLightningIndexer()
    indexer_ref = indexer._registry.get("torch")()

    indexer.forward_diff_with(indexer_ref, query, query_scale, key, key_scale)


@pytest.mark.parametrize(
    "batch, q_seq_len, head_dim, dim, q_lora_rank, dummy_tensor, dtype",
    [
        (
            batch,
            q_seq_len,
            64,  # 128
            7168,
            1536,
            1,
            dtype,
        )
        for batch in [
            # 1,
            # 2,
            8,
            # 16,
        ]  # 32, 128
        for q_seq_len in [
            # 1,
            # 1024,
            4096,
        ]  # 4096, 8192
        # for q_head_num in [128, 64]
        for dtype in ["bfloat16", "float32"]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_indexer(batch, q_seq_len, head_dim, dim, q_lora_rank, dummy_tensor, dtype):
    device = get_platform()
    dtype = dtype_str_map[dtype]
    rope_head_dim = 32
    n_heads = 64
    start_pos = 0

    x = torch.randn(batch, q_seq_len, dim, device=device, dtype=dtype)
    dummy_tensor = torch.tensor(dummy_tensor, device=device)
    query_scale = torch.randn(batch, q_seq_len, q_lora_rank, device=device, dtype=dtype)
    topk = 2048 if q_seq_len >= 4096 else q_seq_len // 2
    freqs_cis = precompute_freqs_cis(q_seq_len, rope_head_dim, device=device)
    k_w = torch.randn(head_dim, dim, dtype=x.dtype, device=x.device)
    q_w = torch.randn(
        (n_heads * head_dim),
        q_lora_rank,
        dtype=query_scale.dtype,
        device=query_scale.device,
    )
    weights_w = torch.randn(n_heads, dim, dtype=torch.float32)

    indexer = MojoIndexer(n_heads=n_heads, head_dim=head_dim, qk_rope_head_dim=rope_head_dim, topk=topk)
    indexer_ref = MojoIndexer._registry.get("torch")(n_heads=n_heads, head_dim=head_dim, qk_rope_head_dim=rope_head_dim, topk=topk)

    indexer.to(dtype=dtype, device=device)

    #* sync weight
    wq_b = torch.randn_like(indexer.wq_b.weight.data)
    indexer.wq_b.weight.data.copy_(wq_b)
    indexer_ref.wq_b.weight.data.copy_(wq_b)

    wk = torch.randn_like(indexer.wk.weight.data)
    indexer.wk.weight.data.copy_(wk)
    indexer_ref.wk.weight.data.copy_(wk)

    k_norm_weight = torch.randn_like(indexer.k_norm.weight.data)
    k_norm_bias = torch.randn_like(indexer.k_norm.bias.data)
    indexer.k_norm.weight.data.copy_(k_norm_weight)
    indexer.k_norm.bias.data.copy_(k_norm_bias)
    indexer_ref.k_norm.weight.data.copy_(k_norm_weight)
    indexer_ref.k_norm.bias.data.copy_(k_norm_bias)

    weights_proj = torch.randn_like(indexer.weights_proj.weight.data)
    indexer.weights_proj.weight.data.copy_(weights_proj)
    indexer_ref.weights_proj.weight.data.copy_(weights_proj)

    indexer.forward_diff_with(
        indexer_ref,
        x,
        query_scale,
        start_pos,
        freqs_cis,
        None,
        k_w,
        q_w,
        weights_w,
    )


def precompute_freqs_cis(seqlen, dim, device) -> torch.Tensor:
    base = 10000.0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs, device=device), freqs)
    return freqs_cis


if __name__ == "__main__":
    pytest.main(["-v", "-s", "tests/accuracy/operators/test_indexer.py::test_indexer"])
    pass
