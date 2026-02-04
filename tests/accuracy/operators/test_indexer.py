import random

import pytest
import torch

from mojo_opset import MojoLightningIndexer
from mojo_opset.experimental import MojoIndexer
from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform, bypass_not_implemented

TEST_SHAPES = [
    (8, 1024, 1024, 64, 64),
    (128, 256, 256, 64, 128),
    (24, 1024, 1024, 128, 128),
    (24, 1, 16384, 128, 128),
]
dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


@pytest.mark.parametrize(
    "B, M, N, H, K, dtype",
    [(B, M, N, H, K, dtype) for (B, M, N, H, K) in TEST_SHAPES for dtype in dtype_str_map.keys()],
)
@auto_switch_platform()
@bypass_not_implemented
def test_lightning_indexer(B, M, N, H, K, dtype):
    device = get_platform()
    dtype = dtype_str_map[dtype]
    query = torch.randn(B, M, H, K, dtype=dtype, device=device)
    query_scale = torch.randn(B, M, H, dtype=torch.float32, device=device)
    key = torch.randn(B, N, K, dtype=dtype, device=device)
    key_scale = torch.randn(B, N, dtype=torch.float32, device=device)

    indexer = MojoLightningIndexer()
    indexer_ref = indexer._registry.get("torch")()

    indexer.forward_diff_with(indexer_ref, query, query_scale, key, key_scale)


@pytest.mark.parametrize(
    "batch, q_seq_len, head_dim, dim, q_lora_rank, dummy_tensor, dtype",
    [
        (
            batch,
            q_seq_len,
            64,
            7168,
            1536,
            1,
            dtype,
        )
        for batch in [
            1,
            16,
            32,
        ]
        for q_seq_len in [
            1,
            1024,
            4096,
        ]
        # for head_dim in [128, 64]
        for dtype in ["bfloat16", "float16", "float32"]  # ",
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_indexer(batch, q_seq_len, head_dim, dim, q_lora_rank, dummy_tensor, dtype):
    device = get_platform()
    map_tol = {
        "bfloat16": (1.6e-2, 1e-5, 1.0),
        "float16": (1e-3, 1e-5, 1.0),
        "float32": (1.3e-6, 1e-5, 1.0),
    }
    atol, rtol, ptol = map_tol[dtype]
    dtype = dtype_str_map[dtype]

    rope_head_dim = 32
    n_heads = 64
    start_pos = 0

    x = torch.randn(batch, q_seq_len, dim, device=device, dtype=dtype)
    dummy_tensor = torch.tensor(dummy_tensor, device=device)
    query_scale = torch.randn(batch, q_seq_len, q_lora_rank, device=device, dtype=dtype)
    topk = 2048 if q_seq_len >= 4096 else q_seq_len // 2
    freqs_cis = precompute_freqs_cis(q_seq_len, rope_head_dim, device=device)

    indexer_ref = MojoIndexer._registry.get("torch")(n_heads=n_heads, head_dim=head_dim, qk_rope_head_dim=rope_head_dim, topk=topk)
    indexer_ref.to(dtype=dtype, device=device)

    # * ===========================  init weight  ===========================
    ##* wq_b MojoLinear
    indexer_ref.wq_b.weight.data.copy_(torch.randn_like(indexer_ref.wq_b.weight.data))
    if indexer_ref.wq_b.bias is not None:
        indexer_ref.wq_b.bias.data.copy_(torch.randn_like(indexer_ref.wq_b.bias.data))
    ##* wk MojoLinear
    indexer_ref.wk.weight.data.copy_(torch.randn_like(indexer_ref.wk.weight.data))
    if indexer_ref.wk.bias is not None:
        indexer_ref.wk.bias.data.copy_(torch.randn_like(indexer_ref.wk.bias.data))
    ##* k_norm MojoLayerNorm
    indexer_ref.k_norm.weight.data.copy_(torch.randn_like(indexer_ref.k_norm.weight.data))
    if indexer_ref.k_norm.bias is not None:
        indexer_ref.k_norm.bias.data.copy_(torch.randn_like(indexer_ref.k_norm.bias.data))
    ##* weights_proj MojoLinear
    indexer_ref.weights_proj.weight.data.copy_(torch.randn_like(indexer_ref.weights_proj.weight.data))
    if indexer_ref.weights_proj.bias is not None:
        indexer_ref.weights_proj.bias.data.copy_(torch.randn_like(indexer_ref.weights_proj.bias.data))

    indexer = MojoIndexer(parent_instance=indexer_ref)
    indexer.to(dtype=dtype, device=device)

    indexer.forward_diff_with(
        indexer_ref,
        x,
        query_scale,
        start_pos,
        freqs_cis,
        None,
        atol=atol,
        rtol=rtol,
        ptol=ptol,
    )


def precompute_freqs_cis(seqlen, dim, device) -> torch.Tensor:
    base = 10000.0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs, device=device), freqs)
    return freqs_cis
