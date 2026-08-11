import gc

import pytest
import torch

from mojo_opset.experimental import MojoLightningIndexer
from mojo_opset.experimental import MojoIndexer
from mojo_opset.utils.acc import check_tol_diff
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform, bypass_not_implemented

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
    device = get_torch_device()
    dtype = dtype_str_map[dtype]
    query = torch.randn(B, M, H, K, dtype=dtype, device=device)
    query_scale = torch.randn(B, M, H, dtype=torch.float32, device=device)
    key = torch.randn(B, N, K, dtype=dtype, device=device)
    key_scale = torch.randn(B, N, dtype=torch.float32, device=device)

    indexer = MojoLightningIndexer()
    indexer_ref = indexer._registry.get("torch")()

    indexer.forward_diff_with(indexer_ref, query, query_scale, key, key_scale)


@pytest.mark.parametrize(
    "batch, q_seq_len, head_dim, dim, q_lora_rank, dtype",
    [
        (batch, q_seq_len, 64, 7168, 1536, dtype)
        for batch in [1, 16, 32]
        for q_seq_len in [1, 1024, 4096]
        for dtype in ["bfloat16", "float16", "float32"]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_indexer(batch, q_seq_len, head_dim, dim, q_lora_rank, dtype):
    torch.manual_seed(42)
    device = get_torch_device()
    dtype = dtype_str_map[dtype]

    rope_head_dim = 32
    n_heads = 64
    start_pos = 0

    x = torch.randn(batch, q_seq_len, dim, device=device, dtype=dtype)
    query_scale = torch.randn(batch, q_seq_len, q_lora_rank, device=device, dtype=dtype)
    topk = 2048 if q_seq_len >= 4096 else q_seq_len // 2
    freqs_cis = precompute_freqs_cis(q_seq_len, rope_head_dim, device=device)

    init_kwargs = dict(
        n_heads=n_heads,
        head_dim=head_dim,
        qk_rope_head_dim=rope_head_dim,
        topk=topk,
        # size the k_cache buffers to this case instead of the 128x32768
        # defaults (~284MB per instance), to bound memory over the sweep
        max_batch_size=batch,
        max_seq_len=q_seq_len,
    )

    indexer_ref = MojoIndexer._registry.get("torch")(**init_kwargs)
    indexer_ref.to(dtype=dtype, device=device)

    indexer_ref.wq_b.weight.data.copy_(torch.randn_like(indexer_ref.wq_b.weight.data))
    if indexer_ref.wq_b.bias is not None:
        indexer_ref.wq_b.bias.data.copy_(torch.randn_like(indexer_ref.wq_b.bias.data))
    indexer_ref.wk.weight.data.copy_(torch.randn_like(indexer_ref.wk.weight.data))
    if indexer_ref.wk.bias is not None:
        indexer_ref.wk.bias.data.copy_(torch.randn_like(indexer_ref.wk.bias.data))
    indexer_ref.k_norm.weight.data.copy_(torch.randn_like(indexer_ref.k_norm.weight.data))
    if indexer_ref.k_norm.bias is not None:
        indexer_ref.k_norm.bias.data.copy_(torch.randn_like(indexer_ref.k_norm.bias.data))
    indexer_ref.weights_proj.weight.data.copy_(torch.randn_like(indexer_ref.weights_proj.weight.data))
    if indexer_ref.weights_proj.bias is not None:
        indexer_ref.weights_proj.bias.data.copy_(torch.randn_like(indexer_ref.weights_proj.bias.data))

    indexer = MojoIndexer(**init_kwargs)
    indexer.to(dtype=dtype, device=device)
    indexer.load_state_dict(indexer_ref.state_dict(), strict=False)

    # NOTE: index_score comes from an int8-quantized pipeline. One quantization
    # step is ~1/127 (~0.8%) of the per-token max abs value, and upstream bf16
    # ulp differences between two implementations flip a sparse set of int8
    # values, shifting scores by O(1-100) on |score|~1e4. Comparing the integer
    # topk indices element-wise would require bit-identical score rankings and
    # is not achievable across implementations, so compare score and selection
    # separately with quantization-step-scale tolerances.
    res_indices, res_score = indexer.forward(x, query_scale, start_pos, freqs_cis, None)
    ref_indices, ref_score = indexer_ref.forward(x, query_scale, start_pos, freqs_cis, None)

    # 1) score check with int8-quantization-step-scale tolerances.
    check_tol_diff(res_score, ref_score, atol=1e-2, rtol=2e-2, ptol=0.98)

    # 2) topk selection check, order-invariant: the selected indices must score
    #    (under the reference's own scoring) as high as the reference selection.
    if ref_indices.numel() > 0:
        ref_sorted = ref_score.gather(-1, ref_indices).sort(dim=-1, descending=True).values
        res_sorted = ref_score.gather(-1, res_indices).sort(dim=-1, descending=True).values
        check_tol_diff(res_sorted, ref_sorted, atol=1.0, rtol=1e-2, ptol=0.999)

    # release the big tensors deterministically before the next parametrized case
    del res_indices, res_score, ref_indices, ref_score, x, query_scale, indexer, indexer_ref
    gc.collect()
    empty_cache = getattr(getattr(torch, device, None), "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


def precompute_freqs_cis(seqlen, dim, device) -> torch.Tensor:
    base = 10000.0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs, device=device), freqs)
    return freqs_cis


if __name__ == "__main__":
    pytest.main([__file__ + "::test_indexer"])
