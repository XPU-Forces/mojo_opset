import functools

import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import auto_switch_platform

from mojo_opset import MojoSdpaFunction


@functools.lru_cache()
def generate_diffusion_attention_mask(
    seq_length: int,
    block_size: int,
) -> torch.Tensor:
    print("generate diffusion attention mask...")
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


def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
    block_size: int,
):
    query = torch.randn(bsz, q_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=True)
    # blockwise_diffusion_attn_mask = generate_diffusion_attention_mask(seq_length, block_size)
    blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, key, value, blockwise_diffusion_attn_mask, q_head_num != kv_head_num


def compare_tensor(a: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype = None):
    assert a.shape == ref.shape
    assert a.dtype == ref.dtype
    if dtype is None:
        dtype = a.dtype
    if dtype == torch.bfloat16:
        max_atol = 0.1
        max_rtol = 0.05
        mean_atol = 0.01
        mean_rtol = 0.01
    elif dtype == torch.float32:
        max_atol = 5e-3
        max_rtol = 5e-3
        mean_atol = 1e-4
        mean_rtol = 1e-4
    else:
        print(f"dtype {dtype} is not supported.")
        assert False

    assert torch.allclose(a, ref, atol=max_atol, rtol=max_rtol)
    assert torch.mean(torch.abs(ref - a)) < max_atol or torch.mean(torch.abs((ref - a) / (ref + mean_atol))) < mean_rtol


@pytest.mark.parametrize(
    "query, key, value, blockwise_diffusion_attn_mask, enable_gqa",
    [
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=5,
                kv_head_num=1,
                head_dim=128,
                seq_length=1024,
                block_size=32,
            )
        ),
    ],
)
@auto_switch_platform()
def test_diffusion_attention_func(monkeypatch, query, key, value, blockwise_diffusion_attn_mask, enable_gqa):
    ctx = MockFunctionCtx()
    o = MojoSdpaFunction.forward(ctx, query, key, value, blockwise_diffusion_attn_mask, 1.0, enable_gqa)

    ctx_ref = MockFunctionCtx()
    o_ref = MojoSdpaFunction._registry.get("ref").forward(
        ctx_ref, query, key, value, blockwise_diffusion_attn_mask, 1.0, enable_gqa
    )

    compare_tensor(o, o_ref)

    do = torch.rand_like(o)
    dq, dk, dv, _, _, _ = MojoSdpaFunction.backward(ctx, do)

    dq_ref, dk_ref, dv_ref, _, _, _ = MojoSdpaFunction._registry.get("ref").backward(ctx_ref, do)

    compare_tensor(dq, dq_ref)
    compare_tensor(dk, dk_ref)
    compare_tensor(dv, dv_ref)
