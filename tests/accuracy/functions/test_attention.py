import functools

import pytest
import torch

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
    blockwise_diffusion_attn_mask = generate_diffusion_attention_mask(seq_length, block_size)
    scale = 1.0 / (head_dim ** 0.5)
    enable_gqa = q_head_num != kv_head_num
    return query, key, value, blockwise_diffusion_attn_mask, scale, enable_gqa


@pytest.mark.parametrize(
    "query, key, value, blockwise_diffusion_attn_mask, scale, enable_gqa",
    [
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=1,
                kv_head_num=1,
                head_dim=128,
                seq_length=1024,
                block_size=32,
            ), 
        ),
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=16,
                kv_head_num=4,
                head_dim=128,
                seq_length=1024,
                block_size=32,
            ), 
        ),
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=16,
                kv_head_num=16,
                head_dim=128,
                seq_length=4096,
                block_size=32,
            ), 
        ),
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=16,
                kv_head_num=4,
                head_dim=128,
                seq_length=4096,
                block_size=32,
            ), 
        ),
    ],
)
@auto_switch_platform()
def test_diffusion_attention_func(monkeypatch, query, key, value, blockwise_diffusion_attn_mask, scale, enable_gqa):
    monkeypatch.setenv("MOJOSILUFUNCTION_FWD_MODE", "DIFF")
    monkeypatch.setenv("MOJOSILUFUNCTION_BWD_MODE", "DIFF")

    output = MojoSdpaFunction.apply(query, key, value, blockwise_diffusion_attn_mask, scale, enable_gqa)

    grad_output = torch.rand_like(output)
    output.backward(grad_output)
