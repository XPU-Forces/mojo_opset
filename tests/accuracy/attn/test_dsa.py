import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoDecodeDSA
from mojo_opset import MojoPrefillDSA


@pytest.mark.parametrize(
    "batch, q_seq_len, k_seq_len, head_num, qk_head_dim, v_head_dim, dummy_tensor",
    [
        (
            batch,
            q_seq_len,
            q_seq_len,
            16,
            192,
            128,
            torch.randn(1),
        )
        for batch, q_seq_len in [
            [1, 1024], [1, 4096], [1, 8192], [2, 1024], [2, 4096], [2, 8192],
            [8, 1024], [8, 4096], [16, 1024], [16, 4096], [32, 1024], [128, 1024]
        ]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_dsa_prefill(batch, q_seq_len, k_seq_len, head_num, qk_head_dim, v_head_dim, dummy_tensor):
    device = dummy_tensor.device
    query = torch.randn(
        batch,
        q_seq_len,
        head_num,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        k_seq_len,
        head_num,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        batch,
        k_seq_len,
        head_num,
        v_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    topk_indices = torch.randint(0, k_seq_len, (batch, 1, 1024), device=device, dtype=torch.int32)

    op = MojoPrefillDSA(is_causal=True)

    atol, rtol = 1e-3, 1e-3
    op.forward_diff(query, key, value, topk_indices, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "batch, q_seq_len, k_seq_len, head_num, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, dummy_tensor",
    [
        (
            batch,
            1,
            k_seq_len,
            16,
            128,
            64,
            512,
            128,
            torch.randn(1),
        )
        for batch in [1, 2, 8, 16, 32, 128]
        for k_seq_len in [1024, 4096, 8192]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_dsa_decode(batch, q_seq_len, k_seq_len, head_num, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, dummy_tensor):
    device = dummy_tensor.device
    query_nope = torch.randn(
        batch,
        q_seq_len,
        head_num,
        qk_nope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    query_pe = torch.randn(
        batch,
        q_seq_len,
        head_num,
        qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        batch,
        k_seq_len,
        kv_lora_rank,
        device=device,
        dtype=torch.bfloat16,
    )
    pe_cache = torch.randn(
        batch,
        k_seq_len,
        qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    wkv_b = torch.randn(
        head_num * (qk_nope_head_dim + v_head_dim),
        kv_lora_rank,
        device=device,
        dtype=torch.bfloat16,
    )

    random_values = torch.randn(
        batch,
        q_seq_len,
        k_seq_len,
        device=device,
    )
    _, topk_indices = torch.topk(random_values, 1024, dim=-1)  # top_k = 1024

    op = MojoDecodeDSA()

    atol, rtol = 1e-3, 1e-3
    op.forward_diff(query_nope, query_pe, wkv_b, kv_cache, pe_cache, topk_indices, start_pos=k_seq_len-1, atol=atol, rtol=rtol)
