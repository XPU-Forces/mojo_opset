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
        for batch in [1, 2, 8, 16, 32, 128]
        for q_seq_len in [1024, 4096, 8192]
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
    "batch, q_seq_len, k_seq_len, head_num, qk_head_dim, v_head_dim, dummy_tensor",
    [
        (
            batch,
            1,
            k_seq_len,
            16,
            192,
            128,
            torch.randn(1),
        )
        for batch in [1, 2, 8, 16, 32, 128]
        for k_seq_len in [1024, 4096, 8192]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_dsa_decode(batch, q_seq_len, k_seq_len, head_num, qk_head_dim, v_head_dim, dummy_tensor):
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

    op = MojoDecodeDSA(is_causal=True)

    atol, rtol = 1e-3, 1e-3
    op.forward_diff(query, key, value, topk_indices, start_pos=0, atol=atol, rtol=rtol)
