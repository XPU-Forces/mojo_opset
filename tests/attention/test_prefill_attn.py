import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoPrefillGQA


def generate_prefill_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype,
):
    assert num_q_heads % num_kv_heads == 0, \
        f"GQA need {num_q_heads} % {num_kv_heads} != 0"

    query = torch.randn(batch_size, num_q_heads, seq_len, head_dim, dtype=dtype)
    k_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype)
    v_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype)

    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32)

    return query, k_cache, v_cache, cu_seqlens_q

test_configs = [
    (2, 16, 4, 128, 1024, torch.bfloat16, "M_BF16"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, atol, rtol",
    [
        pytest.param(
            *generate_prefill_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                seq_len=Q_LEN,
                dtype=dtype,
            ),
            2e-2 if dtype != torch.float32 else 1e-5,
            2e-3 if dtype != torch.float32 else 1e-6,
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, dtype, ID in test_configs
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB"])
@auto_switch_platform()
@bypass_not_implemented
def test_prefill_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    atol: float,
    rtol: float,
    gqa_layout: str,
):
    op = MojoPrefillGQA(
        is_causal=True,
        is_prefill=True,
        gqa_layout=gqa_layout,
    )

    op.forward_diff(
        query,
        k_cache,
        v_cache,
        cu_seqlens_q,
        atol=atol,
        rtol=rtol,
    )
