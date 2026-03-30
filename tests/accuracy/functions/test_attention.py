import pytest
import torch
import math

from mojo_opset import MojoSWAFunction

from tests.utils import assert_close
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

def generate_sdpa_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_seqlens_kv = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0)])

    total_q_tokens = cu_seqlens_q[-1].item()
    total_kv_tokens = cu_seqlens_kv[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    key = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    grad_out = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)

    return query, key, value, grad_out, cu_seqlens_q, cu_seqlens_kv

test_configs_swa = [
    (2, 16, 4, 128, 1024, 0, torch.float32, "M_F32"),
    (2, 16, 4, 96, 1024, 0, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 16, 4, 128, 4096, 0, torch.bfloat16, "M_BF16_LONG"),
]

@pytest.mark.parametrize(
    "query, key, value, grad_out, cu_seqlens_q, cu_seqlens_kv",
    [
        pytest.param(
            *generate_sdpa_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, dtype, ID in test_configs_swa
    ],
)
@pytest.mark.parametrize("gqa_interleave, global_window, local_window", [
    (True, 4, 255),
    (False, 4, 1023),
])
@bypass_not_implemented
@auto_switch_platform()
def test_swa_function(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    gqa_interleave: bool,
    global_window: int,
    local_window: int,
):
    swa_func = MojoSWAFunction.apply

    swa_func_ref = MojoSWAFunction._registry.get("torch").apply

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = query.clone().detach().requires_grad_(True)
    k = key.clone().detach().requires_grad_(True)
    v = value.clone().detach().requires_grad_(True)
    o = swa_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        True,
        local_window,
        global_window,
        softmax_scale,
        gqa_interleave,
        True,
    )
    o.backward(grad_out)

    q_ref = query.clone().detach().requires_grad_(True)
    k_ref = key.clone().detach().requires_grad_(True)
    v_ref = value.clone().detach().requires_grad_(True)
    o_ref = swa_func_ref(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens_q,
        cu_seqlens_kv,
        True,
        local_window,
        global_window,
        softmax_scale,
        gqa_interleave,
        True,
    )
    o_ref.backward(grad_out)

    assert_close(o, o_ref)
    assert_close(q.grad, q_ref.grad)
    assert_close(k.grad, k_ref.grad)
    assert_close(v.grad, v_ref.grad)

    


