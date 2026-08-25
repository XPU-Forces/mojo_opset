from typing import Optional, Tuple

import pytest
import torch
import math

from mojo_opset import MojoSWAFunction

from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform

def _generate_window_mask_chunk(
    q_start: int,
    q_end: int,
    kv_seq_len: int,
    kv_computed_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    q_arange = torch.arange(q_start, q_end, device=device)
    kv_arange = torch.arange(0, kv_seq_len, device=device)
    causal_mask = (q_arange[:, None] + kv_computed_len) >= kv_arange[None, :]
    if local_window_size is not None or global_window_size is not None:
        local_window_mask = (
            (q_arange[:, None] + kv_computed_len <= kv_arange[None, :] + local_window_size)
            if local_window_size is not None
            else False
        )
        global_window_mask = (
            (kv_arange < global_window_size)[None, :]
            if global_window_size is not None
            else False
        )
        mask = causal_mask & (local_window_mask | global_window_mask)
    else:
        mask = causal_mask
    return mask


def _chunked_swa_torch_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    output_f32: bool = False,
    q_chunk_size: int = 1024,
) -> Tuple[torch.Tensor, ...]:
    total_q_len, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    o_f32 = torch.empty_like(q, dtype=torch.float32)
    softmax_lse = torch.empty((n_q_heads, total_q_len), dtype=torch.float32, device=q.device)
    bsz = cu_q_lens.shape[0] - 1

    for i in range(bsz):
        q_batch_start = cu_q_lens[i].item()
        q_batch_end = cu_q_lens[i + 1].item()
        kv_batch_start = cu_total_seq_lens[i].item()
        kv_batch_end = cu_total_seq_lens[i + 1].item()
        q_seq_len = q_batch_end - q_batch_start
        kv_seq_len = kv_batch_end - kv_batch_start
        kv_computed_len = kv_seq_len - q_seq_len

        k_i = k[kv_batch_start:kv_batch_end]
        v_i = v[kv_batch_start:kv_batch_end]

        k_i_T = k_i.permute(1, 2, 0)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                k_i_T = k_i_T.repeat_interleave(n_q_heads // n_kv_heads, dim=0)

        v_i_perm = v_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i_perm = v_i_perm.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                v_i_perm = v_i_perm.repeat_interleave(n_q_heads // n_kv_heads, dim=0)

        for qc_start in range(0, q_seq_len, q_chunk_size):
            qc_end = min(qc_start + q_chunk_size, q_seq_len)

            q_chunk = q[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)

            s_chunk = torch.bmm(q_chunk, k_i_T).float() * softmax_scale

            if is_causal:
                s_mask = _generate_window_mask_chunk(
                    qc_start, qc_end, kv_seq_len, kv_computed_len,
                    local_window_size, global_window_size, device=s_chunk.device,
                )
                s_chunk = torch.where(s_mask, s_chunk, float("-inf"))

            m_chunk = torch.max(s_chunk, dim=-1, keepdim=True).values
            s_chunk = s_chunk - m_chunk
            p_chunk = torch.exp(s_chunk)
            l_chunk = torch.sum(p_chunk, dim=-1, keepdim=True)
            p_chunk = p_chunk.to(v.dtype)

            o_chunk = torch.bmm(p_chunk, v_i_perm).float()
            o_chunk = o_chunk / l_chunk
            o_chunk = o_chunk.permute(1, 0, 2)
            o_f32[q_batch_start + qc_start : q_batch_start + qc_end] = o_chunk

            lse_chunk = m_chunk + torch.log(l_chunk)
            softmax_lse[:, q_batch_start + qc_start : q_batch_start + qc_end] = lse_chunk.squeeze(-1)

    o = o_f32.to(q.dtype)
    if output_f32:
        return o, softmax_lse, o_f32
    else:
        return o, softmax_lse


def _chunked_swa_torch_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    q_chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = torch.sum(o.float() * do.float(), dim=-1)
    bsz = cu_q_lens.shape[0] - 1
    group_size = n_q_heads // n_kv_heads if n_q_heads != n_kv_heads else 1

    for i in range(bsz):
        q_batch_start = cu_q_lens[i].item()
        q_batch_end = cu_q_lens[i + 1].item()
        kv_batch_start = cu_total_seq_lens[i].item()
        kv_batch_end = cu_total_seq_lens[i + 1].item()
        q_seq_len = q_batch_end - q_batch_start
        kv_seq_len = kv_batch_end - kv_batch_start
        kv_computed_len = kv_seq_len - q_seq_len

        k_i = k[kv_batch_start:kv_batch_end]
        v_i = v[kv_batch_start:kv_batch_end]

        k_i_perm = k_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_expanded = k_i_perm.repeat((group_size, 1, 1))
            else:
                k_i_expanded = k_i_perm.repeat_interleave(group_size, dim=0)
        else:
            k_i_expanded = k_i_perm

        v_i_perm = v_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i_expanded = v_i_perm.repeat((group_size, 1, 1))
            else:
                v_i_expanded = v_i_perm.repeat_interleave(group_size, dim=0)
        else:
            v_i_expanded = v_i_perm

        # 使用 fp32 累加 dk/dv，避免长序列下 bf16 累加的舍入误差累积
        # （全局 token 被所有 query 关注，bf16 累加次数过多导致精度损失）
        dk_i = torch.zeros((kv_seq_len, n_kv_heads, head_dim), dtype=torch.float32, device=k.device)
        dv_i = torch.zeros((kv_seq_len, n_kv_heads, head_dim), dtype=torch.float32, device=v.device)

        for qc_start in range(0, q_seq_len, q_chunk_size):
            qc_end = min(qc_start + q_chunk_size, q_seq_len)

            q_chunk = q[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)
            do_chunk = do[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)

            s_chunk = torch.bmm(q_chunk, k_i_expanded.mT).float() * softmax_scale

            if is_causal:
                s_mask = _generate_window_mask_chunk(
                    qc_start, qc_end, kv_seq_len, kv_computed_len,
                    local_window_size, global_window_size, device=s_chunk.device,
                )
                s_chunk = torch.where(s_mask, s_chunk, float("-inf"))

            lse_chunk = softmax_lse[:, q_batch_start + qc_start : q_batch_start + qc_end]
            p_chunk = torch.exp(s_chunk - lse_chunk.unsqueeze(-1))

            dp_chunk = torch.bmm(do_chunk, v_i_expanded.mT).float()
            delta_chunk = delta[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0).unsqueeze(-1)
            ds_chunk = p_chunk * (dp_chunk - delta_chunk)
            ds_chunk = ds_chunk * softmax_scale
            ds_chunk = ds_chunk.to(do_chunk.dtype)
            p_chunk = p_chunk.to(do_chunk.dtype)

            dq_chunk = torch.bmm(ds_chunk, k_i_expanded)
            dq[q_batch_start + qc_start : q_batch_start + qc_end] = dq_chunk.permute(1, 0, 2)

            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    ds_reduced = ds_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    q_reduced = q_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    p_reduced = p_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    do_reduced = do_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                else:
                    ds_reduced = ds_chunk.unflatten(0, (n_kv_heads, group_size))
                    q_reduced = q_chunk.unflatten(0, (n_kv_heads, group_size))
                    p_reduced = p_chunk.unflatten(0, (n_kv_heads, group_size))
                    do_reduced = do_chunk.unflatten(0, (n_kv_heads, group_size))

                ds_reduced = ds_reduced.flatten(1, 2)
                q_reduced = q_reduced.flatten(1, 2)
                p_reduced = p_reduced.flatten(1, 2)
                do_reduced = do_reduced.flatten(1, 2)
            else:
                ds_reduced = ds_chunk
                q_reduced = q_chunk
                p_reduced = p_chunk
                do_reduced = do_chunk

            dk_i += torch.bmm(ds_reduced.mT, q_reduced).float().permute(1, 0, 2)
            dv_i += torch.bmm(p_reduced.mT, do_reduced).float().permute(1, 0, 2)

        dk[kv_batch_start:kv_batch_end] = dk_i.to(k.dtype)
        dv[kv_batch_start:kv_batch_end] = dv_i.to(v.dtype)


    return dq, dk, dv

def generate_sdpa_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    dtype: torch.dtype,
):
    #torch.manual_seed(43)
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0).to(torch.int32)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0).to(torch.int32)])

    total_q_tokens = cu_q_lens[-1].item()
    total_kv_tokens = cu_total_seq_lens[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    key = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    grad_out = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)

    return query, key, value, grad_out, cu_q_lens, cu_total_seq_lens

test_configs_swa = [
    (2, 16, 4, 128, 1024, 0, torch.float32, "M_F32"),
    (2, 16, 4, 96, 1024, 0, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 16, 4, 128, 4096, 0, torch.bfloat16, "M_BF16_LONG"),
    (1, 12, 4, 128, 1024, 0,torch.bfloat16, "M_BF16_1024"),
    (1, 12, 4, 128, 2048, 0, torch.bfloat16, "M_BF16_2048"),
    (1, 12, 4, 128, 4096, 0, torch.bfloat16, "M_BF16_4096"),
    (1, 12, 4, 128, 8192, 0, torch.bfloat16, "M_BF16_8192"),
    (1, 12, 4, 128, 16384, 0, torch.bfloat16, "M_BF16_16384"),
    (1, 12, 4, 128, 32768, 0, torch.bfloat16, "M_BF16_32768"),
    # (1, 12, 4, 128, 65536, 0, torch.bfloat16, "M_BF16_65536"),
    # (1, 12, 4, 128, 131072, 0, torch.bfloat16, "M_BF16_131072"),
]

@pytest.mark.parametrize(
    "query, key, value, grad_out, cu_q_lens, cu_total_seq_lens",
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
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    gqa_interleave: bool,
    global_window: int,
    local_window: int,
):   
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    max_q_len = q_lens.max().item()
    if max_q_len > 8192 and get_platform() != "npu":
        pytest.skip("large shape only on NPU, to avoid OOM on other platform")
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
        cu_q_lens,
        cu_total_seq_lens,
        True,
        local_window,
        global_window,
        softmax_scale,
        gqa_interleave,
        True,
    )
    o.backward(grad_out)
    if max_q_len <= 8192:
        print(f"max_q_len {max_q_len}, use swa_func_ref")
        q_ref = query.clone().detach().requires_grad_(True)
        k_ref = key.clone().detach().requires_grad_(True)
        v_ref = value.clone().detach().requires_grad_(True)
        o_ref = swa_func_ref(
            q_ref,
            k_ref,
            v_ref,
            cu_q_lens,
            cu_total_seq_lens,
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
    else:
        print(f"max_q_len {max_q_len}, use chunked_swa_torch")
        q_ref = query.detach()
        k_ref = key.detach()
        v_ref = value.detach()
        do_ref = grad_out.detach()

        o_ref_cpu, softmax_lse_ref, o_f32_ref = _chunked_swa_torch_forward(
            q_ref,
            k_ref,
            v_ref,
            cu_q_lens,
            cu_total_seq_lens,
            True,
            local_window,
            global_window,
            softmax_scale,
            gqa_interleave,
            True,
        )
        dq_ref, dk_ref, dv_ref = _chunked_swa_torch_backward(
            do_ref,
            q_ref,
            k_ref,
            v_ref,
            o_f32_ref,
            softmax_lse_ref,
            cu_q_lens,
            cu_total_seq_lens,
            True,
            local_window,
            global_window,
            softmax_scale,
            gqa_interleave,
        )

        assert_close(o, o_ref_cpu)
        assert_close(q.grad, dq_ref)
        assert_close(k.grad, dk_ref)
        assert_close(v.grad, dv_ref)
