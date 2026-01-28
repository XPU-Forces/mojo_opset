import torch
from mojo_opset.core.function import MojoFunction
from typing import Optional, Tuple

def swa_torch_forward(
    q: torch.Tensor, # [total_q_len, n_q_heads, head_dim]
    k: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
    v: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
    cu_seqlens_q: torch.Tensor, # [bsz + 1]
    cu_seqlens_kv: torch.Tensor, # [bsz + 1]
    max_seqlen_q: int,
    max_seqlen_kv: int,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    sm_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    _, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    
    o = torch.empty_like(q)
    bsz = cu_seqlens_q.shape[0] - 1
    for i in range(bsz):
        q_i = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        q_seq_len = q_i.shape[0]
        q_i = q_i.permute(1, 0, 2) # -> [n_q_heads, q_seq_len, head_dim]

        k_i = k[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]]
        kv_seq_len = k_i.shape[0]
        kv_cache_len = kv_seq_len - q_seq_len
        k_i_T = k_i.permute(1, 2, 0)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                k_i_T = k_i_T.repeat_interleave(n_q_heads // n_kv_heads, dim=0) # -> [n_q_heads, head_dim, kv_seq_len]
        qk_i = torch.bmm(q_i, k_i_T).float() * sm_scale # -> [n_q_heads, q_seq_len, kv_seq_len]

        if is_causal:
            causal_mask = (torch.arange(0, q_seq_len)[:, None] + kv_cache_len) >= torch.arange(0, kv_seq_len)[None, :]
        else:
            causal_mask = torch.ones((q_seq_len, kv_seq_len), dtype=torch.bool, device=q.device)
        
        if local_window_size is not None:
            local_window_mask = (
                torch.arange(0, q_seq_len)[:, None] + kv_cache_len
                <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
            ) & causal_mask
        else:
            local_window_mask = causal_mask
        
        if global_window_size is not None:
            global_window_mask = (torch.arange(0, q_seq_len) < global_window_size)[
                None, :
            ] & causal_mask
        else:
            global_window_mask = causal_mask

        qk_mask = local_window_mask | global_window_mask
        qk_i = torch.where(qk_mask, qk_i, -float("inf"))

        p_i = torch.softmax(qk_i, dim=-1) # -> [n_q_heads, q_seq_len, kv_seq_len]
        p_i = p_i.to(k.dtype)

        v_i = v[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]].permute(1, 0, 2) # -> [n_kv_heads, kv_seq_len, head_dim]
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0) # -> [n_q_heads, kv_seq_len, head_dim]
        o_i = torch.bmm(p_i, v_i) # -> [n_q_heads, q_seq_len, head_dim]
        o_i = o_i.permute(1, 0, 2) # -> [q_seq_len, n_q_heads, head_dim]
        o[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = o_i
    return o

class MojoSWAFunction(MojoFunction):

    @staticmethod
    def forward(
        ctx, 
        q: torch.Tensor, # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor, # [bsz + 1]
        cu_seqlens_kv: torch.Tensor, # [bsz + 1]
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:
        o = swa_torch_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, is_causal, local_window_size, global_window_size, sm_scale, gqa_interleave
        )

        ctx.save_for_backward(q, k, v, o, cu_seqlens_q, cu_seqlens_kv)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.is_causal = is_causal
        ctx.local_window_size = local_window_size
        ctx.global_window_size = global_window_size
        ctx.gqa_interleave = gqa_interleave
        return o


    def backward(
        ctx, 
        do: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               None, None, None, None, None, None, None, None, None]:
        q, k, v, o, cu_seqlens_q, cu_seqlens_kv = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_kv = ctx.max_seqlen_kv
        is_causal = ctx.is_causal
        local_window_size = ctx.local_window_size
        global_window_size = ctx.global_window_size
        gqa_interleave = ctx.gqa_interleave

        _, n_q_heads, head_dim = q.shape
        n_kv_heads = k.shape[1]

        bsz = cu_seqlens_q.shape[0] - 1

        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            o = swa_torch_forward(
                q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, is_causal, local_window_size, global_window_size, sm_scale, gqa_interleave
            )
            o.backward(do)
        dq = q.grad
        dk = k.grad
        dv = v.grad
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def flash_attn_sparse_torch(
    q,
    k,
    v,
    cu_seqlens,
    softmax_scale=None,
    local_window_size=0,
    global_window_size=0,
):

    T, H, Dq = q.shape
    Hk = k.shape[-2]
    gqa_ratio = H // Hk
    o = torch.zeros_like(q)
    if softmax_scale == None:
        softmax_scale = Dq ** (-0.5)

    bz = len(cu_seqlens) - 1

    for b in range(bz):
        for h in range(H):
            seq_start = cu_seqlens[b]
            seq_end = cu_seqlens[b + 1]
            seq_len = seq_end - seq_start
            hk = h // gqa_ratio

            b_q = q[seq_start:seq_end, h, :].float()
            b_k = k[seq_start:seq_end, hk, :].float()
            b_v = v[seq_start:seq_end, hk, :].float()
            b_qk = b_q @ b_k.T

            casual_mask = (
                torch.arange(0, seq_len)[:, None] >= torch.arange(0, seq_len)[None, :]
            )
            local_window_mask = (
                torch.arange(0, seq_len)[:, None]
                <= torch.arange(0, seq_len)[None, :] + local_window_size
            )
            global_window_mask = (torch.arange(0, seq_len) < global_window_size)[
                None, :
            ]

            b_qk_mask = casual_mask & (local_window_mask | global_window_mask)

            b_qk = torch.where(b_qk_mask.to(device=b_qk.device), b_qk, -float("inf"))
            b_qk = b_qk * softmax_scale
            b_qk = b_qk.softmax(dim=-1)

            b_o = b_qk @ b_v
            o[seq_start:seq_end, h, :] = b_o.to(o.dtype)

    return o

def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    max_q_len: int,
    max_kv_prefix_len: int,
    dtype: torch.dtype = torch.bfloat16,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (bsz,), dtype=torch.int32)
    if max_kv_prefix_len > 0:
        kv_prefix_lens = torch.randint(max_kv_prefix_len // 2, max_kv_prefix_len, (bsz,), dtype=torch.int32)
    else:
        kv_prefix_lens = torch.zeros(bsz, dtype=torch.int32)
    kv_lens = kv_prefix_lens + q_lens
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), q_lens.cumsum(0)])
    cu_seqlens_kv = torch.cat([torch.zeros(1, dtype=torch.int32), kv_lens.cumsum(0)])

    query = torch.randn(cu_seqlens_q[-1].item(), q_head_num, head_dim, dtype=dtype)
    key = torch.randn(cu_seqlens_kv[-1].item(), kv_head_num, head_dim, dtype=dtype)
    value = torch.randn(cu_seqlens_kv[-1].item(), kv_head_num, head_dim, dtype=dtype)

    # blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, key, value, cu_seqlens_q, cu_seqlens_kv


def test_swa_function():
    test_configs = [
        (4, 4, 4, 128, 256, 0, torch.float32),
        (4, 4, 4, 128, 256, 0, torch.bfloat16),
    ]

    local_window = 1023
    global_window = 4
    for bsz, q_head_num, kv_head_num, head_dim, max_q_len, max_kv_prefix_len, dtype in test_configs:
        print(bsz, q_head_num, kv_head_num, head_dim, max_q_len, max_kv_prefix_len, dtype)
        scale = 1.0 / head_dim ** 0.5
        query, key, value, cu_seqlens_q, cu_seqlens_kv = generate_test_data(bsz, q_head_num, kv_head_num, head_dim, max_q_len, max_kv_prefix_len, dtype)
        q_ref = query.clone().detach().requires_grad_(True)
        k_ref = key.clone().detach().requires_grad_(True)
        v_ref = value.clone().detach().requires_grad_(True)

        q_mojo = query.clone().detach().requires_grad_(True)
        k_mojo = key.clone().detach().requires_grad_(True)
        v_mojo = value.clone().detach().requires_grad_(True)

        o_ref = flash_attn_sparse_torch(
            q_ref, k_ref, v_ref, cu_seqlens_q, scale, local_window, global_window
        )

        o_mojo = MojoSWAFunction.apply(
            q_mojo, k_mojo, v_mojo, cu_seqlens_q, cu_seqlens_kv, max_q_len, max_kv_prefix_len, True, local_window, global_window, scale, False
        )

        torch.testing.assert_close(o_ref, o_mojo, atol=2e-2, rtol=2e-3)
        grad_out = torch.randn_like(o_ref)
        o_ref.backward(grad_out)
        o_mojo.backward(grad_out)
        torch.testing.assert_close(q_ref.grad, q_mojo.grad, atol=2e-2, rtol=2e-3)
        torch.testing.assert_close(k_ref.grad, k_mojo.grad, atol=2e-2, rtol=2e-3)
        torch.testing.assert_close(v_ref.grad, v_mojo.grad, atol=2e-2, rtol=2e-3)

if __name__ == "__main__":
    test_swa_function()