import torch
from ..function import MojoFunction
from typing import Optional, Tuple


def _generate_window_mask(
    q_seq_len: int,
    kv_seq_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
) -> torch.Tensor:
    kv_computed_len = kv_seq_len - q_seq_len
    causal_mask = (torch.arange(0, q_seq_len)[:, None] + kv_computed_len) >= torch.arange(0, kv_seq_len)[None, :]
    if local_window_size is not None or global_window_size is not None:
        local_window_mask = (
            (
                torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
            )
            if local_window_size is not None
            else False
        )
        global_window_mask = (
            (torch.arange(0, kv_seq_len) < global_window_size)[None, :] if global_window_size is not None else False
        )
        mask = causal_mask & (local_window_mask | global_window_mask)
    else:
        mask = causal_mask

    return mask


def _swa_torch_forward(
    q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
    k: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
    v: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
    cu_seqlens_q: torch.Tensor,  # [bsz + 1]
    cu_seqlens_kv: torch.Tensor,  # [bsz + 1]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    output_f32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_q_len, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o_f32 = torch.empty_like(q, dtype=torch.float32)
    softmax_lse = torch.empty((n_q_heads, total_q_len), dtype=torch.float32, device=q.device)
    bsz = cu_seqlens_q.shape[0] - 1
    for i in range(bsz):
        q_i = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
        q_seq_len = q_i.shape[0]
        q_i = q_i.permute(1, 0, 2)  # -> [n_q_heads, q_seq_len, head_dim]

        k_i = k[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]]
        kv_seq_len = k_i.shape[0]
        k_i_T = k_i.permute(1, 2, 0)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                k_i_T = k_i_T.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, head_dim, kv_seq_len]
        s_i = torch.bmm(q_i, k_i_T).float() * softmax_scale  # -> [n_q_heads, q_seq_len, kv_seq_len]

        if is_causal:
            s_mask = _generate_window_mask(
                q_seq_len,
                kv_seq_len,
                local_window_size,
                global_window_size,
            ).to(s_i.device)
            s_i = torch.where(s_mask, s_i, float("-inf"))
        m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, q_seq_len, 1]
        s_i = s_i - m_i  # -> [n_q_heads, q_seq_len, kv_seq_len]
        p_i = torch.exp(s_i)
        l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, q_seq_len, 1]
        p_i = p_i.to(v.dtype)

        v_i = v[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]].permute(1, 0, 2)  # -> [n_kv_heads, kv_seq_len, head_dim]
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
        o_i = torch.bmm(p_i, v_i).float()  # -> [n_q_heads, q_seq_len, head_dim]
        o_i = o_i / l_i  # -> [n_q_heads, q_seq_len, head_dim]
        o_i = o_i.permute(1, 0, 2)  # -> [q_seq_len, n_q_heads, head_dim]
        o_f32[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = o_i
        lse_i = m_i + torch.log(l_i)  # -> [n_q_heads, q_seq_len, 1]
        assert lse_i.dtype == torch.float32
        softmax_lse[:, cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = lse_i.squeeze(-1)  # -> [q_seq_len, n_q_heads]

    o = o_f32.to(q.dtype)
    if output_f32:
        return o, softmax_lse, o_f32
    else:
        return o, softmax_lse


def _swa_torch_backward(
    do: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
    q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
    k: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
    v: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
    o: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
    softmax_lse: torch.Tensor,  # [n_q_heads, total_q_len]
    cu_seqlens_q: torch.Tensor,  # [bsz + 1]
    cu_seqlens_kv: torch.Tensor,  # [bsz + 1]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = torch.sum(o.float() * do.float(), dim=-1)  # -> [total_q_len, n_q_heads]
    bsz = cu_seqlens_q.shape[0] - 1
    for i in range(bsz):

        # Step 1: recompute p_i
        q_i = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
        q_seq_len = q_i.shape[0]
        q_i = q_i.permute(1, 0, 2)  # -> [n_q_heads, q_seq_len, head_dim]

        k_i = k[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]]
        kv_seq_len = k_i.shape[0]
        k_i = k_i.permute(1, 0, 2)  # -> [n_kv_heads, kv_seq_len, head_dim]
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i = k_i.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                k_i = k_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, head_dim, kv_seq_len]

        s_i = torch.bmm(q_i, k_i.mT).float() * softmax_scale  # -> [n_q_heads, q_seq_len, kv_seq_len]

        if is_causal:
            s_mask = _generate_window_mask(
                q_seq_len,
                kv_seq_len,
                local_window_size,
                global_window_size,
            ).to(s_i.device)
            s_i = torch.where(s_mask, s_i, float("-inf"))

        lse_i = softmax_lse[:, cu_seqlens_q[i] : cu_seqlens_q[i + 1]]  # -> [n_q_heads, q_seq_len]
        p_i = torch.exp(s_i - lse_i.unsqueeze(-1))  # -> [n_q_heads, q_seq_len, kv_seq_len]

        # Step 2: compute dv_i
        assert p_i.dtype == torch.float32
        v_i = v[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]]
        v_i = v_i.permute(1, 0, 2)  # -> [n_kv_heads, kv_seq_len, head_dim]
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]

        do_i = do[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
        do_i = do_i.permute(1, 0, 2)  # -> [n_q_heads, q_seq_len, head_dim]

        dp_i = torch.bmm(do_i, v_i.mT).float()  # -> [n_q_heads, q_seq_len, kv_seq_len]
        assert dp_i.dtype == torch.float32
        # Note: rowsum(P * dP) => rowsum(P * (dO @ V_T)) => rowsum(dO * (P @ V)) => rowsum(dO * O)
        delta_i = delta[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].permute(1, 0).unsqueeze(-1)  # -> [n_q_heads, q_seq_len]
        # print(f"{p_i.shape=} {do_i.shape=}")
        # delta_i_ref = torch.sum(p_i * dp_i, dim=-1, keepdim=True)
        # torch.testing.assert_close(delta_i, delta_i_ref)
        assert delta_i.dtype == torch.float32
        ds_i = p_i * (dp_i - delta_i)
        ds_i = ds_i * softmax_scale
        assert ds_i.dtype == torch.float32
        ds_i = ds_i.to(do_i.dtype)

        p_i = p_i.to(do_i.dtype)

        dq_i = torch.bmm(ds_i, k_i)  # -> [n_q_heads, q_seq_len, head_dim]
        dq[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = dq_i.permute(1, 0, 2)  # -> [q_seq_len, n_q_heads, head_dim]

        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                ds_i = ds_i.unflatten(0, (n_q_heads // n_kv_heads, n_kv_heads)).permute(
                    1, 0, 2, 3
                )  # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, kv_seq_len]
                q_i = q_i.unflatten(0, (n_q_heads // n_kv_heads, n_kv_heads)).permute(
                    1, 0, 2, 3
                )  # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, head_dim]
            else:
                ds_i = ds_i.unflatten(
                    0, (n_kv_heads, n_q_heads // n_kv_heads)
                )  # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, kv_seq_len]
                q_i = q_i.unflatten(
                    0, (n_kv_heads, n_q_heads // n_kv_heads)
                )  # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, head_dim]

            ds_i = ds_i.flatten(1, 2)  # -> [n_kv_heads, n_q_heads // n_kv_heads * q_seq_len, kv_seq_len]
            q_i = q_i.flatten(1, 2)  # -> [n_kv_heads, n_q_heads // n_kv_heads * q_seq_len, head_dim]

        dk_i = torch.bmm(ds_i.mT, q_i)  # -> [n_kv_heads, kv_seq_len, head_dim]
        dk[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]] = dk_i.permute(1, 0, 2)  # -> [kv_seq_len, n_kv_heads, head_dim]

        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                p_i = p_i.unflatten(0, (n_q_heads // n_kv_heads, n_kv_heads)).permute(
                    1, 0, 2, 3
                )  # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, kv_seq_len]
                do_i = do_i.unflatten(0, (n_q_heads // n_kv_heads, n_kv_heads)).permute(
                    1, 0, 2, 3
                )  # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, head_dim]
            else:
                p_i = p_i.unflatten(
                    0, (n_kv_heads, n_q_heads // n_kv_heads)
                )  # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, kv_seq_len]
                do_i = do_i.unflatten(
                    0, (n_kv_heads, n_q_heads // n_kv_heads)
                )  # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, head_dim]

            p_i = p_i.flatten(1, 2)  # -> [n_kv_heads, n_q_heads // n_kv_heads * q_seq_len, kv_seq_len]
            do_i = do_i.flatten(1, 2)  # -> [n_kv_heads, n_q_heads // n_kv_heads * q_seq_len, head_dim]

        dv_i = torch.bmm(p_i.mT, do_i)  # -> [n_q_heads, kv_seq_len, head_dim]
        dv[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]] = dv_i.permute(1, 0, 2)  # -> [kv_seq_len, n_kv_heads, head_dim]
    return dq, dk, dv


class MojoSWAFunction(MojoFunction):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        cu_seqlens_kv: torch.Tensor,  # [bsz + 1]
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        gqa_interleave: bool = False,
        output_f32: bool = False,
    ) -> torch.Tensor:
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        fwd_results = _swa_torch_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            is_causal,
            local_window_size,
            global_window_size,
            softmax_scale,
            gqa_interleave,
            output_f32,
        )
        if output_f32:
            o, softmax_lse, o_f32 = fwd_results
            ctx.save_for_backward(o_f32, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv)
        else:
            o, softmax_lse = fwd_results
            ctx.save_for_backward(o, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv)
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.local_window_size = local_window_size
        ctx.global_window_size = global_window_size
        ctx.gqa_interleave = gqa_interleave
        ctx.output_f32 = output_f32
        return o

    def backward(
        ctx,
        do: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None, None, None]:
        o, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        is_causal = ctx.is_causal
        local_window_size = ctx.local_window_size
        global_window_size = ctx.global_window_size
        gqa_interleave = ctx.gqa_interleave

        dq, dk, dv = _swa_torch_backward(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            is_causal,
            local_window_size,
            global_window_size,
            softmax_scale,
            gqa_interleave,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None

