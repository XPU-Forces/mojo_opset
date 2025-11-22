import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional
import torch_npu


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    i_hk = i_h // (H // HK)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    dh += (boh * H + i_h) * K * V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * HK + i_hk) * K
    w += (bos * HK + i_hk) * K
    do += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = HK * K
    stride_q = H * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        b_dh1 = (b_dh1 + b_dh1) / 2

        if K > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
            b_dh2 = (b_dh2 + b_dh2) / 2
        if K > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            bg_last = tl.load(g + (bos + last_idx) * HK + i_hk)
            bg_last_exp = tl.exp(bg_last)
            p_g = tl.make_block_ptr(g + bos * HK + i_hk, (T,), (HK,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = tl.exp(b_g)
        else:
            bg_last, b_g, b_g_exp, bg_last_exp = 0.0, 0.0, 1.0, 1.0

        p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))
        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))
        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))
        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, tl.exp(bg_last - b_g), 0.0)[:, None]

        b_dv += tl.load(p_dv, boundary_check=(0, 1))
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, stride_q), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))

        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]

        b_q = (b_q * scale).to(b_q.dtype)

        b_dh1 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_q), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh2 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_q), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh3 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_q), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh4 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_offsets: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    HK = k.shape[2]
    BT = 64

    if cu_seqlens is None:
        N, NT = B, triton.cdiv(T, BT)
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.zeros_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.zeros_like(dv)

    grid = lambda meta: (triton.cdiv(V, meta["BV"]), N * H)

    g_ptr = g if g is not None else q

    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g_ptr,
        dht=dht if dht is not None else q,
        dh0=dh0 if dh0 is not None else q,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        HK=HK,
        K=K,
        V=V,
        BT=BT,
        BV=64,
        USE_G=(g is not None),
        USE_INITIAL_STATE=(dh0 is not None),
        USE_FINAL_STATE_GRADIENT=(dht is not None),
        IS_VARLEN=(cu_seqlens is not None),
    )
    return dh, dh0, dv2


def chunk_gated_delta_rule_bwd_dhu_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: Optional[torch.Tensor],
    h0: Optional[torch.Tensor],
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
):
    """
    PyTorch Reference for the backward pass.
    Assumes fixed length (cu_seqlens is None) for verification simplicity.
    """
    B, T, H, K = q.shape
    _, _, HK, _ = k.shape
    _, _, _, V = do.shape

    NT = (T + chunk_size - 1) // chunk_size
    dh = torch.zeros(B, NT, H, K, V, device=q.device, dtype=torch.float32)
    dh0 = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.zeros_like(dv)

    for b in range(B):
        for h in range(H):
            hk = h // (H // HK)

            if dht is not None:
                d_state = dht[b, h].float().clone()
            else:
                d_state = torch.zeros(K, V, device=q.device, dtype=torch.float32)

            for i_t in range(NT - 1, -1, -1):
                t_start = i_t * chunk_size
                t_end = min(t_start + chunk_size, T)
                len_chunk = t_end - t_start

                q_c = q[b, t_start:t_end, h, :].float()
                k_c = k[b, t_start:t_end, hk, :].float()
                w_c = w[b, t_start:t_end, hk, :].float()
                do_c = do[b, t_start:t_end, h, :].float()
                dv_in_c = dv[b, t_start:t_end, h, :].float()

                if g is not None:
                    g_c = g[b, t_start:t_end, hk].float()

                    g_last = g[b, t_end - 1, hk].float()

                    decay_for_dv = torch.exp(g_last - g_c).unsqueeze(-1)

                    decay_for_dh = torch.exp(g_last)
                    q_scale = torch.exp(g_c).unsqueeze(-1)
                else:
                    decay_for_dv = 1.0
                    decay_for_dh = 1.0
                    q_scale = 1.0

                dh[b, i_t, h] = d_state

                dv_term = (k_c @ d_state) * decay_for_dv

                dv_out_c = dv_term + dv_in_c
                dv2[b, t_start:t_end, h] = dv_out_c

                d_state = d_state * decay_for_dh

                q_c_scaled = q_c * scale * q_scale

                d_state += q_c_scaled.transpose(0, 1) @ do_c

                d_state -= w_c.transpose(0, 1) @ dv_out_c

            if dh0 is not None:
                dh0[b, h] = d_state

    return dh, dh0, dv2


def test_chunk_gated_delta_rule_bwd():
    torch.manual_seed(42)

    B, T, H, K, V = 2, 128, 4, 32, 64
    HK = H
    chunk_size = 64
    dtype = torch.float32
    device = "npu"

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, HK, K, device=device, dtype=dtype)
    w = torch.randn(B, T, HK, K, device=device, dtype=dtype)
    g = torch.randn(B, T, HK, device=device, dtype=dtype).cumsum(dim=1)

    h0 = torch.zeros(B, H, K, V, device=device, dtype=dtype)
    dht = torch.randn(B, H, K, V, device=device, dtype=dtype)
    do = torch.randn(B, T, H, V, device=device, dtype=dtype)
    dv = torch.randn(B, T, H, V, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(K)

    print("Running Triton Kernel...")
    dh_tri, dh0_tri, dv2_tri = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=k, w=w, g=g, h0=h0, dht=dht, do=do, dv=dv, scale=scale, chunk_size=chunk_size
    )

    print("Running PyTorch Reference...")
    dh_ref, dh0_ref, dv2_ref = chunk_gated_delta_rule_bwd_dhu_ref(
        q=q, k=k, w=w, g=g, h0=h0, dht=dht, do=do, dv=dv, scale=scale, chunk_size=chunk_size
    )

    print("\nComparing results...")

    diff_dv = (dv2_tri - dv2_ref).abs().max().item()
    print(f"Max Diff dv2: {diff_dv}")

    diff_dh = (dh_tri - dh_ref).abs().max().item()
    print(f"Max Diff dh:  {diff_dh}")

    diff_dh0 = (dh0_tri - dh0_ref).abs().max().item()
    print(f"Max Diff dh0: {diff_dh0}")

    tol = 1e-4
    assert diff_dv < tol, "dv2 mismatch"
    assert diff_dh < tol, "dh mismatch"
    assert diff_dh0 < tol, "dh0 mismatch"

    print("\nTest Passed!")


if __name__ == "__main__":
    test_chunk_gated_delta_rule_bwd()
