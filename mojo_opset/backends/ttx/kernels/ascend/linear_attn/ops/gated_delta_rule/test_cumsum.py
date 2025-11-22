import torch
import triton
import triton.language as tl
from typing import Optional
import torch_npu


def chunk_local_cumsum_scalar_ref(
    g: torch.Tensor, chunk_size: int, reverse: bool = False, head_first: bool = False
) -> torch.Tensor:
    if head_first:
        g = g.transpose(1, 2).contiguous()

    B, T, H = g.shape

    assert T % chunk_size == 0, "For reference implementation, T must be divisible by chunk_size"
    g_reshaped = g.view(B, T // chunk_size, chunk_size, H)

    if reverse:
        out_reshaped = torch.cumsum(g_reshaped.flip(dims=[2]), dim=2).flip(dims=[2])
    else:
        out_reshaped = torch.cumsum(g_reshaped, dim=2)

    out = out_reshaped.view(B, T, H)

    if head_first:
        out = out.transpose(1, 2).contiguous()

    return out


@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    # if REVERSE:
    #     b_z = tl.sum(b_s, axis=0)
    #     b_o = -b_o + b_z[None] + b_s
    if REVERSE:
        offs = tl.arange(0, BT)
        mask = offs[:, None] <= offs[None, :]
        b_o = tl.sum(tl.where(mask, b_s[None, :], 0.0), axis=1)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.zeros_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        HAS_SCALE=False,
        IS_VARLEN=cu_seqlens is not None,
    )
    return g


def run_test():
    torch.manual_seed(42)

    B, T, H = 2, 128, 4
    chunk_size = 32
    dtype = torch.float32
    device = "npu"

    g = torch.randn((B, T, H), dtype=dtype, device=device)

    out_triton_fwd = chunk_local_cumsum_scalar(
        g=g, chunk_size=chunk_size, reverse=False, head_first=False, output_dtype=dtype
    )
    out_ref_fwd = chunk_local_cumsum_scalar_ref(g=g, chunk_size=chunk_size, reverse=False, head_first=False)

    assert torch.allclose(out_triton_fwd, out_ref_fwd, atol=1e-5)

    out_triton_rev = chunk_local_cumsum_scalar(
        g=g, chunk_size=chunk_size, reverse=True, head_first=False, output_dtype=dtype
    )
    out_ref_rev = chunk_local_cumsum_scalar_ref(g=g, chunk_size=chunk_size, reverse=True, head_first=False)

    assert torch.allclose(out_triton_rev, out_ref_rev, atol=1e-5)


if __name__ == "__main__":
    run_test()
