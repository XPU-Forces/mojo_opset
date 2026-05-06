from typing import List
from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl


def _get_num_programs():
    """Get number of programs for triton kernel launch. Backend agnostic."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).multi_processor_count
    elif hasattr(torch, "npu") and torch.npu.is_available():
        try:
            return torch.npu.get_device_properties(0).core_count
        except Exception:
            return 64
    else:
        return 64


@triton.jit
def _triton_mrope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    """
    Triton kernel for Multimodal RoPE (MRoPE).

    Supports:
    - Flat input tensors from vLLM style: q [num_tokens, n_qh * head_dim], k [num_tokens, n_kh * head_dim]
    - cos/sin with shape [3, num_tokens, head_dim // 2] (T/H/W positions for multimodal inputs)

    Args:
        q_ptr: pointer to q tensor
        k_ptr: pointer to k tensor
        cos_ptr: pointer to cos tensor [3, num_tokens, head_dim // 2]
        sin_ptr: pointer to sin tensor [3, num_tokens, head_dim // 2]
        num_tokens: number of tokens
        n_qh: number of query heads
        n_kh: number of key heads
        hd: head dimension
        rope_dim: rotary dimension
        pad_n_qh: padded number of query heads (power of 2)
        pad_n_kh: padded number of key heads (power of 2)
        pad_hd: padded head dimension (power of 2)
        mrope_section_t: time section size
        mrope_section_h: height section size
        mrope_section_w: width section size
        is_interleaved: whether T/H/W positions are interleaved
    """
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * n_qh * hd
    k_ptr = k_ptr + pid * n_kh * hd

    half_rope_dim = rope_dim // 2
    half_rd = half_rope_dim
    stride_token = half_rd

    t_cos = cos_ptr + pid * stride_token
    h_cos = t_cos + num_tokens * stride_token
    w_cos = h_cos + num_tokens * stride_token
    t_sin = sin_ptr + pid * stride_token
    h_sin = t_sin + num_tokens * stride_token
    w_sin = h_sin + num_tokens * stride_token

    cos_offsets = tl.arange(0, pad_hd // 2)

    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    cos_mask = cos_offsets < half_rd
    t_cos_row = tl.load(t_cos + cos_offsets, mask=cos_mask, other=0.0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=cos_mask, other=0.0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=cos_mask, other=0.0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=cos_mask, other=0.0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=cos_mask, other=0.0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=cos_mask, other=0.0)

    t_cos_row = tl.where(t_mask, t_cos_row, 0.0)
    h_cos_row = tl.where(h_mask, h_cos_row, 0.0)
    w_cos_row = tl.where(w_mask, w_cos_row, 0.0)
    t_sin_row = tl.where(t_mask, t_sin_row, 0.0)
    h_sin_row = tl.where(h_mask, h_sin_row, 0.0)
    w_sin_row = tl.where(w_mask, w_sin_row, 0.0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]

    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < half_rd)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < half_rd)

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask)

    second_half_q_offsets = first_half_q_offsets + half_rd
    second_half_k_offsets = first_half_k_offsets + half_rd

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=first_q_mask)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=first_k_mask)

    cos_row = cos_row.to(q_tile_1.dtype)
    sin_row = sin_row.to(q_tile_1.dtype)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row

    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row

    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)


def mrope_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    is_interleaved: bool = False,
    head_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Multimodal RoPE (MRoPE) to q and k tensors using Triton kernel.

    Args:
        q: [num_tokens, n_qh * head_dim] tensor
        k: [num_tokens, n_kh * head_dim] tensor
        cos: [3, num_tokens, rotary_dim // 2] cos values for T/H/W dimensions
        sin: [3, num_tokens, rotary_dim // 2] sin values for T/H/W dimensions
        mrope_section: [t_section, h_section, w_section] - how half rope_dim is split
        is_interleaved: if True, T/H/W positions are interleaved
        head_dim: head dimension. If None, inferred from cos (assumes rope_dim == head_dim).

    Returns:
        (q, k) with RoPE applied
    """
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    num_tokens, n_qh_hd = q.shape
    rope_dim = sum(mrope_section) * 2

    if head_dim is None:
        head_dim = cos.shape[-1] * 2
        if head_dim != rope_dim:
            raise ValueError(
                f"head_dim ({head_dim}) inferred from cos does not match "
                f"rope_dim ({rope_dim}). Please pass head_dim explicitly."
            )

    n_qh = n_qh_hd // head_dim
    n_kh = k.shape[1] // head_dim

    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)

    n_row = num_tokens

    _triton_mrope_kernel[(n_row,)](
        q,
        k,
        cos,
        sin,
        num_tokens,
        n_qh,
        n_kh,
        head_dim,
        rope_dim,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        is_interleaved,
    )

    return q, k


def apply_interleaved_mrope(x: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
    """
    Apply interleaved MRoPE to cos/sin tensors.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to interleaved [THWHW...].

    Args:
        x: [3, num_tokens, head_dim // 2] tensor
        mrope_section: [t, h, w] sections

    Returns:
        Reorganized tensor [num_tokens, head_dim // 2]
    """
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


def mrope_native(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    is_interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Native PyTorch implementation of MRoPE for reference.

    Args:
        q: [num_tokens, n_qh * head_dim] tensor
        k: [num_tokens, n_kh * head_dim] tensor
        cos: [3, num_tokens, head_dim // 2] or [num_tokens, sum(mrope_section)] tensor
        sin: [3, num_tokens, head_dim // 2] or [num_tokens, sum(mrope_section)] tensor
        mrope_section: [t_section, h_section, w_section]
        is_interleaved: if True, T/H/W positions are interleaved

    Returns:
        (q, k) with RoPE applied
    """
    num_tokens, n_qh_hd = q.shape
    head_dim = cos.shape[-1] * 2
    rope_dim = sum(mrope_section) * 2
    half_rope_dim = rope_dim // 2

    q = q.view(num_tokens, -1, head_dim)
    k = k.view(num_tokens, -1, head_dim)

    q_rot = q[..., :rope_dim]
    q_pass = q[..., rope_dim:]
    k_rot = k[..., :rope_dim]
    k_pass = k[..., rope_dim:]

    if cos.dim() == 3:
        if is_interleaved:
            cos = apply_interleaved_mrope(cos, mrope_section)
            sin = apply_interleaved_mrope(sin, mrope_section)
        else:
            cos = torch.cat([m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
            sin = torch.cat([m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)

    cos = cos.view(num_tokens, half_rope_dim)
    sin = sin.view(num_tokens, half_rope_dim)

    q_rot_half1 = q_rot[..., :half_rope_dim]
    q_rot_half2 = q_rot[..., half_rope_dim:]
    k_rot_half1 = k_rot[..., :half_rope_dim]
    k_rot_half2 = k_rot[..., half_rope_dim:]

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_rot_new_half1 = q_rot_half1 * cos - q_rot_half2 * sin
    q_rot_new_half2 = q_rot_half2 * cos + q_rot_half1 * sin
    k_rot_new_half1 = k_rot_half1 * cos - k_rot_half2 * sin
    k_rot_new_half2 = k_rot_half2 * cos + k_rot_half1 * sin

    q_rot = torch.cat([q_rot_new_half1, q_rot_new_half2], dim=-1)
    k_rot = torch.cat([k_rot_new_half1, k_rot_new_half2], dim=-1)

    q = torch.cat([q_rot, q_pass], dim=-1).view(num_tokens, -1)
    k = torch.cat([k_rot, k_pass], dim=-1).view(num_tokens, -1)

    return q, k


def compute_cos_sin_cache(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin cache for rotary position embedding.

    Args:
        head_dim: total head dimension
        rotary_dim: rotary dimension
        max_position: maximum position
        base: base for frequency computation

    Returns:
        (cos_cache, sin_cache) tensors
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim // 2, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_position, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = freqs.repeat_interleave(2, dim=-1)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def prepare_mrope_cos_sin(
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    mrope_section: List[int],
    head_dim: int,
    is_interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare cos/sin tensors for MRoPE from position indices.

    Args:
        positions: [3, num_tokens] position indices for T/H/W
        cos_cache: [max_position, rotary_dim // 2] cos cache
        sin_cache: [max_position, rotary_dim // 2] sin cache
        mrope_section: [t, h, w] sections
        head_dim: head dimension
        is_interleaved: whether positions are interleaved

    Returns:
        (cos, sin) tensors with shape [3, num_tokens, head_dim // 2]
    """
    num_tokens = positions.shape[-1]
    rotary_dim = sum(mrope_section) * 2
    half_rotary_dim = rotary_dim // 2

    cos_3d = torch.zeros(3, num_tokens, half_rotary_dim, device=positions.device, dtype=torch.float32)
    sin_3d = torch.zeros(3, num_tokens, half_rotary_dim, device=positions.device, dtype=torch.float32)

    for dim_idx in range(3):
        pos = positions[dim_idx]
        cos_3d[dim_idx] = cos_cache[pos][:, :half_rotary_dim]
        sin_3d[dim_idx] = sin_cache[pos][:, :half_rotary_dim]

    return cos_3d, sin_3d
