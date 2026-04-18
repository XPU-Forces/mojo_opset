#!/usr/bin/env python3
import torch
import triton
import triton.language as tl

torch.random.manual_seed(42)


def compute_cos_sin_cache(head_dim, rotary_dim, max_position, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim // 2, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_position, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = freqs.repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


def adjust_mrope_section(mrope_section, actual_rotary_dim):
    mrope_section_adjusted = []
    remaining = actual_rotary_dim // 2
    for i, s in enumerate(mrope_section):
        if i == len(mrope_section) - 1:
            mrope_section_adjusted.append(remaining)
        else:
            adj_s = min(s, remaining)
            mrope_section_adjusted.append(adj_s)
            remaining -= adj_s
    return mrope_section_adjusted


@triton.jit
def _mrope_kernel(
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
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * n_qh * hd
    k_ptr = k_ptr + pid * n_kh * hd
    half_rope_dim = rope_dim // 2

    t_cos = cos_ptr + pid * half_rope_dim
    h_cos = t_cos + num_tokens * half_rope_dim
    w_cos = h_cos + num_tokens * half_rope_dim
    t_sin = sin_ptr + pid * half_rope_dim
    h_sin = t_sin + num_tokens * half_rope_dim
    w_sin = h_sin + num_tokens * half_rope_dim

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
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rope_dim)

    t_cos_row = tl.load(t_cos + cos_offsets)
    h_cos_row = tl.load(h_cos + cos_offsets)
    w_cos_row = tl.load(w_cos + cos_offsets)
    t_sin_row = tl.load(t_sin + cos_offsets)
    h_sin_row = tl.load(h_sin + cos_offsets)
    w_sin_row = tl.load(w_sin + cos_offsets)

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
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < half_rope_dim)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < half_rope_dim)

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask)

    second_half_q_offsets = first_half_q_offsets + half_rope_dim
    second_half_k_offsets = first_half_k_offsets + half_rope_dim
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


def triton_mrope(q, k, cos, sin, mrope_section, is_interleaved=False):
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    num_tokens, n_qh_hd = q.shape
    half_head_dim = cos.shape[-1]
    n_qh = n_qh_hd // (half_head_dim * 2)
    n_kh = k.shape[1] // (half_head_dim * 2)
    head_dim = half_head_dim * 2
    rope_dim = sum(mrope_section) * 2
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    n_row = min(num_tokens, 64)
    _mrope_kernel[(n_row,)](
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


def native_mrope(q, k, cos, sin, mrope_section, is_interleaved=False):
    # cos shape: (3*num_tokens, half_head_dim) = [T0..T(n-1), H0..H(n-1), W0..W(n-1)]
    num_tokens = cos.shape[0] // 3
    half_head_dim = cos.shape[-1]
    head_dim = half_head_dim * 2
    rope_dim = sum(mrope_section) * 2
    half_rope_dim = rope_dim // 2

    q = q.view(num_tokens, -1, head_dim)
    k = k.view(num_tokens, -1, head_dim)
    q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]

    # Extract T, H, W for all tokens
    cos_T = cos[:num_tokens]  # (num_tokens, half_head_dim)
    cos_H = cos[num_tokens : num_tokens * 2]
    cos_W = cos[num_tokens * 2 : num_tokens * 3]
    sin_T = sin[:num_tokens]
    sin_H = sin[num_tokens : num_tokens * 2]
    sin_W = sin[num_tokens * 2 : num_tokens * 3]

    if is_interleaved:
        # Interleaved layout: reorganize within each section
        cos_cat = torch.zeros(num_tokens, half_rope_dim, device=cos.device, dtype=cos.dtype)
        sin_cat = torch.zeros(num_tokens, half_rope_dim, device=sin.device, dtype=sin.dtype)
        cos_cat[..., : mrope_section[0]] = cos_T[..., : mrope_section[0]]
        cos_cat[..., mrope_section[0] : mrope_section[0] + mrope_section[1]] = cos_H[..., : mrope_section[1]]
        cos_cat[..., mrope_section[0] + mrope_section[1] :] = cos_W[..., : mrope_section[2]]
        sin_cat[..., : mrope_section[0]] = sin_T[..., : mrope_section[0]]
        sin_cat[..., mrope_section[0] : mrope_section[0] + mrope_section[1]] = sin_H[..., : mrope_section[1]]
        sin_cat[..., mrope_section[0] + mrope_section[1] :] = sin_W[..., : mrope_section[2]]
    else:
        # Non-interleaved: T section, then H section, then W section
        # Each token's cos is [T_i[:mrope_section[0]], H_i[:mrope_section[1]], W_i[:mrope_section[2]]]
        cos_cat = torch.zeros(num_tokens, half_rope_dim, device=cos.device, dtype=cos.dtype)
        sin_cat = torch.zeros(num_tokens, half_rope_dim, device=sin.device, dtype=sin.dtype)
        cos_cat[..., : mrope_section[0]] = cos_T[..., : mrope_section[0]]
        cos_cat[..., mrope_section[0] : mrope_section[0] + mrope_section[1]] = cos_H[..., : mrope_section[1]]
        cos_cat[..., mrope_section[0] + mrope_section[1] :] = cos_W[..., : mrope_section[2]]
        sin_cat[..., : mrope_section[0]] = sin_T[..., : mrope_section[0]]
        sin_cat[..., mrope_section[0] : mrope_section[0] + mrope_section[1]] = sin_H[..., : mrope_section[1]]
        sin_cat[..., mrope_section[0] + mrope_section[1] :] = sin_W[..., : mrope_section[2]]

    q_rot_half1, q_rot_half2 = q_rot[..., :half_rope_dim], q_rot[..., half_rope_dim:]
    k_rot_half1, k_rot_half2 = k_rot[..., :half_rope_dim], k_rot[..., half_rope_dim:]
    cos_cat = cos_cat.unsqueeze(1)
    sin_cat = sin_cat.unsqueeze(1)
    q_rot_new_half1 = q_rot_half1 * cos_cat - q_rot_half2 * sin_cat
    q_rot_new_half2 = q_rot_half2 * cos_cat + q_rot_half1 * sin_cat
    k_rot_new_half1 = k_rot_half1 * cos_cat - k_rot_half2 * sin_cat
    k_rot_new_half2 = k_rot_half2 * cos_cat + k_rot_half1 * sin_cat
    q_rot = torch.cat([q_rot_new_half1, q_rot_new_half2], dim=-1)
    k_rot = torch.cat([k_rot_new_half1, k_rot_new_half2], dim=-1)
    q = torch.cat([q_rot, q_pass], dim=-1).view(num_tokens, -1)
    k = torch.cat([k_rot, k_pass], dim=-1).view(num_tokens, -1)
    return q, k


print("Testing MRoPE...")
device = "npu"
num_tokens, n_qh, n_kh, head_dim = 32, 16, 8, 128
mrope_section = [16, 24, 24]
rotary_dim = sum(mrope_section) * 2
actual_rotary_dim = int(head_dim * rotary_dim / head_dim)
mrope_section_adj = adjust_mrope_section(mrope_section, actual_rotary_dim)
positions = torch.randint(0, 1000, (3, num_tokens), device=device, dtype=torch.long)
cos_cache, sin_cache = compute_cos_sin_cache(head_dim, rotary_dim, 4000)
cos_cache = cos_cache.to(device)
sin_cache = sin_cache.to(device)
half_head_dim = actual_rotary_dim // 2

# Layout: [T0..T(n-1), H0..H(n-1), W0..W(n-1)]
cos_T = cos_cache[positions[0]][:, :half_head_dim]
cos_H = cos_cache[positions[1]][:, :half_head_dim]
cos_W = cos_cache[positions[2]][:, :half_head_dim]
sin_T = sin_cache[positions[0]][:, :half_head_dim]
sin_H = sin_cache[positions[1]][:, :half_head_dim]
sin_W = sin_cache[positions[2]][:, :half_head_dim]
cos_flat = torch.cat([cos_T, cos_H, cos_W], dim=0)
sin_flat = torch.cat([sin_T, sin_H, sin_W], dim=0)

print(f"cos_flat.shape={cos_flat.shape}")

q = torch.randn(num_tokens, n_qh * head_dim, device=device, dtype=torch.float32)
k = torch.randn(num_tokens, n_kh * head_dim, device=device, dtype=torch.float32)

q_triton, k_triton = triton_mrope(q.clone(), k.clone(), cos_flat, sin_flat, mrope_section_adj)
q_native, k_native = native_mrope(q.clone(), k.clone(), cos_flat, sin_flat, mrope_section_adj)

print("Comparing Q (Triton vs Native)...")
try:
    torch.testing.assert_close(q_triton, q_native, atol=1e-2, rtol=1e-2)
    print("  Q: PASS")
except Exception as e:
    print(f"  Q: FAIL")
    diff = (q_triton - q_native).abs()
    print(f"  Max diff: {diff.max()}")

print("Comparing K (Triton vs Native)...")
try:
    torch.testing.assert_close(k_triton, k_native, atol=1e-2, rtol=1e-2)
    print("  K: PASS")
except Exception as e:
    print(f"  K: FAIL")
    diff = (k_triton - k_native).abs()
    print(f"  Max diff: {diff.max()}")

print("All tests completed!")
