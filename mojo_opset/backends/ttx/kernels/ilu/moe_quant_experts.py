from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import libentry, smart_triton_autotune


# ---------------------------------------------------------------------------
# Triton: grouped int8 matmul with per-group weight scales and per-token input scales.
#
# Int8 matmul uses per-K rank-1 int32 accumulation, not tl.dot: ILU Triton
# int8 tl.dot can fail LLVM layout conversion / segfault.
#
# EPILOGUE enum selects post-matmul activation:
#   EPILOGUE_NONE   – plain dequant output
#   EPILOGUE_SWIGLU – fused SwiGLU (B has 2*HALF_N columns: gate + up)
# ---------------------------------------------------------------------------

EPILOGUE_NONE = 0
EPILOGUE_SWIGLU = 1


def _quant_moe_autotune_configs():
    configs = []
    for BM, BN, nw in [
        (32, 32, 4), (32, 64, 4), (64, 32, 4),
        (64, 64, 4), (64, 128, 4), (128, 64, 4),
    ]:
        for ns in [2, 3]:
            configs.append(triton.Config(
                {"BLOCK_M": BM, "BLOCK_N": BN},
                num_warps=nw, num_stages=ns,
            ))
    return configs


@smart_triton_autotune(configs=_quant_moe_autotune_configs(), selected_idx=0, key=["N", "K", "MAX_M"])
@libentry()
@triton.jit
def _quant_moe_gemm_kernel(
    A,
    B,
    C,
    input_scale_ptr,
    weight_scale_ptr,
    group_offsets_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_M,
    stride_bg,
    strideBN,
    strideBK,
    stride_ws_g,
    stride_ws_n,
    stride_ws_k,
    QUANT_GROUP_SIZE: tl.constexpr,
    NUM_GROUPS_K: tl.constexpr,
    EPILOGUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Grouped int8 matmul with per-group dequant and optional epilogue.

    EPILOGUE_NONE:   output = dequant(A @ B.T)
    EPILOGUE_SWIGLU: B has N columns (gate + up), output HALF_N = N//2 columns
                     after silu(gate) * up.
    """
    n_tile_id = tl.program_id(0)
    m_tile_id = tl.program_id(1)
    group_id = tl.program_id(2)

    if m_tile_id * BLOCK_M >= MAX_M:
        return

    group_start = tl.load(group_offsets_ptr + group_id).to(tl.int32)
    group_end = tl.load(group_offsets_ptr + group_id + 1).to(tl.int32)
    m_g = group_end - group_start

    if m_tile_id * BLOCK_M >= m_g:
        return

    HALF_N: tl.constexpr = N // 2

    offs_m = group_start + m_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < group_end

    b_base = B + group_id * stride_bg
    ws_base = weight_scale_ptr + group_id * stride_ws_g

    _SWIGLU: tl.constexpr = 1

    if EPILOGUE == _SWIGLU:
        out_N = HALF_N
        offs_n_up = offs_n + HALF_N
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    else:
        out_N = N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kg in range(NUM_GROUPS_K):
        k_start = kg * QUANT_GROUP_SIZE

        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        if EPILOGUE == _SWIGLU:
            partial_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k_idx in range(k_start, k_start + QUANT_GROUP_SIZE):
            if k_idx < K:
                a_col = tl.load(A + offs_m * K + k_idx, mask=mask_m, other=0)
                b_row = tl.load(
                    b_base + offs_n * strideBN + k_idx * strideBK,
                    mask=offs_n < out_N, other=0,
                )
                partial += a_col.to(tl.int32)[:, None] * b_row.to(tl.int32)[None, :]

                if EPILOGUE == _SWIGLU:
                    bu_row = tl.load(
                        b_base + offs_n_up * strideBN + k_idx * strideBK,
                        mask=offs_n_up < N, other=0,
                    )
                    partial_up += a_col.to(tl.int32)[:, None] * bu_row.to(tl.int32)[None, :]

        ws = tl.load(
            ws_base + offs_n * stride_ws_n + kg * stride_ws_k,
            mask=offs_n < out_N, other=0.0,
        )
        acc += partial.to(tl.float32) * ws[None, :]

        if EPILOGUE == _SWIGLU:
            ws_up = tl.load(
                ws_base + offs_n_up * stride_ws_n + kg * stride_ws_k,
                mask=offs_n_up < N, other=0.0,
            )
            acc_up += partial_up.to(tl.float32) * ws_up[None, :]

    i_scale = tl.load(input_scale_ptr + offs_m, mask=mask_m, other=1.0)
    acc *= i_scale[:, None]

    if EPILOGUE == _SWIGLU:
        acc_up *= i_scale[:, None]
        gate_f = acc.to(tl.bfloat16).to(tl.float32)
        up_f = acc_up.to(tl.bfloat16).to(tl.float32)
        result = (gate_f * tl.sigmoid(gate_f)) * up_f
    else:
        result = acc

    c = result.to(C.dtype.element_ty)
    c_ptrs = C + offs_m[:, None] * out_N + offs_n[None, :]
    c_mask = mask_m[:, None] & (offs_n[None, :] < out_N)
    tl.store(c_ptrs, c, mask=c_mask)


def _prepare_quant_gemm_args(A, input_scale, weight_scale, size_per_group, num_groups, K, quant_group_size):
    if quant_group_size <= 0:
        quant_group_size = K
    num_groups_k = (K + quant_group_size - 1) // quant_group_size

    if weight_scale.ndim == 2:
        weight_scale = weight_scale.unsqueeze(-1)
    input_scale_flat = input_scale.reshape(-1).float()

    cum = size_per_group.cumsum(0, dtype=torch.int32)
    group_offsets = torch.empty(num_groups + 1, dtype=torch.int32, device=A.device)
    group_offsets[0] = 0
    group_offsets[1:] = cum
    max_m = size_per_group.max().item()

    return quant_group_size, num_groups_k, weight_scale, input_scale_flat, group_offsets, max_m


def _quant_m_grouped_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    size_per_group: torch.Tensor,
    num_groups: int,
    M: int,
    N: int,
    K: int,
    quant_group_size: int,
) -> torch.Tensor:
    quant_group_size, num_groups_k, weight_scale, input_scale_flat, group_offsets, max_m = \
        _prepare_quant_gemm_args(A, input_scale, weight_scale, size_per_group, num_groups, K, quant_group_size)

    def grid(META):
        return (
            triton.cdiv(N, META["BLOCK_N"]),
            triton.cdiv(max_m, META["BLOCK_M"]),
            num_groups,
        )

    _quant_moe_gemm_kernel[grid](
        A, B, C,
        input_scale_flat,
        weight_scale,
        group_offsets,
        N, K, max_m,
        B.stride(0), B.stride(1), B.stride(2),
        weight_scale.stride(0), weight_scale.stride(1), weight_scale.stride(2),
        quant_group_size, num_groups_k,
        EPILOGUE_NONE,
    )
    return C


def _quant_m_grouped_matmul_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    size_per_group: torch.Tensor,
    num_groups: int,
    M: int,
    N: int,
    K: int,
    quant_group_size: int,
) -> torch.Tensor:
    half_n = N // 2
    quant_group_size, num_groups_k, weight_scale, input_scale_flat, group_offsets, max_m = \
        _prepare_quant_gemm_args(A, input_scale, weight_scale, size_per_group, num_groups, K, quant_group_size)

    def grid(META):
        return (
            triton.cdiv(half_n, META["BLOCK_N"]),
            triton.cdiv(max_m, META["BLOCK_M"]),
            num_groups,
        )

    _quant_moe_gemm_kernel[grid](
        A, B, C,
        input_scale_flat,
        weight_scale,
        group_offsets,
        N, K, max_m,
        B.stride(0), B.stride(1), B.stride(2),
        weight_scale.stride(0), weight_scale.stride(1), weight_scale.stride(2),
        quant_group_size, num_groups_k,
        EPILOGUE_SWIGLU,
    )
    return C


# ---------------------------------------------------------------------------
# Triton: per-expert smooth + per-token dynamic int8 quantization
# ---------------------------------------------------------------------------

@libentry()
@triton.jit
def _moe_smooth_dynamic_quant_kernel(
    input_ptr,
    smooth_ptr,
    output_ptr,
    scale_ptr,
    group_offsets_ptr,
    total_tokens,
    K: tl.constexpr,
    stride_smooth_e,
    BLOCK_K: tl.constexpr,
):
    """Per-token: output_int8 = round(clamp(input * smooth / scale)), scale = absmax / 127.

    smooth is indexed by expert via group_offsets (no repeat_interleave needed).
    """
    row = tl.program_id(0)
    if row >= total_tokens:
        return

    num_groups = tl.num_programs(1)
    expert_id = tl.program_id(1)

    lo = tl.load(group_offsets_ptr + expert_id).to(tl.int32)
    hi = tl.load(group_offsets_ptr + expert_id + 1).to(tl.int32)

    actual_row = lo + row
    if actual_row >= hi:
        return

    max_abs = 0.0

    for tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < K

        inp = tl.load(input_ptr + actual_row * K + offs_k, mask=mask, other=0.0).to(tl.float32)
        sm = tl.load(smooth_ptr + expert_id * stride_smooth_e + offs_k, mask=mask, other=1.0).to(tl.float32)
        val = inp * sm

        tile_max = tl.max(tl.abs(val))
        max_abs = tl.where(tile_max > max_abs, tile_max, max_abs)

    RCP_127: tl.constexpr = 1.0 / 127.0
    q_scale = max_abs * RCP_127
    q_scale = tl.where(q_scale < 1e-6, 1.0, q_scale)
    rcp_scale = 1.0 / q_scale

    tl.store(scale_ptr + actual_row, q_scale)

    for tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < K

        inp = tl.load(input_ptr + actual_row * K + offs_k, mask=mask, other=0.0).to(tl.float32)
        sm = tl.load(smooth_ptr + expert_id * stride_smooth_e + offs_k, mask=mask, other=1.0).to(tl.float32)
        val = inp * sm

        quant_val = val * rcp_scale
        quant_val = tl.where(quant_val < 0, quant_val - 0.5, quant_val + 0.5)
        quant_val = tl.clamp(quant_val, -127.0, 127.0)

        tl.store(output_ptr + actual_row * K + offs_k, quant_val.to(tl.int8), mask=mask)


def _moe_smooth_dynamic_quant(
    input_tensor: torch.Tensor,
    inv_smooth_scale: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton: grouped smooth + per-token dynamic int8 quantization.

    input_tensor: [total_tokens, K] bf16/fp32
    inv_smooth_scale: [num_experts, K] (1 / smooth_scale)
    tokens_per_expert: [num_experts] int32
    Returns: (int8_output [total_tokens, K], scale [total_tokens, 1] fp32)
    """
    total_tokens, K = input_tensor.shape
    device = input_tensor.device

    output = torch.empty(total_tokens, K, dtype=torch.int8, device=device)
    scale = torch.empty(total_tokens, dtype=torch.float32, device=device)

    cum = tokens_per_expert.cumsum(0, dtype=torch.int32)
    group_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    group_offsets[0] = 0
    group_offsets[1:] = cum

    max_tokens = int(tokens_per_expert.max().item())
    if max_tokens == 0:
        return output, scale.unsqueeze(-1)

    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K > 4096:
        BLOCK_K = 4096

    grid = (max_tokens, num_experts)
    _moe_smooth_dynamic_quant_kernel[grid](
        input_tensor,
        inv_smooth_scale.float(),
        output,
        scale,
        group_offsets,
        total_tokens,
        K,
        inv_smooth_scale.stride(0),
        BLOCK_K=BLOCK_K,
    )
    return output, scale.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Triton: int4 unpack (2 x int4 packed in 1 x int8)
# ---------------------------------------------------------------------------

@libentry()
@triton.jit
def _unpack_int4_kernel(
    packed_ptr,
    out_ptr,
    num_packed_rows,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Unpack int4: packed[i, k] -> out[2*i, k] (low 4 bits), out[2*i+1, k] (high 4 bits)."""
    row = tl.program_id(0)
    if row >= num_packed_rows:
        return

    for tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < K

        packed = tl.load(packed_ptr + row * K + offs_k, mask=mask, other=0).to(tl.uint8)

        low = (packed & 0x0F).to(tl.int8)
        high = ((packed >> 4) & 0x0F).to(tl.int8)
        low = tl.where(low >= 8, low - 16, low)
        high = tl.where(high >= 8, high - 16, high)

        out_row_low = row * 2
        out_row_high = row * 2 + 1
        tl.store(out_ptr + out_row_low * K + offs_k, low, mask=mask)
        tl.store(out_ptr + out_row_high * K + offs_k, high, mask=mask)


def _unpack_int4_weight(packed: torch.Tensor) -> torch.Tensor:
    """Triton: unpack [num_experts, N//2, K] int8 -> [num_experts, N, K] int8."""
    num_experts, packed_n, K = packed.shape
    out = torch.empty(num_experts, packed_n * 2, K, dtype=torch.int8, device=packed.device)

    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K > 4096:
        BLOCK_K = 4096

    for e in range(num_experts):
        _unpack_int4_kernel[(packed_n,)](
            packed[e],
            out[e],
            packed_n,
            K,
            BLOCK_K=BLOCK_K,
        )
    return out


def _cached_unpacked_proj_weight(module, attr: str) -> torch.Tensor:
    """Return int8 expert weight; unpack int4 once and cache until weights are reloaded."""
    packed = getattr(module, attr)
    w_dtype = module.up_weight_dtype if "up_" in attr else module.down_weight_dtype
    if w_dtype != "int4":
        return packed
    cache_key = f"_ttx_{attr}_unpacked_int8"
    ver_key = f"{cache_key}_version"
    version = (packed.data_ptr(), tuple(packed.shape), tuple(packed.stride()))
    if getattr(module, ver_key, None) != version:
        setattr(module, cache_key, _unpack_int4_weight(packed))
        setattr(module, ver_key, version)
    return getattr(module, cache_key)


def clear_quant_moe_weight_unpack_cache(module) -> None:
    """Drop cached int4 unpack buffers (call from load_state_dict)."""
    for attr in ("up_proj_weight", "down_proj_weight"):
        for key in (f"_ttx_{attr}_unpacked_int8", f"_ttx_{attr}_unpacked_int8_version"):
            if hasattr(module, key):
                delattr(module, key)


# ---------------------------------------------------------------------------
# Main entry: quant MoE experts (all-Triton, no torch operator calls)
# ---------------------------------------------------------------------------

def quant_moe_experts_impl(
    module, sorted_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor
) -> torch.Tensor:
    """ILU Triton: quantized MoE experts.

    Full pipeline without any torch operator calls:
      1. smooth + dynamic int8 quant (Triton)
      2. int8 grouped GEMM with fused SwiGLU + per-group dequant (Triton)
      3. smooth + dynamic int8 quant (Triton)
      4. int8 grouped GEMM + per-group dequant (Triton)
    """
    num_experts = tokens_per_expert.size(0)
    t_tokens = sorted_hidden_states.shape[0]
    dtype = sorted_hidden_states.dtype
    device = sorted_hidden_states.device

    # --- Step 1: smooth + dynamic quant for up_proj input ---
    x_int8, x_scale = _moe_smooth_dynamic_quant(
        sorted_hidden_states,
        module.up_proj_quantize.inv_smooth_scale,
        tokens_per_expert,
        num_experts,
    )

    up_w = _cached_unpacked_proj_weight(module, "up_proj_weight")
    up_ws = module.up_proj_weight_scale

    _, n_up, k_in = up_w.shape
    inter = module.intermediate_size

    # --- Step 2: int8 grouped GEMM + SwiGLU epilogue ---
    # Use fp32 output to match core precision chain (core keeps activated in fp32
    # before passing to down_proj_quantize).
    fc1_out = torch.empty(t_tokens, inter, device=device, dtype=torch.float32)
    _quant_m_grouped_matmul_swiglu(
        x_int8,
        up_w,
        fc1_out,
        x_scale,
        up_ws,
        tokens_per_expert,
        num_experts,
        t_tokens,
        n_up,
        k_in,
        module.up_quant_group_size,
    )

    # --- Step 3: smooth + dynamic quant for down_proj input ---
    y_int8, y_scale = _moe_smooth_dynamic_quant(
        fc1_out,
        module.down_proj_quantize.inv_smooth_scale,
        tokens_per_expert,
        num_experts,
    )

    down_w = _cached_unpacked_proj_weight(module, "down_proj_weight")
    down_ws = module.down_proj_weight_scale

    _, h_out, k_inter = down_w.shape

    # --- Step 4: int8 grouped GEMM ---
    out = torch.empty(t_tokens, h_out, device=device, dtype=dtype)
    _quant_m_grouped_matmul(
        y_int8,
        down_w,
        out,
        y_scale,
        down_ws,
        tokens_per_expert,
        num_experts,
        t_tokens,
        h_out,
        k_inter,
        module.down_quant_group_size,
    )
    return out
