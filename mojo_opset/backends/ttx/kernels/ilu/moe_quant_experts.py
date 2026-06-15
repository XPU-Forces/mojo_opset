from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import libentry, smart_triton_autotune


# ---------------------------------------------------------------------------
# Triton: grouped int8 matmul with per-group weight scales and per-token input scales.
#
# Int8 matmul uses tl.dot on the matrix engine. ILU's int8 tl.dot still miscompiles
# (SharedToDotOperand lowering emits invalid <2xf32><->4xi8> bitcasts -> segfault in
# make_llir), so the int8 operands (|v| <= 127) are cast losslessly to fp16 and fed to
# an fp16 MMA. The int8->fp16 cast is lossless (|v| <= 127), and ILU's dot does a true
# fp32 multiply-accumulate (verified: dot of two fp16 vectors [64,1] returns exactly 4097,
# which fp16 intra-dot accumulation would round to 4096; large-magnitude sums that exceed
# the fp16 max of 65504 also come back finite/exact rather than inf). Each BLOCK_K tile
# (partial sum <= 128*127^2 << 2^24) is thus computed exactly in the fp32 dot output,
# rounded back to int32, and accumulated into an int32
# partial (so the per-group total, which can exceed 2^24 for large groups, stays exact);
# the int32 partial is then dequantized by the per-group weight scale. An autotune
# prune (_prune_block_k_gt_group) keeps BLOCK_K <= QUANT_GROUP_SIZE so a tile never spans
# group boundaries. The weight tile is loaded as [BLOCK_N, BLOCK_K] (B is row-major
# [N, K]) and transposed via tl.trans before the dot.
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
        for BK in [32, 64, 128]:
            for ns in [2, 3]:
                configs.append(triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                    num_warps=nw, num_stages=ns,
                ))
    return configs


def _prune_block_k_gt_group(configs, named_args, **kwargs):
    """Drop configs whose BLOCK_K exceeds the quant group size.

    A BLOCK_K larger than QUANT_GROUP_SIZE is numerically correct (the
    ``k_in_group < QUANT_GROUP_SIZE`` mask zeros the out-of-group lanes) but
    wastes up to BLOCK_K/QUANT_GROUP_SIZE of the MMA work on masked lanes, so
    such configs must never be picked by the autotuner. QUANT_GROUP_SIZE is
    already normalized to a positive value (``<= 0`` -> K) before launch.
    """
    qgs = named_args.get("QUANT_GROUP_SIZE") or kwargs.get("QUANT_GROUP_SIZE")
    if not qgs or qgs <= 0:
        return list(configs)
    kept = [c for c in configs if c.kwargs.get("BLOCK_K", 1) <= qgs]
    return kept or list(configs)


@smart_triton_autotune(
    configs=_quant_moe_autotune_configs(),
    selected_idx=0,
    key=["N", "K", "MAX_M", "QUANT_GROUP_SIZE"],
    prune_configs_by={"early_config_prune": _prune_block_k_gt_group},
)
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
    BLOCK_K: tl.constexpr,
):
    """Grouped int8 matmul with per-group dequant and optional epilogue.

    EPILOGUE_NONE:   output = dequant(A @ B.T)
    EPILOGUE_SWIGLU: B has N columns (gate + up), output HALF_N = N//2 columns
                     after silu(gate) * up.

    The int8 matmul uses tl.dot (int32 accumulator). The weight tile is loaded
    as [BLOCK_N, BLOCK_K] from row-major B[N, K] and transposed with tl.trans.
    Each quant group accumulates int32 over BLOCK_K tiles, then is dequantized
    by the per-group weight scale before being summed into the fp32 accumulator.
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
    offs_k = tl.arange(0, BLOCK_K)
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
    mask_n = offs_n < out_N

    # Number of BLOCK_K tiles needed to cover one quant group (constexpr).
    K_TILES_PER_GROUP: tl.constexpr = (QUANT_GROUP_SIZE + BLOCK_K - 1) // BLOCK_K

    for kg in range(NUM_GROUPS_K):
        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        if EPILOGUE == _SWIGLU:
            partial_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for kt in range(K_TILES_PER_GROUP):
            k_in_group = kt * BLOCK_K + offs_k
            k_off = kg * QUANT_GROUP_SIZE + k_in_group
            # Stay inside both the current quant group and the global K extent.
            k_mask = (k_in_group < QUANT_GROUP_SIZE) & (k_off < K)

            a = tl.load(
                A + offs_m[:, None] * K + k_off[None, :],
                mask=mask_m[:, None] & k_mask[None, :], other=0,
            )  # [BLOCK_M, BLOCK_K] int8
            a_f16 = a.to(tl.float16)
            b = tl.load(
                b_base + offs_n[:, None] * strideBN + k_off[None, :] * strideBK,
                mask=mask_n[:, None] & k_mask[None, :], other=0,
            )  # [BLOCK_N, BLOCK_K] int8
            tile = tl.dot(a_f16, tl.trans(b).to(tl.float16), out_dtype=tl.float32)
            # tile holds the exact integer A@B.T for this BLOCK_K tile (|sum| <=
            # BLOCK_K*127^2 << 2^24, fp32-exact). Round (not truncate) before the
            # int32 cast so any sub-0.5 fp drift cannot flip the integer.
            partial += (tile + tl.where(tile >= 0, 0.5, -0.5)).to(tl.int32)

            if EPILOGUE == _SWIGLU:
                bu = tl.load(
                    b_base + offs_n_up[:, None] * strideBN + k_off[None, :] * strideBK,
                    mask=(offs_n_up[:, None] < N) & k_mask[None, :], other=0,
                )  # [BLOCK_N, BLOCK_K] int8
                tile_up = tl.dot(a_f16, tl.trans(bu).to(tl.float16), out_dtype=tl.float32)
                partial_up += (tile_up + tl.where(tile_up >= 0, 0.5, -0.5)).to(tl.int32)

        ws = tl.load(
            ws_base + offs_n * stride_ws_n + kg * stride_ws_k,
            mask=mask_n, other=0.0,
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


def _make_group_offsets(
    size_per_group: torch.Tensor, num_groups: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Build [num_groups + 1] int32 prefix sums and the per-group max.

    The single ``.max().item()`` here is the only device->host sync in the
    experts pipeline; compute it once and reuse for every quant/GEMM launch.
    """
    cum = size_per_group.cumsum(0, dtype=torch.int32)
    group_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device=device)
    group_offsets[1:] = cum
    max_m = int(size_per_group.max().item())
    return group_offsets, max_m


def _prepare_quant_gemm_args(
    A, input_scale, weight_scale, size_per_group, num_groups, K, quant_group_size,
    group_offsets=None, max_m=None,
):
    if quant_group_size <= 0:
        quant_group_size = K
    num_groups_k = (K + quant_group_size - 1) // quant_group_size

    if weight_scale.ndim == 2:
        weight_scale = weight_scale.unsqueeze(-1)
    input_scale_flat = input_scale.reshape(-1).float()

    if group_offsets is None or max_m is None:
        group_offsets, max_m = _make_group_offsets(size_per_group, num_groups, A.device)

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
    group_offsets: torch.Tensor | None = None,
    max_m: int | None = None,
) -> torch.Tensor:
    quant_group_size, num_groups_k, weight_scale, input_scale_flat, group_offsets, max_m = \
        _prepare_quant_gemm_args(
            A, input_scale, weight_scale, size_per_group, num_groups, K, quant_group_size,
            group_offsets=group_offsets, max_m=max_m,
        )

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
    group_offsets: torch.Tensor | None = None,
    max_m: int | None = None,
) -> torch.Tensor:
    half_n = N // 2
    quant_group_size, num_groups_k, weight_scale, input_scale_flat, group_offsets, max_m = \
        _prepare_quant_gemm_args(
            A, input_scale, weight_scale, size_per_group, num_groups, K, quant_group_size,
            group_offsets=group_offsets, max_m=max_m,
        )

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
    group_offsets: torch.Tensor | None = None,
    max_tokens: int | None = None,
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

    if group_offsets is None or max_tokens is None:
        group_offsets, max_tokens = _make_group_offsets(tokens_per_expert, num_experts, device)
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

    # All four launches below share the same token-to-expert layout, so the
    # group offsets and per-expert max token count (the only device->host sync)
    # are computed once here and threaded through every step.
    group_offsets, max_m = _make_group_offsets(tokens_per_expert, num_experts, device)

    # --- Step 1: smooth + dynamic quant for up_proj input ---
    x_int8, x_scale = _moe_smooth_dynamic_quant(
        sorted_hidden_states,
        module.up_proj_quantize.inv_smooth_scale,
        tokens_per_expert,
        num_experts,
        group_offsets=group_offsets,
        max_tokens=max_m,
    )

    up_w = _cached_unpacked_proj_weight(module, "up_proj_weight")
    up_ws = module.up_proj_weight_scale

    _, n_up, k_in = up_w.shape
    inter = module.intermediate_size

    # --- Step 2: int8 grouped GEMM + SwiGLU epilogue ---
    # Use fp32 output to match the core precision chain (core keeps the activated
    # intermediate in fp32 before passing it to down_proj_quantize, whose per-token
    # dynamic scale is derived from this tensor's amax).
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
        group_offsets=group_offsets,
        max_m=max_m,
    )

    # --- Step 3: smooth + dynamic quant for down_proj input ---
    y_int8, y_scale = _moe_smooth_dynamic_quant(
        fc1_out,
        module.down_proj_quantize.inv_smooth_scale,
        tokens_per_expert,
        num_experts,
        group_offsets=group_offsets,
        max_tokens=max_m,
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
        group_offsets=group_offsets,
        max_m=max_m,
    )
    return out
