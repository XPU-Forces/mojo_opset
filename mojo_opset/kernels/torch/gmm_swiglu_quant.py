"""Reference GmmSwigluQuant forward (MXFP8 in/out) used by accuracy checks and dispatch fallbacks."""

import torch
import torch.nn.functional as F

# E4M3FN: 3 mantissa bits, maximum finite value 448, emax = 8.
_E4M3_EMAX = 8
_E4M3_MAX = 448.0
_MX_GROUP = 32


def e8m0_to_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Decode e8m0 scale bytes (uint8/int8 carrier) to 2^(byte - 127)."""
    return torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0)


def mxfp8_quantize(values: torch.Tensor):
    """Quantize ``[..., K]`` to MXFP8 (OCP scales): e4m3 data + e8m0 scales.

    V3 numerics: values are rint-rounded (RNE) to BF16 first and the group
    amax / element quantization run on that BF16 view; elements are then
    quantized by one RTNE cast straight onto the fp8 grid (the reference
    ``cast_E4M3FN,rint``), with an explicit +-448 clamp providing the
    satfinite half of the OCP no-carry contract.

    Returns ``(data, scale_bytes)`` where data keeps the input shape and
    scale_bytes is ``[..., ceil(K / 32)]``: one e8m0 byte per 32 trailing
    elements, K-groups innermost (axis order mirrors the data tensor).
    """
    shape = values.shape
    k = shape[-1]
    v2 = values.to(torch.bfloat16).to(torch.float32).reshape(-1, k)
    blocks = -(-k // 32)
    pad = blocks * _MX_GROUP - v2.shape[1]
    if pad:
        v2 = F.pad(v2, (0, pad))
    blocks_t = v2.reshape(-1, blocks, _MX_GROUP)
    amax = blocks_t.abs().amax(dim=-1)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax)) - _E4M3_EMAX, torch.full_like(amax, -127.0))
    e = e.clamp_(min=-127.0)  # e8m0 encoding floor; keeps the byte cast in range
    q = (blocks_t * torch.exp2(-e)[..., None]).clamp_(-_E4M3_MAX, _E4M3_MAX)
    data = q.reshape(q.shape[0], -1)[:, :k].to(torch.float8_e4m3fn).reshape(shape)
    scales = (e + 127.0).to(torch.uint8).reshape(*shape[:-1], blocks)
    return data, scales


def mxfp8_quantize_weight(values: torch.Tensor):
    """MXFP8-quantize a ``[K, N]`` weight with 32-element groups along K.

    Returns ``(data [K, N], scale_bytes [N, ceil(K/32)])``: N-major — each
    column's K-group scales are contiguous, the orientation tl.dot_scaled's
    ``[N, K // 32]`` rhs_scale tiles consume directly (zero transposes).
    """
    data, scales = mxfp8_quantize(values.t())
    return data.t().contiguous(), scales.contiguous()


def gmm_swiglu_quant_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 grouped matmul + SwiGLU + MX quant with FP32 math.

    C = (x * x_scale) @ (w_g * w_scale_g) with the per-32-K scales expanded
    before the reduction; S = silu(clamp(C[:, :I], max=L)) *
    clamp(C[:, I:], +-L); (y, y_scale) = MXFP8 quant of S. Rows
    [group_list[g-1], group_list[g]) of x use weight[g].
    """
    k = x.shape[-1]
    rows = x.numel() // k
    e = weight.shape[0]
    n = weight.shape[-1]
    inter = n // 2
    k_blocks = -(-k // 32)
    ends = group_list.cpu().tolist()  # [E] per-group row cumsum

    x2 = x.reshape(rows, k).to(torch.float32)
    # x_scale [rows, kb] (row-major); weight_scale [E, N, kb] (N-major).
    xs = e8m0_to_scale(x_scale.reshape(rows, k_blocks))
    ws_all = e8m0_to_scale(weight_scale.reshape(e, n, k_blocks))
    ys, y_scales = [], []
    start = 0
    for g in range(e):
        end = int(ends[g])
        if end > start:  # empty experts contribute no rows
            w2 = weight[g].reshape(k, n).to(torch.float32)
            ws = ws_all[g].t()
            c = (x2[start:end] * xs[start:end].repeat_interleave(_MX_GROUP, dim=1)[:, :k]) @ (
                w2 * ws.repeat_interleave(_MX_GROUP, dim=0)[:k, :]
            )
            act, lin = c[:, :inter], c[:, inter:]
            if swiglu_limit > 0:
                act = act.clamp(max=swiglu_limit)
                lin = lin.clamp(min=-swiglu_limit, max=swiglu_limit)
            y_g, ys_g = mxfp8_quantize(F.silu(act) * lin)
            ys.append(y_g)
            y_scales.append(ys_g)
        start = end
    if not ys:  # all groups empty
        return (x.new_empty((0, inter), dtype=torch.float8_e4m3fn),
                x.new_zeros((0, -(-inter // 32)), dtype=torch.uint8))
    return torch.cat(ys, dim=0), torch.cat(y_scales, dim=0)
