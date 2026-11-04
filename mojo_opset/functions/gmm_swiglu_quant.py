"""GmmSwigluQuant (grouped matmul + SwiGLU + MX quant, forward only) dispatched to one exact provider."""

import torch

from ._dispatch import load_impl


def _check_gmm_swiglu_quant(x, weight, weight_scale, x_scale, group_list, swiglu_limit):
    if x.dtype is not torch.float8_e4m3fn or weight.dtype is not torch.float8_e4m3fn:
        raise ValueError("gmm_swiglu_quant supports float8_e4m3fn x/weight only.")
    if x.dim() < 1:
        raise ValueError("gmm_swiglu_quant input must have at least one dimension.")
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size if hidden_size else 0
    e = weight.shape[0] if weight.dim() == 3 else 0
    if weight.dim() != 3 or weight.shape[1] != hidden_size or e < 1:
        raise ValueError(
            f"gmm_swiglu_quant weight must be [E, K, N]=[E>=1, {hidden_size}, N], "
            f"got {list(weight.shape)}."
        )
    n = weight.shape[2]
    if n % 2:
        raise ValueError("gmm_swiglu_quant weight N must be even (two SwiGLU halves).")
    k_blocks = (hidden_size + 31) // 32
    if weight_scale.dtype not in (torch.uint8, torch.int8) or weight_scale.dim() != 3 \
            or weight_scale.shape != (e, n, k_blocks):
        raise ValueError(
            f"gmm_swiglu_quant weight_scale must be uint8/int8 e8m0 of shape [{e}, {n}, {k_blocks}] (N-major), "
            f"got {tuple(weight_scale.shape)} {weight_scale.dtype}."
        )
    if x_scale.dtype not in (torch.uint8, torch.int8) or x_scale.dim() != 2 \
            or x_scale.shape != (rows, k_blocks):
        raise ValueError(
            f"gmm_swiglu_quant x_scale must be uint8/int8 e8m0 of shape [{rows}, {k_blocks}], "
            f"got {tuple(x_scale.shape)} {x_scale.dtype}."
        )
    if group_list.dtype != torch.int64 or group_list.numel() != e:
        raise ValueError(
            f"gmm_swiglu_quant group_list must be int64 [{e}] (per-group row cumsum, last == {rows})."
        )
    if not (x.device == weight.device == weight_scale.device == x_scale.device == group_list.device):
        raise ValueError("gmm_swiglu_quant tensors must share device.")
    if swiglu_limit < 0:
        raise ValueError("gmm_swiglu_quant swiglu_limit must be nonnegative.")


def gmm_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    implementation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 grouped matmul + SwiGLU + output quantization, forward only.

    Follows ops-transformer ``grouped_matmul_swiglu_quant_v2`` (950PR MX
    path) with the ``swiglu_limit`` clamp kept from the glm-5.3 lineage
    (``GlmMoeDsaMLP`` / ``Glm5NextTextMLP``; ``swiglu_limit=0`` disables it,
    glm-5.3-flash uses 10.0):

    ``C = (x . weight) with MXFP8 dequant fused into the dot;``
    ``S = silu(clamp(C[:, :N/2], max=L)) * clamp(C[:, N/2:], +-L);``
    ``(y, y_scale) = MX quant of S`` (32-element groups, e8m0 scales,
    OCP algorithm ``e = floor(log2 amax) - 8``, E4M3FN output).

    Inputs: ``x`` ``[M, K]`` float8_e4m3fn; ``weight`` ``[E, K, N]``
    float8_e4m3fn with the two SwiGLU halves along N (first half feeds the
    sigmoid branch); ``x_scale`` ``[M, ceil(K/32)]`` and ``weight_scale``
    ``[E, N, ceil(K/32)]`` (N-major) uint8/int8-carried e8m0 scales (one per
    32 K elements); ``group_list`` int64 ``[E]`` per-group row cumsum with
    ``last == M`` — rows ``[group_list[g-1], group_list[g])`` use
    ``weight[g]`` (E=1 is the dense case).

    Returns ``(y [M, N/2] float8_e4m3fn, y_scale [M, ceil((N/2)/32)]
    uint8 e8m0)`` ready for the next quantized GEMM.
    """
    _check_gmm_swiglu_quant(x, weight, weight_scale, x_scale, group_list, swiglu_limit)
    forward, _ = load_impl("gmm_swiglu_quant", implementation)
    return forward(x, weight, weight_scale, x_scale, group_list, float(swiglu_limit))
