"""Forward-only grouped and quantized GEMMs from the original Mojo operators."""

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def group_gemm(input, weight, group_list, *, trans_weight=False, implementation=None):
    """Multiply contiguous row groups by their respective [G, K, N] weights."""
    if input.ndim != 2 or weight.ndim != 3 or group_list.ndim != 1:
        raise ValueError("expected input [M,K], weight [G,K,N], and group_list [G]")
    if weight.shape[0] != group_list.numel() or input.shape[1] != weight.shape[2 if trans_weight else 1]:
        raise ValueError("weight dimensions must match input and group_list")
    _require_inference("group_gemm", input, weight)
    forward, _ = load_impl("group_gemm", implementation)
    return forward(input, weight, group_list, trans_weight)


def quant_gemm(
    input, weight, input_scale, weight_scale, *, output_dtype=torch.bfloat16, trans_weight=False, implementation=None
):
    """INT8 GEMM with per-token and per-output scales; return [M, N]."""
    if input.ndim != 2 or weight.ndim != 2 or input.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("input and weight must be 2D int8 tensors")
    k, n = weight.shape[::-1] if trans_weight else weight.shape
    if input.shape[1] != k or weight_scale.shape != (n,):
        raise ValueError("weight K and scale N must match the input and output dimensions")
    if input_scale.shape not in ((input.shape[0],), (input.shape[0], 1)):
        raise ValueError("input_scale must have shape [M] or [M, 1]")
    _require_inference("quant_gemm", input_scale, weight_scale)
    forward, _ = load_impl("quant_gemm", implementation)
    return forward(input, weight, input_scale, weight_scale, output_dtype, trans_weight)


def quant_batch_gemm_reduce_sum(input, weight, x1_scale, x2_scale, *, trans_weight=False, implementation=None):
    """Batch INT8 GEMM with scaled BF16 accumulation over the batch dimension."""
    if input.ndim != 3 or weight.ndim != 3 or input.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("input and weight must be 3D int8 tensors")
    if input.shape[0] != weight.shape[0] or input.shape[2] != weight.shape[2 if trans_weight else 1]:
        raise ValueError("input and weight batch and K dimensions must match")
    if x1_scale.shape != input.shape[:2] or x2_scale.shape != (weight.shape[1 if trans_weight else 2],):
        raise ValueError("x1_scale must be [B,M] and x2_scale must be [N]")
    _require_inference("quant_batch_gemm_reduce_sum", x1_scale, x2_scale)
    forward, _ = load_impl("quant_batch_gemm_reduce_sum", implementation)
    return forward(input, weight, x1_scale, x2_scale, trans_weight)
