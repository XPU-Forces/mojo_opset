"""Trainable grouped GEMM and forward-only dense/quantized GEMMs."""

import torch

from torch.autograd.function import once_differentiable

from ._checks import _require_inference
from ._dispatch import load_impl


def _call_dense(op, input, weight, input_scale, weight_scale, trans_input, trans_weight,
                output_dtype, c, alpha, recipe, compiled_dims, implementation):
    from mojo_opset.kernels._dense_gemm import check

    check(op, input, weight, input_scale, weight_scale, trans_input, trans_weight,
          output_dtype, c, alpha, recipe, compiled_dims)
    _require_inference(op, *(x for x in (input, weight, input_scale, weight_scale, c) if x is not None))
    forward, _ = load_impl(op, implementation)
    return forward(input, weight, input_scale, weight_scale, trans_input, trans_weight,
                   output_dtype, c, alpha, recipe[0], recipe[1], compiled_dims)


def gemm(
    input, weight, *, trans_input=False, trans_weight=False,
    output_dtype=torch.bfloat16, c=None, alpha=None, compiled_dims="", implementation=None,
):
    """Return alpha * op(input) @ op(weight) + C, without mutating inputs.

    Inputs are BF16; default operands are [M,K] and [K,N]. C is [M,N] or a contiguous [1,N]
    row bias; bias and alpha are mutually exclusive. Output is BF16 or FP32.
    The legacy compiled_dims argument is accepted for compatibility; the AOT
    backend uses runtime M/N/K regardless of its value.
    This migration API is inference-only and does not register backward.
    """
    return _call_dense(
        "gemm", input, weight, None, None, trans_input, trans_weight,
        output_dtype, c, alpha, (1, 1, 32), compiled_dims, implementation,
    )


def mxfp8_dequant_gemm(
    input, weight, input_scale, weight_scale, *, trans_input=False, trans_weight=False,
    output_dtype=torch.bfloat16, c=None, alpha=None, recipe=(1, 1, 32), compiled_dims="", implementation=None,
):
    """MXFP8 E4M3 x packed MXFP4 E2M1 GEMM; AIV decodes FP4 weights to FP8 for the Cube.

    FP4 storage is int8 with two values per byte along the contiguous axis,
    low nibble first. Logical shape doubles that axis. Scales describe logical
    [M,K/32] and [N,K/32] (FP32 powers of two or packed UE8M0 int16 pairs).
    recipe=(gm,gn,32) allows repeating a scale across gm/gn rows in [1,128].
    """
    return _call_dense("mxfp8_dequant_gemm", input, weight, input_scale, weight_scale, trans_input, trans_weight,
                       output_dtype, c, alpha, recipe, compiled_dims, implementation)


def mxfp4_gemm(
    input, weight, input_scale, weight_scale, *, trans_input=False, trans_weight=False,
    output_dtype=torch.bfloat16, c=None, alpha=None, recipe=(1, 1, 32), compiled_dims="", implementation=None,
):
    """Packed MXFP4 E2M1 x E2M1 GEMM; storage and scales follow mxfp8_dequant_gemm."""
    return _call_dense("mxfp4_gemm", input, weight, input_scale, weight_scale, trans_input, trans_weight,
                       output_dtype, c, alpha, recipe, compiled_dims, implementation)


def _check_group_gemm(input, weight, group_list, trans_weight):
    if input.ndim != 2 or weight.ndim != 3 or group_list.ndim != 1:
        raise ValueError("expected input [M,K], weight [G,K,N], and group_list [G]")
    if weight.shape[0] != group_list.numel() or input.shape[1] != weight.shape[2 if trans_weight else 1]:
        raise ValueError("weight dimensions must match input and group_list")
    if input.device != weight.device or input.dtype != weight.dtype:
        raise ValueError("input and weight must share device and dtype")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise ValueError("group_list must contain integer row counts")


class GroupGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, group_list, trans_weight, implementation, require_backward=True):
        _check_group_gemm(input, weight, group_list, trans_weight)
        forward, ctx.backward_kernel = load_impl("group_gemm", implementation, require_backward=require_backward)
        ctx.save_for_backward(input, weight, group_list)
        ctx.trans_weight = trans_weight
        return forward(input, weight, group_list, trans_weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        input, weight, group_list = ctx.saved_tensors
        dx, dw = ctx.backward_kernel(grad, input, weight, group_list, ctx.trans_weight, *ctx.needs_input_grad[:2])
        return dx, dw, None, None, None, None


def group_gemm(input, weight, group_list, *, trans_weight=False, implementation=None):
    """Multiply row groups by [G,K,N] weights, or [G,N,K] when transposed.

    group_list contains nonnegative row counts summing to M. Training uses
    the selected backend's backward; inference requires only its forward.
    Input and weight gradients are supported when the backend declares them.
    """
    return GroupGemmFunction.apply(
        input,
        weight,
        group_list,
        trans_weight,
        implementation,
        torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad),
    )


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
