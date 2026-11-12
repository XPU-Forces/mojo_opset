import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def topk_infer(
    input,
    topk,
    sorted=False,
    begin=None,
    end=None,
    indices_type=torch.int32,
    sorted_index=False,
    hint=None,
    output_idx=None,
    output_idx_offset=None,
    idx_oob_fill_value=2147483647,
    value_oob_fill_value=float("-inf"),
    return_value=True,
    abort_when_nan_found=True,
    *,
    implementation=None,
):
    """Forward-only TopK for BF16 rows and small-axis FP32 selection.

    BF16 supports K=1..4096, N<2**21, stride(1)=1 and 32-byte-aligned row strides.
    Outputs also have 32-byte-aligned row strides and may be noncontiguous.
    With end[B], callers must ensure 0 <= end <= N. Short rows retain input
    order and are padded; offsets apply only to valid indices. sorted_index
    orders selected entries by index. FP32 supports [B,N], N=1..1024 and
    K=1..min(N,32), with sorted=True, int64 indices and return_value=True.
    Its last dimension must be contiguous; input row strides need not be aligned.
    FP32 does not support end, offsets or index sorting. Autograd is unsupported.
    NaNs in the valid range abort the native kernel by default; False writes
    index sentinel 0x3f3f3f3f for the entire affected row; its values are unspecified.
    """
    _require_inference("topk_infer", input)
    if begin is not None or hint is not None:
        raise NotImplementedError("topk_infer does not support begin or hint")
    if input.dtype == torch.float32:
        if input.ndim != 2 or not 1 <= input.shape[1] <= 1024 or not 1 <= topk <= min(input.shape[1], 32):
            raise ValueError("FP32 topk_infer requires [B,N], N=1..1024 and K=1..min(N,32)")
        if not sorted or indices_type != torch.int64 or sorted_index or not return_value:
            raise NotImplementedError("FP32 topk_infer requires sorted=True, int64 indices and return_value=True")
        if end is not None or output_idx_offset is not None:
            raise NotImplementedError("FP32 topk_infer does not support end or output_idx_offset")
    elif sorted or indices_type != torch.int32:
        raise NotImplementedError("topk_infer currently requires sorted=False and int32 indices")
    forward, _ = load_impl("topk_infer", implementation)
    return forward(input, topk, end, sorted_index, output_idx, output_idx_offset,
                   idx_oob_fill_value, value_oob_fill_value, return_value, abort_when_nan_found)
