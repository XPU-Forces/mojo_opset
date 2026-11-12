from functools import lru_cache
from typing import Optional
from typing import Tuple

import torch


@lru_cache(maxsize=1)
def _launcher():
    import torch_npu  # noqa: F401 -- load the dependent runtime first

    try:
        from mojo_opset_lib.npu_a5 import _native

        return _native.topk_forward
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Native extension mojo_opset_lib.npu_a5._native could not be loaded "
            "or does not export topk_forward. Build and install it: bash build.sh npu_a5"
        ) from error


@torch.library.custom_op(
    "mojo_npu_native_a5::topk_infer", mutates_args=(),
    schema="(Tensor input, int topk, Tensor? end, bool sorted_index, Tensor? output_idx_offset, "
           "int idx_oob_fill_value, float value_oob_fill_value, bool return_value, bool abort_when_nan_found) "
           "-> (Tensor?, Tensor)",
)
def _topk(
    input: torch.Tensor,
    topk: int,
    end: Optional[torch.Tensor],
    sorted_index: bool,
    output_idx_offset: Optional[torch.Tensor],
    idx_oob_fill_value: int,
    value_oob_fill_value: float,
    return_value: bool,
    abort_when_nan_found: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    return _launcher()(input, topk, end, sorted_index, None, output_idx_offset,
                       idx_oob_fill_value, value_oob_fill_value, return_value, abort_when_nan_found)


@_topk.register_fake
def _topk_fake(input, topk, end, sorted_index, output_idx_offset,
               idx_oob_fill_value, value_oob_fill_value, return_value, abort_when_nan_found):
    def output(dtype, elements_per_block):
        stride = (topk + elements_per_block - 1) // elements_per_block * elements_per_block
        return input.new_empty_strided((input.shape[0], topk), (stride, 1), dtype=dtype)

    fp32 = input.dtype == torch.float32
    indices = output(torch.int64 if fp32 else torch.int32, 4 if fp32 else 8)
    values = output(input.dtype, 32 // input.element_size()) if return_value else None
    return values, indices


def topk_infer_fwd(input, topk, end, sorted_index, output_idx, output_idx_offset,
                   idx_oob_fill_value, value_oob_fill_value, return_value, abort_when_nan_found):
    if output_idx is None:
        return _topk(input, topk, end, sorted_index, output_idx_offset,
                     idx_oob_fill_value, value_oob_fill_value, return_value, abort_when_nan_found)
    return _launcher()(input, topk, end, sorted_index, output_idx, output_idx_offset,
                       idx_oob_fill_value, value_oob_fill_value, return_value, abort_when_nan_found)
