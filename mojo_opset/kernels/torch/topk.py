import torch


def topk_infer_fwd(input, topk, end, sorted_index, output_idx, output_idx_offset,
                   idx_oob_fill_value, value_oob_fill_value, return_value, abort_when_nan_found):
    def aligned_output(dtype, alignment):
        return torch.empty_strided((input.shape[0], topk), ((topk + alignment - 1) // alignment * alignment, 1),
                                   device=input.device, dtype=dtype)

    if input.dtype == torch.float32:
        nan_rows = input.isnan().any(dim=-1)
        if abort_when_nan_found and bool(nan_rows.any()):
            raise RuntimeError("topk_infer: NaN in a valid input row")
        indices = input.argsort(dim=-1, descending=True, stable=True)[:, :topk]
        values = input.gather(1, indices)
        indices = indices.masked_fill(nan_rows[:, None], 0x3f3f3f3f)
        values = values.masked_fill(nan_rows[:, None], float("nan"))
        result_index = output_idx if output_idx is not None else aligned_output(torch.int64, 4)
        result_value = aligned_output(torch.float32, 8)
        result_index.copy_(indices)
        result_value.copy_(values)
        return result_value, result_index
    if input.ndim != 2 or input.dtype != torch.bfloat16:
        raise ValueError("topk_infer requires BF16 or FP32 rows")
    if not 1 <= topk <= 4096:
        raise ValueError("topk must be in [1,4096]")
    batch, width = input.shape
    lengths = end if end is not None else torch.full((batch,), width, device=input.device, dtype=torch.int32)
    valid = torch.arange(width, device=input.device)[None, :] < lengths[:, None]
    nan_rows = (input.isnan() & valid).any(dim=-1)
    if abort_when_nan_found and bool(nan_rows.any()):
        raise RuntimeError("topk_infer: NaN in a valid input row")
    # Use FP32 argsort to avoid CANN 9.1.T103 TopK overruns on tied BF16 scores.
    scores = input.float().masked_fill(~valid, float("-inf"))
    # Keep valid -inf entries ahead of the masked suffix when their values tie.
    selected = scores.argsort(dim=-1, descending=True, stable=True)[:, :min(topk, width)]
    index = torch.full((batch, topk), idx_oob_fill_value, device=input.device, dtype=torch.int32)
    index[:, :selected.shape[1]] = selected.to(torch.int32)
    columns = torch.arange(topk, device=input.device, dtype=torch.int32)[None, :]
    index = torch.where((lengths <= topk)[:, None], columns, index)
    selected_valid = columns < lengths[:, None]
    index = torch.where(selected_valid, index, idx_oob_fill_value)
    values = None
    if return_value:
        values = torch.full((batch, topk), value_oob_fill_value, device=input.device, dtype=input.dtype)
        if width:
            gathered = input.gather(1, index.clamp(0, width - 1).long())
            values = torch.where(selected_valid, gathered, values)
    if output_idx_offset is not None:
        index = torch.where(selected_valid, index + output_idx_offset[:, None], index)
    if sorted_index:
        ordered, permutation = index.sort(dim=-1)
        long_rows = (lengths > topk)[:, None]
        index = torch.where(long_rows, ordered, index)
        if return_value:
            values = torch.where(long_rows, values.gather(1, permutation), values)
    index = torch.where(nan_rows[:, None], 0x3f3f3f3f, index)

    indices = output_idx if output_idx is not None else aligned_output(torch.int32, 8)
    indices.copy_(index)
    if return_value:
        result = aligned_output(input.dtype, 16)
        result.copy_(values)
        values = result
    return values, indices
