"""Executable references for the original Mojo GEMM semantics."""

import torch


def group_gemm_fwd(input, weight, group_list, trans_weight):
    if trans_weight:
        weight = weight.transpose(1, 2)
    groups = group_list.cpu().tolist()
    if any(size < 0 for size in groups) or sum(groups) != input.shape[0]:
        raise ValueError("group_list must contain nonnegative row counts summing to M")
    if not groups:
        return input.new_empty((0, weight.shape[-1]))
    return torch.cat([x @ w for x, w in zip(input.split(groups), weight)], dim=0)


def group_gemm_bwd(grad, input, weight, group_list, trans_weight, need_input, need_weight):
    dx = group_gemm_fwd(grad, weight, group_list, not trans_weight) if need_input else None
    dw = None
    if need_weight:
        groups = group_list.cpu().tolist()
        compute_dtype = torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
        products = [
            x.to(compute_dtype).T @ dy.to(compute_dtype)
            for x, dy in zip(input.split(groups), grad.split(groups))
        ]
        dw = (
            torch.stack(products).to(weight.dtype)
            if products else weight.new_empty((0, input.shape[1], grad.shape[1]))
        )
        if trans_weight:
            dw = dw.transpose(1, 2).contiguous()
    return dx, dw


def quant_gemm_fwd(input, weight, input_scale, weight_scale, output_dtype, trans_weight):
    weight = weight if trans_weight else weight.mT
    # Match the original INT32-products / FP32-sum reference without allocating
    # its full [M,N,K] intermediate for the inherited large GEMM cases.
    rows = []
    for start in range(0, input.shape[0], 4):
        products = input[start : start + 4].int().unsqueeze(-2) * weight.int()
        rows.append(products.float().sum(dim=-1))
    out = torch.cat(rows) if rows else input.new_empty((0, weight.shape[0]), dtype=torch.float32)
    return (out * input_scale.reshape(-1, 1).float() * weight_scale.float()).to(output_dtype)


def quant_batch_gemm_reduce_sum_fwd(input, weight, x1_scale, x2_scale, trans_weight):
    if trans_weight:
        weight = weight.transpose(1, 2).contiguous()
    out = torch.bmm(input.float(), weight.float())
    out = x2_scale.to(torch.bfloat16)[None, None, :] * out
    out = x1_scale[:, :, None] * out
    reduced = out.new_zeros(out.shape[1:], dtype=torch.bfloat16)
    for batch in out:
        reduced += batch.to(torch.bfloat16)
    return reduced
