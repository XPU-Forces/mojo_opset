"""Bounded-memory FP32 attention reference, with backward recomputed per query stripe."""

import torch

from mojo_opset.utils.flex_attention_mask import create_flex_block_mask


def _block_mask_to_dense(block_mask, query_length: int, kv_length: int, device) -> torch.Tensor:
    """Expand a ``BlockMask`` into a ``[query_length, kv_length]`` boolean mask."""
    query_block_size, kv_block_size = block_mask.BLOCK_SIZE
    kv_num_blocks = block_mask.kv_num_blocks
    kv_indices = block_mask.kv_indices
    if kv_num_blocks.shape[0] != 1 or kv_num_blocks.shape[1] != 1:
        raise NotImplementedError(
            "FlexAttention reference supports only batch/head-independent BlockMask tensors with leading shape [1, 1]"
        )
    num_query_blocks = kv_num_blocks.shape[2]
    full_kv_num_blocks = getattr(block_mask, "full_kv_num_blocks", None)
    full_kv_indices = getattr(block_mask, "full_kv_indices", None)
    dense_mask = getattr(block_mask, "dense_mask", None)
    packed_partial_mask = getattr(block_mask, "packed_partial_mask", None)
    partial_block_table = getattr(block_mask, "partial_block_table", None)
    if dense_mask is None and (packed_partial_mask is None or partial_block_table is None):
        raise NotImplementedError(
            "FlexAttention reference needs either block_mask.dense_mask or the packed partial-mask attributes "
            "to expand partial blocks"
        )
    mask = torch.zeros((num_query_blocks * query_block_size, kv_length), dtype=torch.bool, device=device)
    for query_block in range(num_query_blocks):
        query_slice = slice(query_block * query_block_size, (query_block + 1) * query_block_size)
        if full_kv_num_blocks is not None and full_kv_indices is not None:
            for entry in range(int(full_kv_num_blocks[0, 0, query_block])):
                kv_block = int(full_kv_indices[0, 0, query_block, entry])
                kv_slice = slice(kv_block * kv_block_size, min((kv_block + 1) * kv_block_size, kv_length))
                mask[query_slice, kv_slice] = True
        for entry in range(int(kv_num_blocks[0, 0, query_block])):
            kv_block = int(kv_indices[0, 0, query_block, entry])
            kv_stop = min((kv_block + 1) * kv_block_size, kv_length)
            kv_slice = slice(kv_block * kv_block_size, kv_stop)
            width = kv_stop - kv_block * kv_block_size
            packed_index = -1
            if packed_partial_mask is not None and partial_block_table is not None:
                packed_index = int(partial_block_table[query_block, kv_block])
            if packed_index >= 0:
                tile = packed_partial_mask[packed_index][:, :width]
            elif dense_mask is not None:
                tile = dense_mask[0, 0, query_slice, kv_slice]
            else:
                continue
            mask[query_slice, kv_slice] = tile.to(torch.bool)
    return mask[:query_length, :kv_length]


def _attention(q, k, v, mask, scale):
    group = q.shape[1] // k.shape[1]
    if group > 1:
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(mask.any(-1, keepdim=True), probabilities, torch.zeros_like(probabilities))
    return (probabilities @ v.float()).to(q.dtype), torch.logsumexp(scores, dim=-1)


def flex_attention_fwd(q, k, v, block_mask, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    mask = _block_mask_to_dense(block_mask, q.shape[2], k.shape[2], q.device)
    output = q.new_empty((*q.shape[:3], v.shape[-1]))
    lse = q.new_empty(q.shape[:3], dtype=torch.float32)
    for start in range(0, q.shape[2], 256):
        out, logsumexp = _attention(q[:, :, start : start + 256], k, v, mask[start : start + 256], scale)
        output[:, :, start : start + 256] = out
        lse[:, :, start : start + 256] = logsumexp
    return output, lse


def flex_attention_bwd(grad_output, q, k, v, output, lse, block_mask, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    mask = _block_mask_to_dense(block_mask, q.shape[2], k.shape[2], q.device)
    dq, dk, dv = (torch.zeros_like(x, dtype=torch.float32) for x in (q, k, v))
    for start in range(0, q.shape[2], 256):
        # Keep gradient accumulation in FP32 across stripes, including shared KV heads.
        with torch.enable_grad():
            qg = q[:, :, start : start + 256].detach().float().requires_grad_(True)
            kg, vg = (x.detach().float().requires_grad_(True) for x in (k, v))
            out, _ = _attention(qg, kg, vg, mask[start : start + 256], scale)
        grads = torch.autograd.grad(out, (qg, kg, vg), grad_output[:, :, start : start + 256].float())
        dq[:, :, start : start + 256] = grads[0]
        dk += grads[1]
        dv += grads[2]
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


flex_attention_fwd.create_block_mask = create_flex_block_mask
