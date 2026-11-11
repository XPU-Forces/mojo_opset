import torch


def convert_req_index_to_global_index_fwd(
    req_id,
    block_table,
    token_indices,
    block_size,
    num_topk_tokens,
    has_prefill,
    prefill_workspace_request_ids,
    prefill_workspace_starts,
):
    block_id = torch.div(token_indices, block_size, rounding_mode="floor")
    offset = token_indices.remainder(block_size)
    safe_block = block_id.clamp(0, block_table.shape[1] - 1).long()
    base = block_table[req_id.long().unsqueeze(1), safe_block]
    result = base * block_size + offset
    result = result.masked_fill((block_id < 0) | (block_id >= block_table.shape[1]), -1)
    if has_prefill:
        active = prefill_workspace_request_ids >= 0
        starts = prefill_workspace_starts[prefill_workspace_request_ids.clamp(min=0).long()]
        result = torch.where(active.unsqueeze(1), starts.unsqueeze(1) + token_indices, result)
    return result.masked_fill(token_indices == -1, -1).to(torch.int32)
