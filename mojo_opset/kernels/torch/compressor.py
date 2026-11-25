"""Device-independent reference for token-aligned C2 pooling and ring upkeep."""

import torch

STATE_RING_ROWS = 32


def compressor_from_projected_fwd(
    kv: torch.Tensor,
    scores: torch.Tensor,
    state_cache: torch.Tensor,
    metadata: torch.Tensor,
    out: torch.Tensor,
    max_query_len: int,
    num_cores: int,
    prev_rows,
    prev_row_indices,
) -> torch.Tensor:
    out.zero_()
    tokens, width = kv.shape
    requests = metadata.shape[1]
    if tokens == 0 or requests == 0 or max_query_len == 0:
        return out
    starts, used, _, bases, blocks = metadata.detach().to("cpu").tolist()
    gathered = prev_rows is not None
    collected = prev_row_indices.detach().to("cpu").tolist() if gathered else None
    ring_blocks = state_cache.shape[0]
    for request in range(requests):
        block = int(blocks[request])
        length = int(used[request])
        if not 0 < block < ring_blocks or length <= 0:
            continue
        start = int(starts[request])
        base = int(bases[request])
        offsets = torch.arange(length, device=kv.device)
        rows = base + offsets
        positions = start + offsets
        completed = (positions % 2) == 1
        picked = offsets[completed]
        if picked.numel():
            # Residual reads must happen before the ring update overwrites the slots.
            target = rows[completed]
            lead = picked == 0
            previous_k = kv.new_empty((picked.numel(), width))
            previous_s = kv.new_empty((picked.numel(), width))
            inner = ~lead
            if inner.any():
                source = target[inner] - 1
                previous_k[inner] = kv[source]
                previous_s[inner] = scores[source]
            if lead.any():
                if gathered and int(collected[request]) >= 0:
                    row = prev_rows[int(collected[request])]
                    previous_k[lead] = row[:width]
                    previous_s[lead] = row[width:]
                else:
                    slot = (positions[completed][lead] - 1) % STATE_RING_ROWS
                    history = state_cache[block, slot]
                    previous_k[lead] = history[:, :width]
                    previous_s[lead] = history[:, width:]
            pair_k = torch.stack((previous_k, kv[target]), dim=1)
            pair_s = torch.stack((previous_s, scores[target]), dim=1)
            out[target] = (pair_s.softmax(dim=1) * pair_k).sum(dim=1).to(torch.bfloat16)
        if not gathered:
            tail = min(length, STATE_RING_ROWS)
            tail_offsets = offsets[length - tail :]
            slots = positions[length - tail :] % STATE_RING_ROWS
            state_cache[block, slots] = torch.cat((kv[base + tail_offsets], scores[base + tail_offsets]), dim=1)
    return out
