"""Eager reference for the compressed-KV slot mapping (vllm compressor_utils semantics)."""

import torch


def get_compressed_slot_mapping_fwd(
    num_tokens,
    slot_mapping,
    query_start_loc,
    seq_lens,
    block_table,
    block_size,
    compress_ratio,
    out,
    dcp_world_size,
    cp_interleave,
    max_query_len,
    slot_contig=None,
    positions=None,
):
    """Vectorised token-space replica of the Triton kernel, written into ``out``."""
    if out.numel() > num_tokens:
        out[num_tokens:] = -1
    if num_tokens:
        # qsl[-1] == num_tokens under cu_seqlens semantics; a zero-length request
        # contributes no rows here.
        qsl = query_start_loc.to(torch.int64)
        q_lens = qsl[1:] - qsl[:-1]
        request = torch.repeat_interleave(torch.arange(q_lens.numel(), device=qsl.device), q_lens)
        token = torch.arange(num_tokens, device=qsl.device)
        positions = seq_lens.to(torch.int64)[request] - q_lens[request] + (token - qsl[request])
        slots = slot_mapping.to(torch.int64)
        valid = ((positions + 1) % compress_ratio == 0) & (slots != -1)
        # DCP rank-local position: the state lives on the rank owning its last token.
        local_pos = (
            positions // (cp_interleave * dcp_world_size) * cp_interleave
            + positions % cp_interleave
        )
        pos_after_compress = local_pos // compress_ratio
        block_numbers = block_table.to(torch.int64)[request, pos_after_compress // block_size]
        values = block_numbers * block_size + pos_after_compress % block_size
        out[:num_tokens] = torch.where(valid, values, torch.full_like(values, -1))
    if slot_contig is not None and positions is not None:
        slots = out[:num_tokens]
        prev = torch.cat((torch.tensor([-2], device=slots.device), slots[:-1]))
        pos = positions[:num_tokens].to(torch.int64)
        prev_pos = torch.cat((torch.tensor([-2], device=pos.device), pos[:-1]))
        contig = (
            (slots != -1)
            & (slots == prev + 1)
            & (pos == prev_pos + 1)
            & ((pos + 1) % compress_ratio == 0)
        )
        slot_contig[:num_tokens] = contig.to(torch.uint8)
    return None
