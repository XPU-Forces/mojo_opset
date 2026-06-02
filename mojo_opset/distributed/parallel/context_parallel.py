import logging

import torch


logger = logging.getLogger(__name__)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be > 0, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def pad_deepseek_v4_cp_prefill(input_ids, attention_mask, pad_token_id, cp_size):
    """Pad DeepSeekV4 CP prefill inputs to the CP window alignment."""

    if cp_size <= 1:
        return input_ids, attention_mask

    cp_segment_num = cp_size * 2
    cp_window_alignment = cp_segment_num * 128
    target_len = align_up(input_ids.shape[1], cp_window_alignment)

    if target_len == input_ids.shape[1]:
        return input_ids, attention_mask

    pad_width = target_len - input_ids.shape[1]
    logger.warning(
        "[INPUT_PAD] DeepseekV4 CP prefill pad: orig_len=%s target_len=%s cp_size=%s cp_segment_num=%s",
        input_ids.shape[1],
        target_len,
        cp_size,
        cp_segment_num,
    )
    input_ids = torch.nn.functional.pad(input_ids, (0, pad_width), value=pad_token_id)
    attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_width), value=False)
    return input_ids, attention_mask


def copy_paged_dummy_cache_batch(dst_cache, src_cache, dst_batch_idx: int) -> None:
    dst_batch_size = int(dst_cache.batch_size)
    src_batch_size = int(src_cache.batch_size)
    if src_batch_size != 1:
        raise ValueError(f"CP prefill mini-batch copy expects src batch size 1, got {src_batch_size}")
    if not (0 <= dst_batch_idx < dst_batch_size):
        raise ValueError(f"dst_batch_idx={dst_batch_idx} out of range for batch_size={dst_batch_size}")

    dst_cache.seq_lens[:, dst_batch_idx].copy_(src_cache.seq_lens[:, 0])

    for layer_idx in range(src_cache.num_layers):
        src_layer = src_cache.cache_data[layer_idx]
        dst_layer = dst_cache.cache_data[layer_idx]
        for cache_name, src_tensor in src_layer.items():
            if cache_name.endswith("_block_table"):
                continue
            if src_tensor is None or not torch.is_tensor(src_tensor):
                continue
            dst_tensor = dst_layer.get(cache_name)
            if dst_tensor is None or not torch.is_tensor(dst_tensor):
                continue
            src_blocks_per_batch = (src_tensor.shape[0] - 1) // src_batch_size
            dst_blocks_per_batch = (dst_tensor.shape[0] - 1) // dst_batch_size
            blocks_to_copy = min(src_blocks_per_batch, dst_blocks_per_batch)
            if blocks_to_copy <= 0:
                continue
            src_start = 1
            dst_start = 1 + dst_batch_idx * dst_blocks_per_batch
            dst_tensor[dst_start: dst_start + blocks_to_copy].copy_(
                src_tensor[src_start: src_start + blocks_to_copy]
            )


def forward_cp_prefill_minibatch_mojo(
    model,
    input_ids,
    attention_mask,
    lengths,
    runtime_state,
    *,
    local_shard_start,
    local_shard_end,
    max_seq_len,
    runtime_state_factory,
    forward_prefill_fn,
    copy_cache_fn=copy_paged_dummy_cache_batch,
    use_attn_metadata=True,
    return_all_hidden=False,
):
    """Run DeepSeekV4 Golden-style CP prefill one global sample at a time.

    The orchestration lives in distributed.parallel, while model-specific
    runtime construction and forward execution are provided as callbacks.
    """

    total_batch = input_ids.shape[0]
    local_batch = local_shard_end - local_shard_start
    runtime_batch = int(runtime_state.paged_cache.batch_size)
    if runtime_batch != local_batch:
        raise ValueError(
            "CP prefill runtime batch size mismatch: "
            f"runtime_batch={runtime_batch}, local_batch={local_batch}, "
            f"owner_range=[{local_shard_start}, {local_shard_end})"
        )

    logits_chunks = []
    hidden_chunks = []
    all_hidden_chunks = []
    for global_batch_idx in range(total_batch):
        owns_sample = local_shard_start <= global_batch_idx < local_shard_end
        minibatch_runtime = runtime_state_factory(
            model,
            batch_size=1,
            max_seq_len=max_seq_len,
        )
        next_logits, _, hidden_states = forward_prefill_fn(
            model,
            input_ids[global_batch_idx: global_batch_idx + 1],
            attention_mask[global_batch_idx: global_batch_idx + 1],
            lengths[global_batch_idx: global_batch_idx + 1],
            runtime_state=minibatch_runtime,
            use_attn_metadata=use_attn_metadata,
        )
        if return_all_hidden:
            if hidden_states is None:
                raise RuntimeError("CP prefill did not return hidden_states required by MTP.")
            all_hidden_chunks.append(hidden_states)
        if owns_sample:
            local_slot = global_batch_idx - local_shard_start
            copy_cache_fn(
                runtime_state.paged_cache,
                minibatch_runtime.paged_cache,
                local_slot,
            )
            logits_chunks.append(next_logits)
            if hidden_states is not None:
                hidden_chunks.append(hidden_states)
        del minibatch_runtime

    if not logits_chunks:
        raise RuntimeError(
            "CP-aware prefill did not collect any owner logits for this rank. "
            f"owner_range=[{local_shard_start}, {local_shard_end}), total_batch={total_batch}"
        )

    hidden_out = torch.cat(hidden_chunks, dim=0) if hidden_chunks else None
    if return_all_hidden:
        all_hidden_out = torch.cat(all_hidden_chunks, dim=0) if all_hidden_chunks else None
        return torch.cat(logits_chunks, dim=0), runtime_state.paged_cache, hidden_out, all_hidden_out
    return torch.cat(logits_chunks, dim=0), runtime_state.paged_cache, hidden_out
