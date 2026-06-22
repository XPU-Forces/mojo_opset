from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.distributed as dist


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


def get_zigzag_idx(origin_idx: int, cp_segment_num: int) -> tuple[int, str]:
    midpoint = cp_segment_num // 2 - 1
    if origin_idx <= midpoint:
        return origin_idx, "prev"
    return midpoint + 1 - (origin_idx - midpoint), "next"


def apply_cp_zigzag_to_prefill_inputs(hidden_states, input_ids, position_ids, cp_metadata):
    """Select the local prev/next CP segments from full prefill tensors."""

    if cp_metadata is None:
        return hidden_states, input_ids, position_ids
    split_list = cp_metadata["split_list"]
    zigzag_idx = cp_metadata["zigzag_idx"]
    hidden_segments = hidden_states.split(split_list, dim=1)
    input_segments = input_ids.split(split_list, dim=1)
    pos_segments = position_ids.split(split_list, dim=1)
    return (
        torch.cat([hidden_segments[idx] for idx in zigzag_idx], dim=1),
        torch.cat([input_segments[idx] for idx in zigzag_idx], dim=1),
        torch.cat([pos_segments[idx] for idx in zigzag_idx], dim=1),
    )


def gather_cp_prefill_hidden_states(hidden_states: torch.Tensor, attn_metadata: dict, *, cp_size: int, cp_group):
    """Gather local prev/next hidden states from CP ranks and restore global order."""

    cp_meta = attn_metadata.get("cp_metadata") if attn_metadata is not None else None
    if cp_meta is None or cp_group is None or cp_size <= 1:
        return hidden_states
    if hidden_states.shape[0] != 1:
        raise NotImplementedError("Mojo CP prefill currently only supports batch_size=1.")
    local_seq = hidden_states.shape[1]
    if local_seq % 2 != 0:
        raise ValueError("CP prefill expects local hidden states to split into prev/next evenly.")
    segment_len = local_seq // 2
    gathered = hidden_states.new_empty((cp_size, *hidden_states.shape))
    dist.all_gather_into_tensor(gathered, hidden_states.contiguous(), group=cp_group)
    gathered = gathered.view(cp_size * 2, hidden_states.shape[0], segment_len, hidden_states.shape[-1])
    gathered = gathered[cp_meta["reverse_index"]]
    return gathered.permute(1, 0, 2, 3).reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])


def build_deepseek_v4_cp_metadata(
    *,
    input_ids,
    attn_metadata,
    config,
    cp_size: int,
    cp_group,
    cp_rank: int,
    paged_cache,
    metadata_compress_ratios: Callable[[], list[int]],
    compute_sas_metadata: Callable,
    compute_li_metadata: Callable,
    get_slot_mapping_from_block_table: Callable,
    get_padded_slot_mapping_from_block_table: Callable,
):
    """Build DeepSeek-V4 Golden-style CP metadata in the parallel layer.

    The strategy is generic at the orchestration level and receives model/runtime
    specific cache helpers as callbacks to avoid importing DeepSeek modules here.
    """

    if cp_size <= 1 or not dist.is_initialized():
        attn_metadata["cp_metadata"] = None
        return attn_metadata
    batch_size, seq_len = input_ids.shape
    if batch_size != 1:
        raise NotImplementedError("Mojo CP prefill currently only supports batch_size=1.")
    if cp_group is None:
        raise ValueError("cp_size > 1 requires cp_group to be initialized.")
    cp_segment_num = cp_size * 2
    if seq_len % cp_segment_num != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by cp_segment_num={cp_segment_num}.")

    kv_len = attn_metadata["kv_len"].to(dtype=torch.int32)
    position_ids = attn_metadata["position_ids"]
    segment_len = seq_len // cp_segment_num
    split_list_hidden = [segment_len] * cp_segment_num
    split_position_ids = list(position_ids.split(split_list_hidden, dim=-1))
    zigzag_idx = [cp_rank, cp_segment_num - cp_rank - 1]
    reverse_index = torch.tensor(
        list(range(0, cp_segment_num, 2)) + list(range(cp_segment_num - 1, 0, -2)),
        device=position_ids.device,
        dtype=torch.long,
    )
    split_kv_len = (
        torch.min(
            (torch.arange(cp_segment_num, device=position_ids.device).unsqueeze(0) + 1) * segment_len,
            kv_len.unsqueeze(1),
        )
        - torch.min(
            torch.arange(cp_segment_num, device=position_ids.device).unsqueeze(0) * segment_len,
            kv_len.unsqueeze(1),
        )
    ).to(dtype=torch.int32)
    last_rank_before_zz = int(((split_kv_len > 0).sum(dim=1) - 1).item())
    last_rank_zz, last_rank_flag = get_zigzag_idx(last_rank_before_zz, cp_segment_num)
    attn_metadata["cp_metadata"] = {
        "split_list": split_list_hidden,
        "zigzag_idx": zigzag_idx,
        "reverse_index": reverse_index,
        "split_kv_len": split_kv_len,
        "last_rank": last_rank_before_zz,
        "last_rank_flag": last_rank_flag,
        "last_rank_zz": last_rank_zz,
    }
    attn_metadata["prev"] = {}
    attn_metadata["next"] = {}

    ratios = sorted({r for r in metadata_compress_ratios() if r > 1})
    for zigzag_flag in ["prev", "next"]:
        segment_idx = cp_rank if zigzag_flag == "prev" else 2 * cp_size - 1 - cp_rank
        cur_position_ids = split_position_ids[segment_idx]
        if segment_idx > 0:
            prev_position_ids = split_position_ids[segment_idx - 1]
            position_ids_with_pre_win = torch.cat(
                [prev_position_ids[:, -config.sliding_window:], cur_position_ids],
                dim=-1,
            )
        else:
            position_ids_with_pre_win = cur_position_ids
        last_kv_len = int(split_kv_len[0, last_rank_before_zz].item())
        if last_kv_len >= config.sliding_window:
            position_ids_last_src = split_position_ids[last_rank_before_zz][:, :last_kv_len]
            position_ids_last_win = position_ids_last_src[:, last_kv_len - config.sliding_window:last_kv_len]
        elif last_rank_before_zz > 0:
            prev_last_kv_len = int(split_kv_len[0, last_rank_before_zz - 1].item())
            prev_last_src = split_position_ids[last_rank_before_zz - 1][:, :prev_last_kv_len]
            last_src = split_position_ids[last_rank_before_zz][:, :last_kv_len]
            position_ids_last_win = torch.cat(
                [prev_last_src[:, -(config.sliding_window - last_kv_len):], last_src[:, :last_kv_len]],
                dim=-1,
            )
        else:
            position_ids_last_win = split_position_ids[last_rank_before_zz][:, :config.sliding_window]

        ori_kv_len_val = segment_len + config.sliding_window if segment_idx > 0 else segment_len
        ori_kv_len = torch.tensor([ori_kv_len_val], dtype=torch.int32, device=position_ids.device)
        slot_mapping_ori_kv = get_slot_mapping_from_block_table(
            ori_kv_len,
            position_ids_with_pre_win,
            attn_metadata["block_table"]["full_kv"],
        )
        actual_seq_k_val = (segment_idx + 1) * segment_len
        actual_seq_k = torch.full([batch_size], actual_seq_k_val, dtype=torch.int32, device=position_ids.device)
        actual_seq_q = torch.full([batch_size], segment_len, dtype=torch.int32, device=position_ids.device)
        cu_seq_lens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=position_ids.device), actual_seq_q])

        branch = {
            "is_start": segment_idx == 0,
            "is_end": segment_idx == last_rank_before_zz,
            "cur_kv_len": int(split_kv_len[0, segment_idx].item()),
            "block_table": dict(attn_metadata["block_table"]),
            "full_kv_cache": attn_metadata["full_kv_cache"],
            "kernel_metadata": {},
            "position_ids_with_pre_win": position_ids_with_pre_win,
            "position_ids_last_win": position_ids_last_win,
            "position_ids_cur": split_position_ids[segment_idx],
            "last_kv_len": last_kv_len,
            "slot_mapping_ori_kv": slot_mapping_ori_kv,
            "actual_seq_k": actual_seq_k,
            "actual_seq_q": actual_seq_q,
            "cu_seq_lens_q": cu_seq_lens_q,
            "cmp_out_pad": {},
            "cmp_in_offset": {},
        }
        branch["kernel_metadata"]["c1a_metadata"] = compute_sas_metadata(cu_seq_lens_q, actual_seq_k, batch_size, 1)

        for ratio in ratios:
            cmp_meta = _get_cp_cmp_param(
                segment_idx,
                attn_metadata,
                split_list_hidden,
                split_position_ids,
                split_kv_len,
                ratio,
                paged_cache=paged_cache,
                get_padded_slot_mapping_from_block_table=get_padded_slot_mapping_from_block_table,
            )
            ratio_key = str(ratio)
            branch.setdefault("cu_seq_lens", {})[ratio_key] = cmp_meta["cu_seq_lens"]
            branch.setdefault("seq_used_q_cmp", {})[ratio_key] = cmp_meta["seq_used_q"]
            branch.setdefault("start_pos_cmp", {})[ratio_key] = cmp_meta["start_pos"]
            branch.setdefault("position_ids_cmp_for_rope", {})[ratio_key] = cmp_meta["position_ids_cmp_for_rope"]
            branch.setdefault("slot_mapping_cmp_local", {})[ratio_key] = cmp_meta["slot_mapping_cmp"]
            branch.setdefault("comp_lens", {})[ratio_key] = cmp_meta["comp_lens"]
            branch["kernel_metadata"][f"c{ratio}a_metadata"] = compute_sas_metadata(
                branch["cu_seq_lens_q"],
                branch["actual_seq_k"],
                batch_size,
                ratio,
            )
            if ratio == 4 and attn_metadata["block_table"].get("c4a_cmp_kv") is not None:
                branch["kernel_metadata"]["lightning_indexer_quant"] = compute_li_metadata(
                    branch["actual_seq_q"],
                    branch["actual_seq_k"],
                    attn_metadata["block_table"]["c4a_cmp_kv"],
                )
        attn_metadata[zigzag_flag] = branch

    slot_mapping_cmp_dict = {}
    for ratio in ratios:
        ratio_key = str(ratio)
        slot_mapping_cmp_list = []
        in_lens = []
        for segment_idx in range(cp_segment_num):
            cur_position_ids = split_position_ids[segment_idx]
            cur_in_len = segment_len
            if segment_idx > 0:
                cur_in_len += int(cur_position_ids[:, 0].item() % ratio) + (ratio if ratio == 4 else 0)
            in_lens.append(cur_in_len)
        out_lens = [min(in_len, in_len // ratio + batch_size) for in_len in in_lens]
        max_out_len = max(out_lens)
        for zigzag_flag in ["prev", "next"]:
            branch = attn_metadata[zigzag_flag]
            segment_idx = cp_rank if zigzag_flag == "prev" else 2 * cp_size - 1 - cp_rank
            cur_slot_mapping = attn_metadata[zigzag_flag]["slot_mapping_cmp_local"][ratio_key]
            pad_len = max_out_len - out_lens[segment_idx]
            if pad_len > 0:
                pad_tensor = torch.full([pad_len], -1, dtype=torch.int32, device=position_ids.device)
                cur_slot_mapping = torch.cat([pad_tensor, cur_slot_mapping], dim=0)
            branch.setdefault("cmp_pad_len", {})[ratio_key] = pad_len
            li_pad = torch.zeros(
                (pad_len, config.index_head_dim),
                dtype=torch.bfloat16,
                device=position_ids.device,
            )
            sfa_pad = torch.zeros(
                (pad_len, config.head_dim),
                dtype=torch.bfloat16,
                device=position_ids.device,
            )
            branch["cmp_out_pad"][ratio_key] = (li_pad, sfa_pad)
            if segment_idx == 0:
                cmp_in_offset = torch.zeros([batch_size], dtype=torch.int32, device=position_ids.device)
            else:
                cmp_in_offset = torch.full(
                    [batch_size],
                    config.sliding_window,
                    dtype=torch.int32,
                    device=position_ids.device,
                ) - branch["comp_lens"][ratio_key]
            branch["cmp_in_offset"][ratio_key] = int(cmp_in_offset[0].item())
            slot_mapping_cmp_list.append(cur_slot_mapping)
        cur_slot_mapping_cmp = torch.cat(slot_mapping_cmp_list, dim=0)
        all_slot_mapping_cmp = cur_slot_mapping_cmp.new_empty([cur_slot_mapping_cmp.shape[0] * cp_size])
        dist.all_gather_into_tensor(all_slot_mapping_cmp, cur_slot_mapping_cmp, group=cp_group)
        all_slot_mapping_cmp = all_slot_mapping_cmp.view(-1, cur_slot_mapping_cmp.shape[0] // 2)[reverse_index]
        slot_mapping_cmp_dict[ratio_key] = all_slot_mapping_cmp.flatten(0, 1)
    attn_metadata["cp_metadata"]["slot_mapping_cmp"] = slot_mapping_cmp_dict
    return attn_metadata


def _get_cp_cmp_param(
    segment_idx,
    attn_metadata,
    split_list_hidden,
    split_position_ids,
    split_kv_len,
    ratio,
    *,
    paged_cache,
    get_padded_slot_mapping_from_block_table: Callable,
):
    batch_size = split_kv_len.shape[0]
    cur_kv_len = split_kv_len[:, segment_idx]
    cur_segment_len = split_list_hidden[segment_idx]
    cur_position_ids = split_position_ids[segment_idx]
    if segment_idx == 0 or cur_segment_len == 0:
        comp_len = torch.zeros([batch_size], dtype=torch.int32, device=cur_position_ids.device)
    else:
        overlap_len = ratio if ratio == 4 else 0
        comp_len = cur_position_ids[:, 0].to(dtype=torch.int32) % ratio + overlap_len
    seq_used_q = cur_kv_len.to(dtype=torch.int32) + comp_len
    seq_used_q = torch.where(cur_kv_len > 0, seq_used_q, torch.zeros_like(seq_used_q))
    cu_seq_lens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=cur_position_ids.device),
        (torch.full_like(cur_kv_len, cur_segment_len, dtype=torch.int32) + comp_len),
    ])
    if segment_idx == 0:
        start_pos = torch.zeros([batch_size], dtype=torch.int32, device=cur_position_ids.device)
    else:
        start_pos = torch.full(
            [batch_size],
            sum(split_list_hidden[:segment_idx]),
            dtype=torch.int32,
            device=cur_position_ids.device,
        ) - comp_len
    compressed_len, position_ids_cmp = paged_cache.get_compressed_position_ids(
        start_pos, seq_used_q, cu_seq_lens, ratio
    )
    slot_mapping_cmp = get_padded_slot_mapping_from_block_table(
        compressed_len,
        position_ids_cmp.unsqueeze(0),
        attn_metadata["block_table"][f"c{ratio}a_cmp_kv"],
    )
    if ratio == 4 and segment_idx > 0 and compressed_len.numel() > 0:
        offsets = torch.nn.functional.pad(torch.cumsum(compressed_len, dim=0, dtype=torch.int32), (1, 0))[:-1]
        slot_mapping_cmp[offsets.to(torch.long)] = -1
    return {
        "compressed_len": compressed_len.to(dtype=torch.int32),
        "cu_seq_lens": cu_seq_lens.to(dtype=torch.int32),
        "seq_used_q": seq_used_q.to(dtype=torch.int32),
        "start_pos": start_pos.to(dtype=torch.int32),
        "position_ids_cmp_for_rope": (position_ids_cmp * ratio).to(dtype=torch.long).unsqueeze(0),
        "slot_mapping_cmp": slot_mapping_cmp.to(dtype=torch.int32),
        "comp_lens": comp_len.to(dtype=torch.int32),
    }


def deepseek_v4_cp_get_window(attn, hidden_states: torch.Tensor, attn_inputs: dict, *, apply_partial_rotary: Callable):
    cp_group = attn.hccl_comm_dict.get("cp_group")
    if cp_group is None:
        raise ValueError("CP prefill requires cp_group.")
    cp_rank = dist.get_rank(group=cp_group)
    q_len = hidden_states.shape[1] // 2
    x_prev_cur, x_next_cur = hidden_states.split(q_len, dim=1)
    cur_segments = {}
    cur_win_list = []
    for flag, x_cur in (("prev", x_prev_cur), ("next", x_next_cur)):
        cur_kv_len = int(attn_inputs[flag]["cur_kv_len"])
        cur_segments[flag] = x_cur
        if cur_kv_len <= 0:
            cur_win = x_cur.new_zeros((hidden_states.shape[0], attn.sliding_window, hidden_states.shape[-1]))
        else:
            cur_segment = x_cur[:, :cur_kv_len, :]
            if cur_kv_len >= attn.sliding_window:
                cur_win = cur_segment[:, cur_kv_len - attn.sliding_window:cur_kv_len, :]
            else:
                pad = x_cur.new_zeros(
                    (hidden_states.shape[0], attn.sliding_window - cur_kv_len, hidden_states.shape[-1])
                )
                cur_win = torch.cat([cur_segment, pad], dim=1)
        cur_win_list.append(cur_win)
    local_win = torch.cat(cur_win_list, dim=0).contiguous()
    all_win = local_win.new_empty((local_win.shape[0] * attn.cp_size, *local_win.shape[1:]))
    dist.all_gather_into_tensor(all_win, local_win, group=cp_group)
    reverse_index = attn_inputs["cp_metadata"]["reverse_index"]
    all_win = all_win.view(-1, hidden_states.shape[0], attn.sliding_window, hidden_states.shape[-1])[reverse_index]

    x_prev = cur_segments["prev"]
    if not attn_inputs["prev"]["is_start"]:
        prev_pre_win = all_win[cp_rank - 1]
        x_prev = torch.cat([prev_pre_win, x_prev], dim=1)

    x_next = cur_segments["next"]
    if not attn_inputs["next"]["is_start"]:
        next_pre_win = all_win[2 * attn.cp_size - cp_rank - 2]
        x_next = torch.cat([next_pre_win, x_next], dim=1)

    last_rank = attn_inputs["cp_metadata"]["last_rank"]
    prev_meta = attn_inputs["prev"]
    last_kv_len = int(prev_meta["last_kv_len"])
    if last_kv_len >= attn.sliding_window:
        last_win = all_win[last_rank]
    elif last_rank == 0:
        last_win = all_win[last_rank]
    else:
        last_win = all_win[last_rank, :, :last_kv_len, :]
        second_last_win = all_win[last_rank - 1]
        last_win = torch.cat([second_last_win[:, -(attn.sliding_window - last_kv_len):, :], last_win], dim=1)

    last_position_ids = prev_meta["position_ids_last_win"].to(device=last_win.device, dtype=torch.long)
    cos_last, sin_last = attn._get_rotary_by_position_ids(
        last_win,
        last_position_ids,
        use_compress=attn.compress_ratio > 1,
    )
    last_win_kv = attn.wkv(last_win)
    last_win_kv = attn.kv_norm(last_win_kv)
    last_win_kv = apply_partial_rotary(last_win_kv, cos_last, sin_last, attn.partial_slice)
    return {"prev": x_prev, "next": x_next}, last_win_kv


def _gather_cp_variable_branch_tensor(local_tensor, local_branch_lens, cp_size, cp_group, reverse_index):
    gathered = local_tensor.new_empty((local_tensor.shape[0] * cp_size, local_tensor.shape[-1]))
    dist.all_gather_into_tensor(gathered, local_tensor, group=cp_group)
    local_branch_lens_tensor = torch.tensor(local_branch_lens, dtype=torch.int32, device=local_tensor.device)
    all_branch_lens = local_branch_lens_tensor.new_empty((cp_size * 2,))
    dist.all_gather_into_tensor(all_branch_lens, local_branch_lens_tensor, group=cp_group)
    all_branch_lens = all_branch_lens.view(cp_size, 2)
    gathered = gathered.view(cp_size, local_tensor.shape[0], local_tensor.shape[-1])
    segments = []
    for rank_idx in range(cp_size):
        prev_len = int(all_branch_lens[rank_idx, 0].item())
        next_len = int(all_branch_lens[rank_idx, 1].item())
        rank_tensor = gathered[rank_idx]
        segments.append(rank_tensor[:prev_len])
        segments.append(rank_tensor[prev_len: prev_len + next_len])
    return torch.cat([segments[int(idx)] for idx in reverse_index.tolist()], dim=0)


def deepseek_v4_cp_run_sfa_compressor(attn, x_segments: dict, past_key_values, attn_inputs: dict):
    ratio_key = str(attn.compress_ratio)
    cp_group = attn.hccl_comm_dict.get("cp_group")
    state_cache = attn_inputs["sfa_state_cache"]
    cur_kv_state = state_cache.clone().flatten(0, -3).flatten(-2)
    cmp_outputs = []
    local_branch_lens = []
    for flag in ["prev", "next"]:
        branch = attn_inputs[flag]
        cmp_in_offset = int(branch["cmp_in_offset"][ratio_key])
        x_seg_full = x_segments[flag]
        x_seg = x_seg_full[:, cmp_in_offset:] if cmp_in_offset > 0 else x_seg_full
        cmp_cos, cmp_sin = attn._get_rotary_by_position_ids(
            x_seg, branch["position_ids_cmp_for_rope"][ratio_key], use_compress=True
        )
        state_block_table = past_key_values.get_cmp_state_block_table(
            attn.layer_idx,
            branch["start_pos_cmp"][ratio_key],
            branch["seq_used_q_cmp"][ratio_key],
            True,
        )
        cmp_out = attn.sfa_compressor(
            x_seg,
            cmp_cos,
            cmp_sin,
            state_cache=state_cache,
            state_block_table=state_block_table,
            cu_seqlens=branch["cu_seq_lens"][ratio_key],
            seq_used_q=branch["seq_used_q_cmp"][ratio_key],
            start_pos=branch["start_pos_cmp"][ratio_key],
        ).squeeze(0)
        cmp_pad = branch["cmp_out_pad"][ratio_key][1]
        if cmp_pad.numel() > 0:
            cmp_out = torch.cat([cmp_pad.to(dtype=cmp_out.dtype, device=cmp_out.device), cmp_out], dim=0)
        if branch["is_end"]:
            cur_kv_state = state_cache.flatten(0, -3).flatten(-2)
        cmp_outputs.append(cmp_out)
        local_branch_lens.append(cmp_out.shape[0])
    local_cmp = torch.cat(cmp_outputs, dim=0).contiguous()
    all_cmp = _gather_cp_variable_branch_tensor(
        local_cmp,
        local_branch_lens,
        attn.cp_size,
        cp_group,
        attn_inputs["cp_metadata"]["reverse_index"],
    )
    all_ks = cur_kv_state.new_empty((cur_kv_state.shape[0] * attn.cp_size, cur_kv_state.shape[-1]))
    dist.all_gather_into_tensor(all_ks, cur_kv_state, group=cp_group)
    last_ks = all_ks.view(attn.cp_size, -1, cur_kv_state.shape[-1])[attn_inputs["cp_metadata"]["last_rank_zz"]]
    state_cache[:] = last_ks.view_as(state_cache)
    slot_mapping = attn_inputs["cp_metadata"]["slot_mapping_cmp"][ratio_key]
    valid_mask = slot_mapping >= 0
    if valid_mask.any():
        attn.scatter_nd_update(
            attn_inputs["cmp_kv_cache"].view(-1, attn.head_dim),
            slot_mapping[valid_mask].reshape(-1, 1),
            all_cmp[valid_mask],
        )


def deepseek_v4_cp_run_indexer(
    attn,
    hidden_states,
    qa,
    position_embeddings,
    x_segments,
    past_key_values,
    attn_inputs,
    *,
    rotate_activation: Callable,
    dynamic_quant_per_token: Callable,
    apply_partial_rotary: Callable,
):
    ratio_key = str(attn.compress_ratio)
    cp_group = attn.hccl_comm_dict.get("cp_group")
    li_state_cache = attn_inputs["li_state_cache"]
    cur_kv_state = li_state_cache.clone().flatten(0, -3).flatten(-2)
    q_len = hidden_states.shape[1] // 2
    qa_prev, qa_next = qa.split(q_len, dim=1)
    cos_main, sin_main = position_embeddings
    cos_prev, cos_next = cos_main.split(q_len, dim=1)
    sin_prev, sin_next = sin_main.split(q_len, dim=1)

    li_outputs = []
    local_branch_lens = []
    for flag in ["prev", "next"]:
        branch = attn_inputs[flag]
        cmp_in_offset = int(branch["cmp_in_offset"][ratio_key])
        x_seg_full = x_segments[flag]
        x_seg = x_seg_full[:, cmp_in_offset:] if cmp_in_offset > 0 else x_seg_full
        cmp_cos, cmp_sin = attn._get_rotary_by_position_ids(
            x_seg, branch["position_ids_cmp_for_rope"][ratio_key], use_compress=True
        )
        state_block_table = past_key_values.get_cmp_state_block_table(
            attn.layer_idx,
            branch["start_pos_cmp"][ratio_key],
            branch["seq_used_q_cmp"][ratio_key],
            True,
        )
        li_kv = attn.indexer.compressor(
            x_seg,
            cmp_cos,
            cmp_sin,
            state_cache=li_state_cache,
            state_block_table=state_block_table,
            cu_seqlens=branch["cu_seq_lens"][ratio_key],
            seq_used_q=branch["seq_used_q_cmp"][ratio_key],
            start_pos=branch["start_pos_cmp"][ratio_key],
            apply_indexer_rotate=False,
        ).squeeze(0)
        li_pad = branch["cmp_out_pad"][ratio_key][0]
        if li_pad.numel() > 0:
            li_kv = torch.cat([li_pad.to(dtype=li_kv.dtype, device=li_kv.device), li_kv], dim=0)
        if branch["is_end"]:
            cur_kv_state = li_state_cache.flatten(0, -3).flatten(-2)
        li_outputs.append(li_kv)
        local_branch_lens.append(li_kv.shape[0])

    local_li = torch.cat(li_outputs, dim=0).contiguous()
    all_li = _gather_cp_variable_branch_tensor(
        local_li,
        local_branch_lens,
        attn.cp_size,
        cp_group,
        attn_inputs["cp_metadata"]["reverse_index"],
    )
    if all_li.numel() > 0:
        all_li = rotate_activation(all_li, attn.indexer.hadamard_matrix)
    kv_quant, k_scale = dynamic_quant_per_token(all_li.contiguous())
    k_scale = k_scale.squeeze(-1).to(torch.float16)

    all_ks = cur_kv_state.new_empty((cur_kv_state.shape[0] * attn.cp_size, cur_kv_state.shape[-1]))
    dist.all_gather_into_tensor(all_ks, cur_kv_state, group=cp_group)
    last_ks = all_ks.view(attn.cp_size, -1, cur_kv_state.shape[-1])[attn_inputs["cp_metadata"]["last_rank_zz"]]
    li_state_cache[:] = last_ks.view_as(li_state_cache)
    slot_mapping = attn_inputs["cp_metadata"]["slot_mapping_cmp"][ratio_key]
    valid_mask = slot_mapping >= 0
    if valid_mask.any():
        attn.scatter_nd_update(
            attn_inputs["li_key_dequant_scale"].view(-1, attn_inputs["li_key_dequant_scale"].shape[-1]),
            slot_mapping[valid_mask].reshape(-1, 1),
            k_scale[valid_mask].view(-1, attn_inputs["li_key_dequant_scale"].shape[-1]),
        )
        attn.scatter_nd_update(
            attn_inputs["li_cmp_kv"].view(-1, attn.indexer.head_dim),
            slot_mapping[valid_mask].reshape(-1, 1),
            kv_quant[valid_mask],
        )

    prev_weight_source = x_segments["prev"][:, :q_len] if attn_inputs["prev"].get("is_start") else x_segments["prev"][:, -q_len:]
    next_weight_source = x_segments["next"][:, -q_len:]
    branch_inputs = {
        "prev": (prev_weight_source, qa_prev, cos_prev, sin_prev),
        "next": (next_weight_source, qa_next, cos_next, sin_next),
    }
    topk_dict = {}
    for flag, (x_cur, qa_cur, cos_cur, sin_cur) in branch_inputs.items():
        weights = attn.indexer.weights_proj(x_cur.reshape(-1, attn.config.hidden_size).to(torch.bfloat16))
        weights = weights.view(1, q_len, attn.indexer.n_heads) * (
            attn.indexer.softmax_scale * attn.indexer.n_heads ** -0.5
        )
        qr_flat = qa_cur.reshape(-1, attn.q_lora_rank).to(torch.bfloat16)
        qr_quant, qr_scale = dynamic_quant_per_token(qr_flat)
        q_li = attn.indexer.wq_b(qr_quant, qr_scale)
        q_li = q_li.view(1, q_len, attn.indexer.n_heads, attn.indexer.head_dim)
        q_li = apply_partial_rotary(q_li, cos_cur, sin_cur, attn.indexer.partial_slice)
        q_li = rotate_activation(q_li, attn.indexer.hadamard_matrix)
        q_flat = q_li.flatten(0, 1)
        q_quant, q_scale = dynamic_quant_per_token(q_flat)
        q_scale = q_scale.to(torch.float16)
        li_metadata = attn_inputs[flag]["kernel_metadata"]["lightning_indexer_quant"]
        topk_idxs, _ = attn.indexer.quant_lightning_indexer(
            query=q_quant,
            key=attn_inputs["li_cmp_kv"],
            weights=weights.flatten(0, 1).to(torch.float16),
            query_dequant_scale=q_scale,
            key_dequant_scale=attn_inputs["li_key_dequant_scale"].squeeze(-2),
            actual_seq_lengths_query=attn_inputs[flag]["actual_seq_q"],
            actual_seq_lengths_key=attn_inputs[flag]["actual_seq_k"],
            block_table=attn_inputs["c4a_cmp_kv_block_table"],
            layout_key='PA_BSND',
            sparse_count=attn.indexer.index_topk,
            sparse_mode=3,
            layout_query="TND",
            cmp_ratio=attn.indexer.compress_ratio,
            key_quant_mode=0,
            query_quant_mode=0,
            metadata=li_metadata,
        )
        topk_dict[flag] = topk_idxs.view(q_flat.shape[0], -1, attn.indexer.index_topk)
    return topk_dict


def deepseek_v4_cp_run_prefill(
    attn,
    hidden_states,
    qa,
    q,
    past_key_values,
    attn_inputs,
    position_embeddings,
    *,
    apply_partial_rotary: Callable,
    rotate_activation: Callable,
    dynamic_quant_per_token: Callable,
):
    batch_size, seq_length = hidden_states.shape[:2]
    local_q_len = seq_length // 2
    q_dict = {
        "prev": q[:, :local_q_len],
        "next": q[:, local_q_len:],
    }
    x_segments, last_win_kv = deepseek_v4_cp_get_window(
        attn,
        hidden_states,
        attn_inputs,
        apply_partial_rotary=apply_partial_rotary,
    )
    kv_segments = {}
    for flag in ["prev", "next"]:
        x_seg = x_segments[flag]
        branch = attn_inputs[flag]
        cos_seg, sin_seg = attn._get_rotary_by_position_ids(
            x_seg,
            branch["position_ids_with_pre_win"],
            use_compress=attn.compress_ratio > 1,
        )
        kv_seg = attn.wkv(x_seg.reshape(-1, attn.config.hidden_size).to(torch.bfloat16))
        kv_seg = attn.kv_norm(kv_seg).view(batch_size, x_seg.shape[1], attn.head_dim)
        kv_seg = apply_partial_rotary(
            kv_seg.view(batch_size, x_seg.shape[1], 1, attn.head_dim),
            cos_seg,
            sin_seg,
            attn.partial_slice,
        ).view(batch_size, x_seg.shape[1], attn.head_dim)
        kv_segments[flag] = kv_seg

    full_kv_cache = attn_inputs["full_kv_cache"]
    slot_mapping_ori = torch.cat(
        [attn_inputs["prev"]["slot_mapping_ori_kv"], attn_inputs["next"]["slot_mapping_ori_kv"]],
        dim=0,
    )
    kv_full = torch.cat([kv_segments["prev"], kv_segments["next"]], dim=1).reshape(-1, attn.head_dim)
    attn.scatter_nd_update(full_kv_cache.view(-1, attn.head_dim), slot_mapping_ori.reshape(-1, 1), kv_full)
    past_key_values.update_win_kv(last_win_kv, attn.layer_idx, slot_mapping=attn_inputs["win_slot_mapping"])

    cmp_sparse_indices = {}
    if attn.compress_ratio > 1:
        deepseek_v4_cp_run_sfa_compressor(attn, x_segments, past_key_values, attn_inputs)
        if attn.indexer is not None:
            cmp_sparse_indices = deepseek_v4_cp_run_indexer(
                attn,
                hidden_states,
                qa,
                position_embeddings,
                x_segments,
                past_key_values,
                attn_inputs,
                rotate_activation=rotate_activation,
                dynamic_quant_per_token=dynamic_quant_per_token,
                apply_partial_rotary=apply_partial_rotary,
            )

    out_list = []
    cmp_block_key = f"c{attn.compress_ratio}a_cmp_kv"
    meta_key = f"c{attn.compress_ratio}a_metadata" if attn.compress_ratio > 1 else "c1a_metadata"
    for flag in ["prev", "next"]:
        q_flat = q_dict[flag].contiguous().view(-1, attn.num_heads, attn.head_dim)
        branch = attn_inputs[flag]
        out = attn._run_attn(
            q_flat,
            full_kv_cache,
            branch["block_table"]["full_kv"],
            branch["actual_seq_k"],
            batch_size,
            local_q_len,
            attn.compress_ratio,
            branch["cu_seq_lens_q"],
            attn_inputs.get("cmp_kv_cache"),
            branch["block_table"].get(cmp_block_key),
            cmp_sparse_indices.get(flag),
            q_lens=branch["actual_seq_q"],
            sas_metadata=branch["kernel_metadata"][meta_key],
        )
        out_list.append(out)
    past_key_values.update(
        torch.zeros((batch_size, local_q_len, attn.head_dim), dtype=torch.bfloat16, device=hidden_states.device),
        attn.layer_idx,
        attn_inputs["cu_q_lens"],
        actual_q_lens=attn_inputs["q_lens"],
    )
    return torch.cat(out_list, dim=1)
