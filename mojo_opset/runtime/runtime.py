from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mojo_opset.runtime.config import MojoConfig
from mojo_opset.runtime.generation import MojoSession
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class AttentionMetadata:
    q_lens: torch.Tensor
    cu_q_lens: torch.Tensor
    total_seq_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    key_caches: list[torch.Tensor]
    value_caches: list[torch.Tensor]
    is_prefill: bool


class PagedAttentionRuntimeState(MojoSession):
    def __init__(
        self,
        config: MojoConfig,
        batch_size: int,
        device,
        dtype,
        block_size: int = 128,
    ):
        self.config = config
        self.batch_size = batch_size
        self.num_layers = config.model_config.num_layers
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.num_kv_heads = getattr(config.model_config, "local_num_kv_heads", config.model_config.num_kv_heads)
        self.head_dim = config.model_config.head_dim

        self.max_blocks_per_seq = (config.model_config.max_position_embeddings + block_size - 1) // block_size
        total_blocks = batch_size * self.max_blocks_per_seq
        self.offsets = torch.arange(self.block_size, dtype=torch.int64, device=self.device)

        self.block_tables = torch.full(
            (batch_size, self.max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.total_seq_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        self.free_blocks = torch.arange(total_blocks, dtype=torch.int32, device=device)
        self.num_free_blocks = total_blocks

        cache_shape = (total_blocks, self.num_kv_heads, block_size, self.head_dim)
        self.key_caches = [None] * self.num_layers
        self.value_caches = [None] * self.num_layers

        kv_mirror_layers = getattr(config.model_config, "kv_mirror_layers", [])
        kv_mirror_imitated_layers = getattr(config.model_config, "kv_mirror_imitated_layers", [])
        mirror_map = {
            mirror_layer - 1: imitated_layer - 1
            for mirror_layer, imitated_layer in zip(kv_mirror_layers, kv_mirror_imitated_layers)
        }

        for layer_idx in range(self.num_layers):
            if layer_idx in mirror_map:
                source_layer_idx = mirror_map[layer_idx]
                if self.key_caches[source_layer_idx] is None or self.value_caches[source_layer_idx] is None:
                    raise ValueError(
                        f"Source layer {source_layer_idx + 1} for mirror layer {layer_idx + 1} must exist first."
                    )
                logger.debug(f"Layer {layer_idx + 1} is mirroring layer {source_layer_idx + 1}.")
                self.key_caches[layer_idx] = self.key_caches[source_layer_idx]
                self.value_caches[layer_idx] = self.value_caches[source_layer_idx]
                continue

            logger.debug(f"Creating new cache tensors for layer {layer_idx + 1}.")
            self.key_caches[layer_idx] = torch.zeros(cache_shape, dtype=dtype, device=device)
            self.value_caches[layer_idx] = torch.zeros(cache_shape, dtype=dtype, device=device)

    @property
    def kv_cache(self):
        return self

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        batch_size: int,
        *,
        block_size: int = 128,
        device=None,
        dtype=None,
    ) -> "PagedAttentionRuntimeState":
        if device is None:
            device = model.lm_head.weight.device
        if dtype is None:
            dtype = model.lm_head.weight.dtype
        return cls(
            model.config,
            batch_size,
            device=device,
            dtype=dtype,
            block_size=block_size,
        )

    def _allocate_blocks(self, num_blocks: int) -> torch.Tensor:
        if num_blocks > self.num_free_blocks:
            raise ValueError("PagedAttentionRuntimeState: Out of paged KV cache memory.")
        allocated = self.free_blocks[self.num_free_blocks - num_blocks : self.num_free_blocks]
        self.num_free_blocks -= num_blocks
        return allocated

    def _normalize_q_lens(self, q_lens: Optional[torch.Tensor]) -> torch.Tensor:
        if q_lens is None:
            return torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
        return q_lens.to(device=self.device, dtype=torch.int32)

    def _reserve(self, q_lens: torch.Tensor) -> torch.Tensor:
        q_lens = self._normalize_q_lens(q_lens)
        previous_total_seq_lens = self.total_seq_lens.clone()

        for batch_idx in range(self.batch_size):
            context_len = int(previous_total_seq_lens[batch_idx].item())
            append_len = int(q_lens[batch_idx].item())
            old_num_blocks = (context_len + self.block_size - 1) // self.block_size
            new_total_len = context_len + append_len
            new_num_blocks = (new_total_len + self.block_size - 1) // self.block_size

            if new_num_blocks > old_num_blocks:
                newly_allocated = self._allocate_blocks(new_num_blocks - old_num_blocks)
                self.block_tables[batch_idx, old_num_blocks:new_num_blocks] = newly_allocated

        self.total_seq_lens = previous_total_seq_lens + q_lens
        return previous_total_seq_lens

    def _build_positions(self, context_kv_lens: torch.Tensor, q_lens: torch.Tensor) -> torch.Tensor:
        positions = []
        for batch_idx in range(self.batch_size):
            start = int(context_kv_lens[batch_idx].item())
            query_len = int(q_lens[batch_idx].item())
            if query_len <= 0:
                continue
            positions.append(torch.arange(start, start + query_len, dtype=torch.int64, device=self.device))

        if not positions:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return torch.cat(positions, dim=0)

    def _build_slot_mapping(self, context_kv_lens: torch.Tensor, q_lens: torch.Tensor) -> torch.Tensor:
        slot_mapping_chunks = []
        for batch_idx in range(self.batch_size):
            query_len = q_lens[batch_idx]
            if query_len <= 0:
                continue

            curr_block_tables = self.block_tables[batch_idx]
            first_pad_index = torch.nonzero(curr_block_tables == -1)[0]
            curr_block_tables = curr_block_tables[:first_pad_index]

            block_start = curr_block_tables * self.block_size
            slot_mapping = (block_start.unsqueeze(-1) + self.offsets.unsqueeze(0)).flatten()

            start = context_kv_lens[batch_idx]
            slot_mapping_chunks.append(slot_mapping[start:start+query_len])

        if not slot_mapping_chunks:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return torch.cat(slot_mapping_chunks, dim=0)

    def _build_attention_metadata(
        self,
        *,
        cu_q_lens: Optional[torch.Tensor],
        context_kv_lens: torch.Tensor,
        q_lens: torch.Tensor,
    ) -> AttentionMetadata:
        return AttentionMetadata(
            cu_q_lens=cu_q_lens,
            q_lens=q_lens,
            total_seq_lens=self.total_seq_lens,
            block_tables=self.block_tables,
            slot_mapping=self._build_slot_mapping(context_kv_lens, q_lens),
            key_caches=self.key_caches,
            value_caches=self.value_caches,
            is_prefill=cu_q_lens is not None,
        )

    def prepare_prefill_inputs(
        self,
        input_ids: torch.Tensor,
        q_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        input_ids = input_ids.reshape(-1).to(device=self.device, dtype=torch.int64)
        q_lens = self._normalize_q_lens(q_lens)

        if int(q_lens.sum().item()) != input_ids.numel():
            raise ValueError(
                "Prefill input_ids length must match the sum of q_lens: "
                f"{input_ids.numel()} != {int(q_lens.sum().item())}"
            )

        context_kv_lens = self._reserve(q_lens)
        positions = self._build_positions(context_kv_lens, q_lens)
        cu_q_lens = F.pad(q_lens.cumsum(-1, dtype=torch.int32), (1, 0))
        attention_metadata = self._build_attention_metadata(
            cu_q_lens=cu_q_lens,
            context_kv_lens=context_kv_lens,
            q_lens=q_lens,
        )
        return input_ids, positions, attention_metadata

    def prepare_decode_inputs(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        input_ids = input_ids.reshape(-1).to(device=self.device, dtype=torch.int64)
        if input_ids.numel() != self.batch_size:
            raise ValueError(
                f"Decode input_ids must provide exactly one token per sequence: {input_ids.numel()} != {self.batch_size}"
            )

        q_lens = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
        cu_q_lens = F.pad(q_lens.cumsum(-1, dtype=torch.int32), (1, 0))
        positions = self.total_seq_lens.clone()
        context_kv_lens = self._reserve(q_lens)
        attention_metadata = self._build_attention_metadata(
            cu_q_lens=cu_q_lens,
            context_kv_lens=context_kv_lens,
            q_lens=q_lens,
        )
        return input_ids, positions, attention_metadata


class PagedAttentionGenerationModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        block_size: int = 128,
        session_cls: type[PagedAttentionRuntimeState] = PagedAttentionRuntimeState,
    ):
        super().__init__()
        self.model = model
        self.block_size = block_size
        self.session_cls = session_cls

    def _new_session(
        self,
        input_ids: torch.Tensor,
        context_input_len: Optional[torch.Tensor],
    ) -> PagedAttentionRuntimeState:
        batch_size = int(context_input_len.size(0)) if context_input_len is not None else int(input_ids.numel())
        return self.session_cls.from_model(self.model, batch_size, block_size=self.block_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        context_input_len: torch.Tensor = None,
        session: PagedAttentionRuntimeState = None,
        **kwargs,
    ):
        lm_head_indices = None
        if session is None:
            session = self._new_session(input_ids, context_input_len)

        if context_input_len is not None:
            model_inputs = session.prepare_prefill_inputs(input_ids, context_input_len)
            cu_q_lens = model_inputs[-1].cu_q_lens[1:]
            lm_head_indices = cu_q_lens - 1
        else:
            model_inputs = session.prepare_decode_inputs(input_ids)

        logits = self.model(*model_inputs, lm_head_indices=lm_head_indices)
        return logits, session


class DeepseekSparseAttentionRuntimeState(MojoSession):
    """DeepSeek-V4 runtime state that pre-computes decode metadata outside graph."""

    def __init__(self, paged_cache, config, cp_size=1, global_rank=0, hccl_comm_dict=None):
        self.paged_cache = paged_cache
        self.config = config
        self.cp_size = cp_size
        self.global_rank = global_rank
        self.hccl_comm_dict = hccl_comm_dict or {}

    @property
    def kv_cache(self):
        return self.paged_cache

    def _get_first_layer_for_ratio(self, ratio):
        for layer_idx in range(self.paged_cache.num_layers):
            layer_ratio = self.config.compress_ratios[layer_idx] if layer_idx < len(self.config.compress_ratios) else 0
            if layer_ratio == ratio:
                return layer_idx
        return None

    def _get_cp_rank(self, cp_group) -> int:
        if cp_group is None or not dist.is_initialized():
            return 0
        return dist.get_rank(group=cp_group)

    def _build_golden_style_attn_metadata(
        self,
        *,
        position_ids,
        context_lens,
        q_lens,
        cu_q_lens,
        current_seq_lens,
        start_pos,
        seq_used_q,
        shared_metadata,
        win_slot_mapping,
        full_kv_cache,
        is_prefill,
        batch_size,
        decode_fast_path=False,
    ):
        pkv = self.paged_cache
        position_ids_c = {}
        block_table = {}
        slot_mapping = {}
        kernel_metadata = {}

        win_kv_cache, win_block_table = pkv.get_win_kv_for_decode(0)
        block_table["win_kv"] = win_block_table
        slot_mapping["win_kv"] = win_slot_mapping

        if is_prefill:
            block_table["full_kv"] = pkv.get_full_block_table(batch_size)
            slot_mapping["full_kv"] = pkv.get_full_slot_mapping(context_lens, q_lens)
            slot_mapping["full_kv_gather_indices"] = pkv.get_full_kv_gather_indices(context_lens, q_lens)

        for ratio, metadata in shared_metadata.items():
            attn_ratio = ratio if ratio > 1 else 1
            kernel_metadata[f"c{attn_ratio}a_metadata"] = metadata.get("sas_metadata")
            if ratio <= 1:
                continue

            position_ids_cmp = metadata.get("position_ids_cmp")
            if position_ids_cmp is not None:
                position_ids_c[str(ratio)] = position_ids_cmp * ratio

            layer_idx = self._get_first_layer_for_ratio(ratio)
            if layer_idx is None:
                continue

            cmp_block_table_key = f"c{ratio}a_cmp_kv"
            block_table[cmp_block_table_key] = pkv.get_cmp_kv_block_table(layer_idx)
            if decode_fast_path:
                slot_mapping[cmp_block_table_key] = pkv.get_cmp_slot_mapping_decode(
                    layer_idx,
                    start_pos,
                    compressed_len=metadata.get("compressed_len"),
                    position_ids_cmp=position_ids_cmp,
                )
                block_table[f"c{ratio}a_cmp_state"] = pkv.get_cmp_state_block_table_decode(
                    layer_idx, start_pos
                )
            else:
                slot_mapping[cmp_block_table_key] = pkv.get_cmp_slot_mapping(
                    layer_idx,
                    start_pos,
                    seq_used_q,
                    cu_seqlens_q=cu_q_lens,
                    compressed_len=metadata.get("compressed_len"),
                    position_ids_cmp=position_ids_cmp,
                )
                block_table[f"c{ratio}a_cmp_state"] = pkv.get_cmp_state_block_table(
                    layer_idx, start_pos, seq_used_q, is_prefill
                )

            if ratio == 4:
                block_table["c4a_cmp_kv"] = pkv.get_c4a_cmp_kv_block_table(layer_idx)
                kernel_metadata["lightning_indexer_quant"] = metadata.get("li_metadata")

        return {
            "batch_size_per_rank": batch_size,
            "position_ids": position_ids.to(dtype=torch.int32),
            "kv_len": current_seq_lens.to(dtype=torch.int32),
            "actual_seq_q": cu_q_lens[1:],
            "actual_seq_k": current_seq_lens.to(dtype=torch.int32),
            "cu_seq_lens_q": cu_q_lens,
            "seq_used_q": seq_used_q,
            "start_pos": start_pos,
            "position_ids_c": position_ids_c,
            "block_table": block_table,
            "slot_mapping": slot_mapping,
            "kernel_metadata": kernel_metadata,
            "is_prefill": is_prefill,
            "win_kv_cache": win_kv_cache,
            "full_kv_cache": full_kv_cache,
        }

    def _get_slot_mapping_from_block_table(self, seq_lens, position_ids, block_table):
        seq_lens = seq_lens.to(dtype=torch.int32, device=position_ids.device)
        batch_size = seq_lens.shape[0]
        if batch_size == 0:
            return torch.empty((0,), dtype=torch.int32, device=position_ids.device)
        max_len = position_ids.shape[-1]
        valid_mask = torch.arange(max_len, dtype=torch.int32, device=position_ids.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        block_idx = (position_ids // self.paged_cache.block_size).to(torch.long)
        offset = (position_ids % self.paged_cache.block_size).to(torch.int32)
        row_idx = torch.arange(batch_size, dtype=torch.long, device=position_ids.device).unsqueeze(1).expand_as(block_idx)
        slots = block_table[row_idx, block_idx] * self.paged_cache.block_size + offset
        return slots[valid_mask].to(dtype=torch.int32)

    def _get_padded_slot_mapping_from_block_table(self, seq_lens, position_ids, block_table):
        seq_lens = seq_lens.to(dtype=torch.int32, device=position_ids.device)
        batch_size = seq_lens.shape[0]
        if batch_size == 0:
            return torch.empty((0,), dtype=torch.int32, device=position_ids.device)
        flat_position_ids = position_ids.reshape(-1).to(dtype=torch.int32)
        row_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int32, device=position_ids.device),
            seq_lens,
        )
        pad_length = flat_position_ids.shape[0] - row_indices.shape[0]
        if pad_length > 0:
            valid_position_ids = flat_position_ids[:-pad_length]
        else:
            valid_position_ids = flat_position_ids
        block_idx = (valid_position_ids // self.paged_cache.block_size).to(torch.long)
        offset = (valid_position_ids % self.paged_cache.block_size).to(torch.int32)
        row_indices = row_indices.to(torch.long)
        slot_mapping = block_table[row_indices, block_idx] * self.paged_cache.block_size + offset
        if pad_length > 0:
            pad_tensor = torch.full(
                (pad_length,),
                -1,
                dtype=torch.int32,
                device=position_ids.device,
            )
            slot_mapping = torch.cat([slot_mapping.to(torch.int32), pad_tensor], dim=0)
        return slot_mapping.to(dtype=torch.int32)

    @staticmethod
    def _get_zigzag_idx(origin_idx, cp_segment_num):
        midpoint = cp_segment_num // 2 - 1
        if origin_idx <= midpoint:
            return origin_idx, "prev"
        return midpoint + 1 - (origin_idx - midpoint), "next"

    def _get_cp_cmp_param(
        self,
        segment_idx,
        attn_metadata,
        split_list_hidden,
        split_position_ids,
        split_kv_len,
        ratio,
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
        compressed_len, position_ids_cmp = self.paged_cache.get_compressed_position_ids(
            start_pos, seq_used_q, cu_seq_lens, ratio
        )
        slot_mapping_cmp = self._get_padded_slot_mapping_from_block_table(
            compressed_len,
            position_ids_cmp.unsqueeze(0),
            attn_metadata["block_table"][f"c{ratio}a_cmp_kv"],
        )
        if ratio == 4 and segment_idx > 0 and compressed_len.numel() > 0:
            offsets = torch.nn.functional.pad(
                torch.cumsum(compressed_len, dim=0, dtype=torch.int32), (1, 0)
            )[:-1]
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

    def _build_cp_metadata(self, input_ids, attn_metadata):
        if self.cp_size <= 1 or not dist.is_initialized():
            attn_metadata["cp_metadata"] = None
            return attn_metadata
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise NotImplementedError("Mojo CP prefill currently only supports batch_size=1.")
        cp_group = self.hccl_comm_dict.get("cp_group")
        if cp_group is None:
            raise ValueError("cp_size > 1 requires cp_group to be initialized.")
        cp_rank = self._get_cp_rank(cp_group)
        cp_segment_num = self.cp_size * 2
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
        last_rank_zz, last_rank_flag = self._get_zigzag_idx(last_rank_before_zz, cp_segment_num)
        cp_input_dict = {
            "split_list": split_list_hidden,
            "zigzag_idx": zigzag_idx,
            "reverse_index": reverse_index,
            "split_kv_len": split_kv_len,
            "last_rank": last_rank_before_zz,
            "last_rank_flag": last_rank_flag,
            "last_rank_zz": last_rank_zz,
        }
        attn_metadata["cp_metadata"] = cp_input_dict
        attn_metadata["prev"] = {}
        attn_metadata["next"] = {}

        ratios = sorted({r for r in self.config.compress_ratios if r > 1})
        for zigzag_flag in ["prev", "next"]:
            segment_idx = cp_rank if zigzag_flag == "prev" else 2 * self.cp_size - 1 - cp_rank
            cur_position_ids = split_position_ids[segment_idx]
            if segment_idx > 0:
                prev_position_ids = split_position_ids[segment_idx - 1]
                position_ids_with_pre_win = torch.cat(
                    [prev_position_ids[:, -self.config.sliding_window:], cur_position_ids],
                    dim=-1,
                )
            else:
                position_ids_with_pre_win = cur_position_ids
            last_kv_len = int(split_kv_len[0, last_rank_before_zz].item())
            if last_kv_len >= self.config.sliding_window:
                position_ids_last_src = split_position_ids[last_rank_before_zz][:, :last_kv_len]
                position_ids_last_win = position_ids_last_src[:, last_kv_len - self.config.sliding_window:last_kv_len]
            elif last_rank_before_zz > 0:
                prev_last_kv_len = int(split_kv_len[0, last_rank_before_zz - 1].item())
                prev_last_src = split_position_ids[last_rank_before_zz - 1][:, :prev_last_kv_len]
                last_src = split_position_ids[last_rank_before_zz][:, :last_kv_len]
                position_ids_last_win = torch.cat([
                    prev_last_src[:, -(self.config.sliding_window - last_kv_len):],
                    last_src[:, :last_kv_len],
                ], dim=-1)
            else:
                position_ids_last_win = split_position_ids[last_rank_before_zz][:, :self.config.sliding_window]

            ori_kv_len_val = segment_len + self.config.sliding_window if segment_idx > 0 else segment_len
            ori_kv_len = torch.tensor([ori_kv_len_val], dtype=torch.int32, device=position_ids.device)
            slot_mapping_ori_kv = self._get_slot_mapping_from_block_table(
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
            branch["kernel_metadata"]["c1a_metadata"] = self._compute_sas_metadata(
                cu_seq_lens_q, actual_seq_k, batch_size, 1
            )

            for ratio in ratios:
                cmp_meta = self._get_cp_cmp_param(
                    segment_idx, attn_metadata, split_list_hidden, split_position_ids, split_kv_len, ratio
                )
                branch.setdefault("cu_seq_lens", {})[str(ratio)] = cmp_meta["cu_seq_lens"]
                branch.setdefault("seq_used_q_cmp", {})[str(ratio)] = cmp_meta["seq_used_q"]
                branch.setdefault("start_pos_cmp", {})[str(ratio)] = cmp_meta["start_pos"]
                branch.setdefault("position_ids_cmp_for_rope", {})[str(ratio)] = cmp_meta["position_ids_cmp_for_rope"]
                branch.setdefault("slot_mapping_cmp_local", {})[str(ratio)] = cmp_meta["slot_mapping_cmp"]
                branch.setdefault("comp_lens", {})[str(ratio)] = cmp_meta["comp_lens"]
                branch["kernel_metadata"][f"c{ratio}a_metadata"] = self._compute_sas_metadata(
                    branch["cu_seq_lens_q"],
                    branch["actual_seq_k"],
                    batch_size,
                    ratio,
                )
                if ratio == 4 and attn_metadata["block_table"].get("c4a_cmp_kv") is not None:
                    branch["kernel_metadata"]["lightning_indexer_quant"] = self._compute_li_metadata(
                        branch["actual_seq_q"],
                        branch["actual_seq_k"],
                        attn_metadata["block_table"]["c4a_cmp_kv"],
                    )
            attn_metadata[zigzag_flag] = branch

        slot_mapping_cmp_dict = {}
        for ratio in ratios:
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
                segment_idx = cp_rank if zigzag_flag == "prev" else 2 * self.cp_size - 1 - cp_rank
                cur_slot_mapping = attn_metadata[zigzag_flag]["slot_mapping_cmp_local"][str(ratio)]
                pad_len = max_out_len - out_lens[segment_idx]
                if pad_len > 0:
                    pad_tensor = torch.full([pad_len], -1, dtype=torch.int32, device=position_ids.device)
                    cur_slot_mapping = torch.cat([pad_tensor, cur_slot_mapping], dim=0)
                branch.setdefault("cmp_pad_len", {})[str(ratio)] = pad_len
                li_pad = torch.zeros(
                    (pad_len, self.config.index_head_dim),
                    dtype=torch.bfloat16,
                    device=position_ids.device,
                )
                sfa_pad = torch.zeros(
                    (pad_len, self.config.head_dim),
                    dtype=torch.bfloat16,
                    device=position_ids.device,
                )
                branch["cmp_out_pad"][str(ratio)] = (li_pad, sfa_pad)
                if segment_idx == 0:
                    cmp_in_offset = torch.zeros([batch_size], dtype=torch.int32, device=position_ids.device)
                else:
                    cmp_in_offset = torch.full(
                        [batch_size],
                        self.config.sliding_window,
                        dtype=torch.int32,
                        device=position_ids.device,
                    ) - branch["comp_lens"][str(ratio)]
                branch["cmp_in_offset"][str(ratio)] = int(cmp_in_offset[0].item())
                slot_mapping_cmp_list.append(cur_slot_mapping)
            cur_slot_mapping_cmp = torch.cat(slot_mapping_cmp_list, dim=0)
            all_slot_mapping_cmp = cur_slot_mapping_cmp.new_empty([cur_slot_mapping_cmp.shape[0] * self.cp_size])
            dist.all_gather_into_tensor(all_slot_mapping_cmp, cur_slot_mapping_cmp, group=cp_group)
            all_slot_mapping_cmp = all_slot_mapping_cmp.view(-1, cur_slot_mapping_cmp.shape[0] // 2)[reverse_index]
            slot_mapping_cmp_dict[str(ratio)] = all_slot_mapping_cmp.flatten(0, 1)
        attn_metadata["cp_metadata"]["slot_mapping_cmp"] = slot_mapping_cmp_dict
        return attn_metadata

    @classmethod
    def from_model(
        cls,
        model,
        batch_size,
        block_size=128,
        device=None,
        dtype=None,
        max_seq_len=None,
        pa_max_length=None,
        next_n=None,
    ):
        from mojo_opset.modeling.deepseekv4.mojo_deepseek_v4 import PagedDummyCache

        if device is None:
            device = model.lm_head.weight.device
        config = model.config
        if max_seq_len is None:
            max_seq_len = config.max_position_embeddings
        if pa_max_length is None:
            pa_max_length = config.pa_max_length
        if next_n is None:
            next_n = config.next_n

        cache_data = PagedDummyCache(
            config,
            batch_size=batch_size,
            device=str(device),
            block_size=block_size,
            max_seq_len=max_seq_len,
            pa_max_length=pa_max_length,
            next_n=next_n,
        )
        return cls(
            cache_data,
            config,
            cp_size=getattr(model, "cp_size", 1),
            global_rank=getattr(model, "global_rank", 0),
            hccl_comm_dict=getattr(model, "hccl_comm_dict", {}),
        )

    def prepare_prefill_inputs(self, input_ids, attention_mask=None, q_lens=None):
        pkv = self.paged_cache
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if q_lens is None:
            if attention_mask is not None:
                q_lens = attention_mask.to(device=device, dtype=torch.int32).sum(dim=-1).to(dtype=torch.int32)
            else:
                q_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

        context_lens = pkv.get_seq_length(0).to(device=device, dtype=torch.long)
        if attention_mask is not None:
            position_ids = (attention_mask.to(device=device, dtype=torch.long).cumsum(dim=-1) - 1).clamp(min=0)
            position_ids = position_ids.masked_fill(~attention_mask.to(device=device, dtype=torch.bool), 1)
        else:
            past_len = int(context_lens.max().item()) if context_lens.numel() > 0 else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long).unsqueeze(0)

        cu_q_lens = torch.arange(
            0, (batch_size + 1) * seq_len, step=seq_len,
            dtype=torch.int32, device=device,
        )

        start_pos = context_lens.to(dtype=torch.int32)
        seq_used_q = q_lens
        # Golden keeps kv_len as the real request length even when CP pads the
        # prefill sequence to segment alignment. Treating padding as valid KV
        # pollutes the tail window handed to decode.
        current_seq_lens = q_lens.to(dtype=torch.int32)

        unique_ratios = sorted(set(
            self.config.compress_ratios[l] if l < len(self.config.compress_ratios) else 0
            for l in range(pkv.num_layers)
        ))
        shared_metadata = {}
        for ratio in unique_ratios:
            attn_ratio = ratio if ratio > 1 else 1
            sas_meta = self._compute_sas_metadata(cu_q_lens, current_seq_lens, batch_size, attn_ratio)
            shared_metadata[ratio] = {"sas_metadata": sas_meta}
            if ratio > 1:
                compressed_len, position_ids_cmp = pkv.get_compressed_position_ids(
                    start_pos, seq_used_q, cu_q_lens, ratio,
                )
                cmp_rope_position_ids = (position_ids_cmp * ratio).to(dtype=torch.long).unsqueeze(0)
                shared_metadata[ratio].update({
                    "compressed_len": compressed_len,
                    "position_ids_cmp": position_ids_cmp,
                    "cmp_rope_position_ids": cmp_rope_position_ids,
                })

        full_kv_cache = pkv.init_full_buffer_c1a()
        win_slot_mapping = pkv.get_win_slot_mapping(context_lens, q_lens, pad_to_window=True)
        attn_metadata = self._build_golden_style_attn_metadata(
            position_ids=position_ids,
            context_lens=context_lens,
            q_lens=q_lens,
            cu_q_lens=cu_q_lens,
            current_seq_lens=current_seq_lens,
            start_pos=start_pos,
            seq_used_q=seq_used_q,
            shared_metadata=shared_metadata,
            win_slot_mapping=win_slot_mapping,
            full_kv_cache=full_kv_cache,
            is_prefill=True,
            batch_size=batch_size,
            decode_fast_path=False,
        )
        attn_metadata = self._build_cp_metadata(input_ids, attn_metadata)

        attn_inputs = None
        if os.getenv("MOJO_BUILD_LEGACY_ATTN_INPUTS", "0") == "1":
            attn_inputs = {}
            for layer_idx in range(pkv.num_layers):
                ratio = self.config.compress_ratios[layer_idx] if layer_idx < len(self.config.compress_ratios) else 0
                attn_inputs[layer_idx] = self._prepare_layer_prefill_inputs(
                    layer_idx, ratio, context_lens, q_lens, cu_q_lens,
                    start_pos, seq_used_q, batch_size, seq_len, device,
                    full_kv_cache,
                    shared_metadata.get(ratio, {}),
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "context_lens": context_lens,
            "cu_q_lens": cu_q_lens,
            "q_lens": q_lens,
            "attn_inputs": attn_inputs,
            "attn_metadata": attn_metadata,
        }

    def _prepare_layer_prefill_inputs(
        self, layer_idx, ratio, context_lens, q_lens, cu_q_lens,
        start_pos, seq_used_q, batch_size, seq_len, device,
        full_kv_cache, shared_metadata=None,
    ):
        pkv = self.paged_cache
        layer_inputs = {
            "q_lens": q_lens,
            "cu_q_lens": cu_q_lens,
            "start_pos": start_pos,
            "seq_used_q": seq_used_q,
            "sas_metadata": shared_metadata.get("sas_metadata") if shared_metadata else None,
            "full_kv_cache": full_kv_cache,
            "full_block_table": pkv.get_full_block_table(batch_size),
            "full_slot_mapping": pkv.get_full_slot_mapping(context_lens, q_lens),
            "full_kv_gather_indices": pkv.get_full_kv_gather_indices(context_lens, q_lens),
        }
        if ratio <= 1:
            return layer_inputs

        compressed_len = shared_metadata.get("compressed_len") if shared_metadata else None
        position_ids_cmp = shared_metadata.get("position_ids_cmp") if shared_metadata else None
        cmp_rope_position_ids = shared_metadata.get("cmp_rope_position_ids") if shared_metadata else None
        if compressed_len is None or position_ids_cmp is None:
            compressed_len, position_ids_cmp = pkv.get_compressed_position_ids(
                start_pos, seq_used_q, cu_q_lens, ratio,
            )
            cmp_rope_position_ids = (position_ids_cmp * ratio).to(dtype=torch.long).unsqueeze(0)

        cmp_slot_mapping = pkv.get_cmp_slot_mapping(
            layer_idx, start_pos, seq_used_q,
            cu_seqlens_q=cu_q_lens,
            compressed_len=compressed_len,
            position_ids_cmp=position_ids_cmp,
        )

        layer_inputs.update({
            "sfa_state_cache": pkv.get_sfa_kv_state(layer_idx),
            "state_block_table": pkv.get_cmp_state_block_table(layer_idx, start_pos, seq_used_q, True),
            "cmp_slot_mapping": cmp_slot_mapping,
            "cmp_kv_cache": pkv.get_sfa_cmp_kv(layer_idx),
            "cmp_block_tables": pkv.get_cmp_kv_block_table(layer_idx),
            "compressed_len": compressed_len,
            "cmp_rope_position_ids": cmp_rope_position_ids,
        })
        if ratio != 4:
            return layer_inputs

        c4a_cmp_kv_block_table = pkv.get_c4a_cmp_kv_block_table(layer_idx)
        li_metadata = shared_metadata.get("li_metadata") if shared_metadata else None
        if li_metadata is None:
            indexer_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
            li_metadata = self._compute_li_metadata(cu_q_lens[1:], indexer_seq_lens, c4a_cmp_kv_block_table)

        layer_inputs.update({
            "li_cmp_kv": pkv.get_li_cmp_kv(layer_idx),
            "li_key_dequant_scale": pkv.get_li_key_dequant_scale(layer_idx),
            "c4a_cmp_kv_block_table": c4a_cmp_kv_block_table,
            "li_state_cache": pkv.get_li_kv_state(layer_idx),
            "li_state_block_table": pkv.get_cmp_state_block_table(layer_idx, start_pos, seq_used_q, True),
            "li_cmp_slot_mapping": pkv.get_cmp_slot_mapping(
                layer_idx, start_pos, seq_used_q,
                cu_seqlens_q=cu_q_lens,
                compressed_len=compressed_len,
                position_ids_cmp=position_ids_cmp,
            ),
            "li_metadata": li_metadata,
        })
        return layer_inputs

    def prepare_decode_inputs(self, input_ids):
        pkv = self.paged_cache
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # Keep decode metadata semantics aligned with Golden. The specialized
        # fast path changes compressed position / slot mapping construction and
        # is more brittle when batch has already been sharded for DP-style runs.
        decode_fast_path = False

        context_lens_i32 = pkv.get_seq_length(0).to(device=device, dtype=torch.int32)
        context_lens = context_lens_i32.to(dtype=torch.long)
        if decode_fast_path:
            position_ids = context_lens.unsqueeze(1) + pkv._decode_position_offsets
            current_seq_lens = context_lens_i32 + 1
            q_lens = pkv._decode_q_lens
            cu_q_lens = pkv._decode_cu_q_lens
        else:
            position_offsets = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = context_lens.unsqueeze(1) + position_offsets
            current_seq_lens = context_lens_i32 + seq_len
            q_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
            cu_q_lens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=device,
            )

        start_pos = context_lens_i32
        seq_used_q = q_lens
        if decode_fast_path:
            win_slot_mapping = pkv.get_win_slot_mapping_decode(start_pos)
        else:
            win_slot_mapping = pkv.get_win_slot_mapping(context_lens, q_lens)

        unique_ratios = sorted(set(
            self.config.compress_ratios[l] if l < len(self.config.compress_ratios) else 0
            for l in range(pkv.num_layers)
        ))
        shared_metadata = {}
        for ratio in unique_ratios:
            attn_ratio = ratio if ratio > 1 else 1
            sas_meta = self._compute_sas_metadata(cu_q_lens, current_seq_lens, batch_size, attn_ratio)
            shared_metadata[ratio] = {"sas_metadata": sas_meta}
            if ratio > 1:
                if decode_fast_path:
                    compressed_len, position_ids_cmp = pkv.get_compressed_position_ids_decode(
                        start_pos, ratio,
                    )
                else:
                    compressed_len, position_ids_cmp = pkv.get_compressed_position_ids(
                        start_pos, seq_used_q, cu_q_lens, ratio,
                    )
                cmp_rope_position_ids = (position_ids_cmp * ratio).to(dtype=torch.long).unsqueeze(0)
                shared_metadata[ratio].update({
                    "compressed_len": compressed_len,
                    "position_ids_cmp": position_ids_cmp,
                    "cmp_rope_position_ids": cmp_rope_position_ids,
                })
            if ratio == 4:
                c4a_block_table = pkv.get_c4a_cmp_kv_block_table(0)
                li_meta = self._compute_li_metadata(cu_q_lens[1:], current_seq_lens, c4a_block_table)
                shared_metadata[ratio]["li_metadata"] = li_meta

        attn_metadata = self._build_golden_style_attn_metadata(
            position_ids=position_ids,
            context_lens=context_lens,
            q_lens=q_lens,
            cu_q_lens=cu_q_lens,
            current_seq_lens=current_seq_lens,
            start_pos=start_pos,
            seq_used_q=seq_used_q,
            shared_metadata=shared_metadata,
            win_slot_mapping=win_slot_mapping,
            full_kv_cache=None,
            is_prefill=False,
            batch_size=batch_size,
            decode_fast_path=decode_fast_path,
        )

        attn_inputs = None
        if os.getenv("MOJO_BUILD_LEGACY_ATTN_INPUTS", "0") == "1":
            attn_inputs = {}
            for layer_idx in range(pkv.num_layers):
                ratio = self.config.compress_ratios[layer_idx] if layer_idx < len(self.config.compress_ratios) else 0
                attn_inputs[layer_idx] = self._prepare_layer_decode_inputs(
                    layer_idx, ratio, context_lens, q_lens, cu_q_lens,
                    start_pos, seq_used_q, batch_size, seq_len, device,
                    shared_metadata.get(ratio, {}), win_slot_mapping,
                )

        return {
            "position_ids": position_ids,
            "context_lens": context_lens,
            "current_seq_lens": current_seq_lens,
            "cu_q_lens": cu_q_lens,
            "q_lens": q_lens,
            "start_pos": start_pos,
            "seq_used_q": seq_used_q,
            "attn_inputs": attn_inputs,
            "attn_metadata": attn_metadata,
        }

    def post_decode_step(self, seq_len=1):
        self.paged_cache.seq_lens += seq_len

    def _prepare_layer_decode_inputs(
        self, layer_idx, ratio, context_lens, q_lens, cu_q_lens,
        start_pos, seq_used_q, batch_size, seq_len, device,
        shared_metadata=None, win_slot_mapping=None,
    ):
        pkv = self.paged_cache
        win_kv_cache, win_block_table = pkv.get_win_kv_for_decode(layer_idx)

        layer_inputs = {
            "win_kv_cache": win_kv_cache,
            "win_block_table": win_block_table,
            "win_slot_mapping": win_slot_mapping,
            "q_lens": q_lens,
            "cu_q_lens": cu_q_lens,
            "start_pos": start_pos,
            "seq_used_q": seq_used_q,
            "sas_metadata": shared_metadata.get("sas_metadata") if shared_metadata else None,
        }
        if ratio <= 1:
            return layer_inputs

        compressed_len = shared_metadata.get("compressed_len") if shared_metadata else None
        position_ids_cmp = shared_metadata.get("position_ids_cmp") if shared_metadata else None
        cmp_rope_position_ids = shared_metadata.get("cmp_rope_position_ids") if shared_metadata else None
        if compressed_len is None or position_ids_cmp is None:
            compressed_len, position_ids_cmp = pkv.get_compressed_position_ids(
                start_pos, seq_used_q, cu_q_lens, ratio,
            )
            cmp_rope_position_ids = (position_ids_cmp * ratio).to(dtype=torch.long).unsqueeze(0)

        cmp_slot_mapping = pkv.get_cmp_slot_mapping(
            layer_idx, start_pos, seq_used_q,
            cu_seqlens_q=cu_q_lens,
            compressed_len=compressed_len,
            position_ids_cmp=position_ids_cmp,
        )

        layer_inputs.update({
            "sfa_state_cache": pkv.get_sfa_kv_state(layer_idx),
            "state_block_table": pkv.get_cmp_state_block_table(layer_idx, start_pos, seq_used_q, False),
            "cmp_slot_mapping": cmp_slot_mapping,
            "cmp_kv_cache": pkv.get_sfa_cmp_kv(layer_idx),
            "cmp_block_tables": pkv.get_cmp_kv_block_table(layer_idx),
            "compressed_len": compressed_len,
            "cmp_rope_position_ids": cmp_rope_position_ids,
        })
        if ratio != 4:
            return layer_inputs

        c4a_cmp_kv_block_table = pkv.get_c4a_cmp_kv_block_table(layer_idx)
        li_metadata = shared_metadata.get("li_metadata") if shared_metadata else None
        if li_metadata is None:
            li_metadata = self._compute_li_metadata(cu_q_lens[1:], context_lens + seq_len, c4a_cmp_kv_block_table)

        layer_inputs.update({
            "li_cmp_kv": pkv.get_li_cmp_kv(layer_idx),
            "li_key_dequant_scale": pkv.get_li_key_dequant_scale(layer_idx),
            "c4a_cmp_kv_block_table": c4a_cmp_kv_block_table,
            "li_state_cache": pkv.get_li_kv_state(layer_idx),
            "li_state_block_table": pkv.get_cmp_state_block_table(layer_idx, start_pos, seq_used_q, True),
            "li_cmp_slot_mapping": pkv.get_cmp_slot_mapping(
                layer_idx, start_pos, seq_used_q,
                cu_seqlens_q=cu_q_lens,
                compressed_len=compressed_len,
                position_ids_cmp=position_ids_cmp,
            ),
            "li_metadata": li_metadata,
        })
        return layer_inputs

    def _compute_li_metadata(self, actual_seq_q, actual_seq_k, block_table):
        config = self.config
        batch_size = actual_seq_q.shape[0]
        max_seqlen_q = int(actual_seq_q.max().item())
        max_seqlen_k = int(actual_seq_k.max().item())
        try:
            return torch.ops.custom.npu_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=actual_seq_q,
                actual_seq_lengths_key=actual_seq_k,
                num_heads_q=config.index_n_heads,
                num_heads_k=1,
                head_dim=config.index_head_dim,
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=config.index_topk,
                sparse_mode=3,
                pre_tokens=9223372036854775807,
                next_tokens=9223372036854775807,
                cmp_ratio=4,
                device=f"npu:{torch.npu.current_device()}",
            )
        except Exception:
            return None

    def _compute_sas_metadata(self, cu_q_lens, seq_lens, batch_size, compress_ratio):
        has_cmp_kv = compress_ratio > 1
        metadata_kwargs = {
            "cu_seqlens_q": cu_q_lens,
            "seqused_kv": seq_lens,
            "batch_size": batch_size,
            "cmp_ratio": compress_ratio,
            "ori_mask_mode": 4,
            "cmp_mask_mode": 3,
            "ori_win_left": self.config.sliding_window - 1,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
            "num_heads_q": self.config.num_attention_heads,
            "num_heads_kv": 1,
            "head_dim": self.config.head_dim,
            "has_ori_kv": True,
            "has_cmp_kv": has_cmp_kv,
        }
        if has_cmp_kv and compress_ratio == 4:
            metadata_kwargs["cmp_topk"] = self.config.index_topk
        try:
            from mojo_opset import MojoSparseAttnSharedkvMetadata
            return MojoSparseAttnSharedkvMetadata()(**metadata_kwargs)
        except Exception:
            return None
