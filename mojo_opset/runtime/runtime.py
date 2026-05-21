from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
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

    def __init__(self, paged_cache, config):
        self.paged_cache = paged_cache
        self.config = config

    @property
    def kv_cache(self):
        return self.paged_cache

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
        return cls(cache_data, config)

    def prepare_prefill_inputs(self, input_ids, attention_mask=None, q_lens=None):
        pkv = self.paged_cache
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if q_lens is None:
            if attention_mask is not None:
                q_lens = attention_mask.to(device=device, dtype=torch.int32).sum(dim=-1)
            else:
                q_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

        context_lens = pkv.get_seq_length(0).to(device=device, dtype=torch.long)
        if attention_mask is not None:
            position_ids = (attention_mask.to(device=device, dtype=torch.long).cumsum(dim=-1) - 1).clamp(min=0)
            position_ids = position_ids.masked_fill(~attention_mask.to(device=device, dtype=torch.bool), 1)
        else:
            past_len = int(context_lens.max().item()) if context_lens.numel() > 0 else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long).unsqueeze(0)

        cu_q_lens = torch.cat([
            torch.tensor([0], device=device, dtype=torch.int32),
            q_lens.cumsum(0, dtype=torch.int32),
        ])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "context_lens": context_lens,
            "cu_q_lens": cu_q_lens,
            "q_lens": q_lens,
        }

    def prepare_decode_inputs(self, input_ids):
        pkv = self.paged_cache
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        context_lens = pkv.get_seq_length(0).to(device=device, dtype=torch.long)
        position_offsets = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        position_ids = context_lens.to(dtype=torch.long).unsqueeze(1) + position_offsets
        current_seq_lens = context_lens.to(torch.int32) + seq_len
        q_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        cu_q_lens = torch.arange(
            0, (batch_size + 1) * seq_len, step=seq_len,
            dtype=torch.int32, device=device,
        )

        start_pos = context_lens.to(dtype=torch.int32)
        seq_used_q = q_lens
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

        decode_attn_inputs = {}
        for layer_idx in range(pkv.num_layers):
            ratio = self.config.compress_ratios[layer_idx] if layer_idx < len(self.config.compress_ratios) else 0
            decode_attn_inputs[layer_idx] = self._prepare_layer_decode_inputs(
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
            "decode_attn_inputs": decode_attn_inputs,
        }

    def post_decode_step(self, seq_len=1):
        self.paged_cache.seq_lens += seq_len

    def _prepare_layer_decode_inputs(
        self, layer_idx, ratio, context_lens, q_lens, cu_q_lens,
        start_pos, seq_used_q, batch_size, seq_len, device,
        shared_metadata=None, win_slot_mapping=None,
    ):
        pkv = self.paged_cache
        kv_cache, block_tables = pkv.get_kv_for_decode(layer_idx)
        win_kv_cache, win_block_table = pkv.get_win_kv_for_decode(layer_idx)
        kv_slot_mapping = self._compute_kv_slot_mapping(layer_idx, context_lens, seq_len)

        layer_inputs = {
            "kv_cache": kv_cache,
            "block_tables": block_tables,
            "win_kv_cache": win_kv_cache,
            "win_block_table": win_block_table,
            "win_slot_mapping": win_slot_mapping,
            "kv_slot_mapping": kv_slot_mapping,
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

    def _compute_kv_slot_mapping(self, layer_idx, context_lens, seq_len):
        pkv = self.paged_cache
        batch_size = context_lens.shape[0]
        device = context_lens.device
        positions = context_lens.to(torch.int32).unsqueeze(1) + torch.arange(
            seq_len, dtype=torch.int32, device=device,
        ).unsqueeze(0)
        block_idx = positions // pkv.block_size
        offset = positions % pkv.block_size
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).unsqueeze(1).expand(-1, seq_len)
        max_blocks = pkv.block_tables.shape[2]
        block_idx_clamped = block_idx.clamp(max=max_blocks - 1).to(torch.long)
        phys_blocks = pkv.block_tables[layer_idx, batch_indices, block_idx_clamped]
        slot_mapping = phys_blocks * pkv.block_size + offset
        valid_mask = block_idx < max_blocks
        slot_mapping = torch.where(valid_mask, slot_mapping, torch.full_like(slot_mapping, -1))
        return slot_mapping.reshape(-1).to(torch.int32)

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
