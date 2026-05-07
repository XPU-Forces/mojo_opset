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
