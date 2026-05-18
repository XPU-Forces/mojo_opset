import math
import json
import os
import logging
from typing import Optional, Tuple

import torch
import torch_npu
import custom_ops
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mojo_opset import MojoGemm
from mojo_opset import MojoQuantGemm
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoDynamicQuant
from mojo_opset import MojoMoEDispatch
from mojo_opset import MojoQuantExperts
from mojo_opset import MojoMoECombine


def _get_had_pow2(n: int, norm: bool = True, device: Optional[torch.device] = None) -> torch.Tensor:
    if not ((n & (n - 1) == 0) and (n > 0)):
        raise ValueError(f"n must be a positive power of 2, got {n}")
    had = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
    while had.shape[0] != n:
        had = torch.cat((torch.cat([had, had], 1), torch.cat([had, -had], 1)), 0)
        if norm:
            had /= math.sqrt(2)
    return had


def _rotate_activation(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    init_shape = x.shape
    x = x.to(torch.bfloat16).reshape(-1, matrix.shape[0])
    return x.matmul(matrix.to(device=x.device, dtype=torch.bfloat16)).reshape(init_shape).to(torch.bfloat16)


def _apply_partial_rotary(x, cos, sin, partial_slice):
    if isinstance(partial_slice, list):
        rope_dim = partial_slice[1] - partial_slice[0]
    else:
        rope_dim = partial_slice
    orig_shape = x.shape
    if x.dim() == 4:
        b, s, h, d = x.shape
        x_4d = x.reshape(b * s, h, 1, d)
    elif x.dim() == 3:
        b, s, d = x.shape
        x_4d = x.reshape(b * s, 1, 1, d)
    else:
        x_4d = x.unsqueeze(-3).unsqueeze(-3)
    if cos.dim() == 3:
        cos = cos.reshape(-1, 1, 1, rope_dim)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(-2).unsqueeze(-2)
    if sin.dim() == 3:
        sin = sin.reshape(-1, 1, 1, rope_dim)
    elif sin.dim() == 2:
        sin = sin.unsqueeze(-2).unsqueeze(-2)
    torch.ops.custom.inplace_partial_rotary_mul(
        x_4d, cos, sin,
        rotary_mode="interleave",
        partial_slice=partial_slice,
    )
    if x_4d.shape != orig_shape:
        if len(orig_shape) == 4:
            b, s, h, d = orig_shape
            x_4d = x_4d.reshape(b, s, h, d)
        elif len(orig_shape) == 3:
            b, s, d = orig_shape
            x_4d = x_4d.reshape(b, s, d)
        return x_4d
    return x


class DeepseekV4Config:

    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 129280)
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.intermediate_size = kwargs.get("intermediate_size", 18432)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 43)
        self.num_attention_heads = kwargs.get("num_attention_heads", 64)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 1)

        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", 2048)
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.n_routed_experts = kwargs.get("n_routed_experts", 256)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 6)
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.5)
        self.n_group = kwargs.get("n_group", 8)
        self.topk_group = kwargs.get("topk_group", 4)
        self.first_k_dense_replace = kwargs.get("first_k_dense_replace", 0)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.scoring_func = kwargs.get("scoring_func", "sqrtsoftplus")
        self.topk_method = kwargs.get("topk_method", "noaux_tc")

        self.head_dim = kwargs.get("head_dim", 512)
        self.q_lora_rank = kwargs.get("q_lora_rank", 1024)
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 64)
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.v_head_dim = self.head_dim

        self.o_lora_rank = kwargs.get("o_lora_rank", 1024)
        self.o_groups = kwargs.get("o_groups", 8)

        self.sliding_window = kwargs.get("sliding_window", 128)
        self.compress_ratios = kwargs.get("compress_ratios", [0, 0] + [4, 128] * 20 + [4, 0])
        self.next_n = kwargs.get("next_n", 1)
        self.pa_max_length = kwargs.get("pa_max_length", 2048)

        self.hc_mult = kwargs.get("hc_mult", 4)
        self.hc_sinkhorn_iters = kwargs.get("hc_sinkhorn_iters", 20)
        self.hc_eps = kwargs.get("hc_eps", 1e-6)

        self.index_n_heads = kwargs.get("index_n_heads", 64)
        self.index_head_dim = kwargs.get("index_head_dim", 128)
        self.index_topk = kwargs.get("index_topk", 512)

        self.num_hash_layers = kwargs.get("num_hash_layers", 3)

        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)

        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.hidden_act = kwargs.get("hidden_act", "silu")

        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.compress_rope_theta = kwargs.get("compress_rope_theta", 160000.0)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 1048576)
        self.rope_scaling = kwargs.get("rope_scaling", {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        })

        self.swiglu_limit = kwargs.get("swiglu_limit", 10.0)

    @classmethod
    def from_json(cls, json_path: str) -> "DeepseekV4Config":
        with open(json_path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def _from_hf_config(cls, hf_config) -> "DeepseekV4Config":
        kwargs = {}
        for attr in [
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "moe_intermediate_size",
            "n_shared_experts", "n_routed_experts", "num_experts_per_tok",
            "routed_scaling_factor", "n_group", "topk_group",
            "first_k_dense_replace", "norm_topk_prob", "scoring_func", "topk_method",
            "head_dim", "q_lora_rank", "qk_rope_head_dim", "o_lora_rank", "o_groups",
            "sliding_window", "compress_ratios", "next_n", "pa_max_length",
            "hc_mult", "hc_sinkhorn_iters", "hc_eps", "index_n_heads", "index_head_dim", "index_topk",
            "num_hash_layers", "attention_bias", "attention_dropout",
            "rms_norm_eps", "hidden_act", "rope_theta", "compress_rope_theta",
            "max_position_embeddings", "swiglu_limit",
        ]:
            if hasattr(hf_config, attr):
                kwargs[attr] = getattr(hf_config, attr)
        if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
            kwargs["rope_scaling"] = dict(hf_config.rope_scaling)
        return cls(**kwargs)


class PagedDummyCache:

    def __init__(self, config: DeepseekV4Config, batch_size: int, device: str,
                 block_size: int = 128, max_seq_len: int = 4096,
                 pa_max_length: Optional[int] = None, next_n: Optional[int] = None):
        self.num_layers = config.num_hidden_layers
        self.device = device
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.head_dim = config.head_dim
        self.config = config
        self.sliding_window = config.sliding_window
        self.index_head_dim = config.index_head_dim
        self.next_n = config.next_n if next_n is None else next_n
        self.pa_max_length = config.pa_max_length if pa_max_length is None else pa_max_length
        self.win_cache_size = self.sliding_window + self.next_n

        max_blocks_per_seq = (max_seq_len + self.block_size - 1) // self.block_size
        total_blocks = self.batch_size * max_blocks_per_seq * self.num_layers

        self.kv_cache = torch.zeros(
            (total_blocks, self.block_size, 1, self.head_dim),
            dtype=torch.bfloat16, device=self.device,
        )
        self.block_tables = torch.full(
            (self.num_layers, self.batch_size, max_blocks_per_seq),
            -1, dtype=torch.int32, device=self.device,
        )
        self.seq_lens = torch.zeros(
            (self.num_layers, self.batch_size), dtype=torch.int32, device=self.device,
        )
        self.free_blocks = torch.arange(total_blocks, device=self.device, dtype=torch.int32)
        self.num_free_blocks = total_blocks

        self.cache_data = {}
        for layer_idx in range(self.num_layers):
            ratio = config.compress_ratios[layer_idx] if layer_idx < len(config.compress_ratios) else 0
            cache_dict = {
                "win_kv": None,
                "sfa_cmp_kv": None,
                "sfa_kv_state": None,
                "li_cmp_kv": None,
                "li_kv_state": None,
                "li_key_dequant_scale": None,
                "c4a_cmp_kv_block_table": None,
                "c128a_cmp_kv_block_table": None,
            }
            win_block_num = self._get_block_num(self.win_cache_size)
            cache_dict["win_kv"] = self._create_cache(win_block_num, self.head_dim, torch.bfloat16)

            if ratio == 4:
                cmp_block_num = self._get_block_num(self.pa_max_length // ratio)
                overlap_num = 2
                state_block_num = self._get_block_num((1 + overlap_num) * ratio)
                cache_dict["sfa_cmp_kv"] = self._create_cache(cmp_block_num, self.head_dim, torch.bfloat16)
                cache_dict["sfa_kv_state"] = self._create_state_cache(state_block_num, ratio, self.head_dim)
                cache_dict["li_cmp_kv"] = self._create_cache(cmp_block_num, self.index_head_dim, torch.int8)
                cache_dict["li_kv_state"] = self._create_state_cache(state_block_num, ratio, self.index_head_dim)
                cache_dict["li_key_dequant_scale"] = self._create_cache(cmp_block_num, 1, torch.float16)
                cmp_block_num_per_batch = (cmp_block_num - 1) // self.batch_size
                cache_dict["c4a_cmp_kv_block_table"] = (
                    torch.arange(0, self.batch_size * cmp_block_num_per_batch, dtype=torch.int32, device=self.device)
                    .view(self.batch_size, -1) + 1
                )
            elif ratio == 128:
                cmp_block_num = self._get_block_num(self.pa_max_length // ratio)
                overlap_num = 1
                state_block_num = self._get_block_num(overlap_num * ratio)
                cache_dict["sfa_cmp_kv"] = self._create_cache(cmp_block_num, self.head_dim, torch.bfloat16)
                cache_dict["sfa_kv_state"] = self._create_state_cache(state_block_num, ratio, self.head_dim)
                cmp_block_num_per_batch = (cmp_block_num - 1) // self.batch_size
                cache_dict["c128a_cmp_kv_block_table"] = (
                    torch.arange(0, self.batch_size * cmp_block_num_per_batch, dtype=torch.int32, device=self.device)
                    .view(self.batch_size, -1) + 1
                )

            self.cache_data[layer_idx] = cache_dict

    def _get_block_num(self, cache_size):
        return math.ceil(cache_size / self.block_size) * self.batch_size + 1

    def _create_cache(self, block_num, dim, dtype):
        return torch.zeros(
            (block_num, self.block_size, 1, dim),
            dtype=dtype, device=self.device,
        )

    def _calc_full_block_table(self, cache_size: int, batch_size: int) -> torch.Tensor:
        block_num_per_batch = math.ceil(cache_size / self.block_size)
        return (
            torch.arange(0, batch_size * block_num_per_batch, dtype=torch.int32, device=self.device)
            .view(batch_size, -1)
            + 1
        )

    def _calc_ring_block_table(self, cache_size: int, batch_size: int) -> torch.Tensor:
        block_num_per_batch = math.ceil(cache_size / self.block_size)
        block_table_len = math.ceil(self.pa_max_length / self.block_size)
        block_table_offset = (
            torch.arange(0, batch_size * block_num_per_batch, dtype=torch.int32, device=self.device)
            .view(batch_size, -1)
            + 1
        )
        repeat_num = math.ceil(block_table_len / block_num_per_batch)
        return block_table_offset.repeat(1, repeat_num)[:, :block_table_len]

    def _calc_state_block_table(
        self,
        cache_size: int,
        start_pos: torch.Tensor,
        seq_used_q: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        batch_size = start_pos.shape[0]
        block_num_per_batch = math.ceil(cache_size / self.block_size)
        block_table_len = math.ceil(self.pa_max_length / self.block_size)
        block_table_offset = (
            torch.arange(0, batch_size * block_num_per_batch, dtype=torch.int32, device=self.device)
            .view(batch_size, -1)
            + 1
        )
        repeat_num = math.ceil(block_table_len / block_num_per_batch)
        block_table_offset = block_table_offset.repeat(1, repeat_num)[:, :block_table_len]
        block_pos_ids = torch.arange(block_table_len, dtype=torch.int32, device=self.device).repeat(batch_size, 1)
        actual_seq_len = start_pos + seq_used_q
        actual_block_start = (start_pos // self.block_size).view(batch_size, 1)
        actual_block_end = ((actual_seq_len - 1) // self.block_size).view(batch_size, 1)
        if is_prefill:
            return torch.where(block_pos_ids == actual_block_end, block_table_offset, torch.zeros_like(block_table_offset))
        block_table = torch.where(block_pos_ids >= actual_block_start, block_table_offset, torch.zeros_like(block_table_offset))
        return torch.where(block_pos_ids <= actual_block_end, block_table, torch.zeros_like(block_table))

    def get_cmp_state_block_table(
        self,
        layer_idx: int,
        start_pos: torch.Tensor,
        seq_used_q: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        ratio = self.config.compress_ratios[layer_idx]
        overlap = 1 if ratio == 4 else 0
        state_cache_size = (1 + overlap) * ratio
        return self._calc_state_block_table(state_cache_size, start_pos, seq_used_q, is_prefill)

    def get_compressed_position_ids(
        self,
        start_pos: torch.Tensor,
        seq_used_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        ratio: int,
        pad_value: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_pos = start_pos.to(dtype=torch.int32)
        seq_used_q = seq_used_q.to(dtype=torch.int32)
        cmp_start = start_pos // ratio
        cmp_end = (start_pos + seq_used_q) // ratio
        compressed_len = cmp_end - cmp_start
        offsets = F.pad(torch.cumsum(compressed_len, dim=0, dtype=torch.int32), (1, 0))[:-1]
        expanded_starts = torch.repeat_interleave(cmp_start, compressed_len)
        expanded_offsets = torch.repeat_interleave(offsets, compressed_len)
        flat_range = torch.arange(int(compressed_len.sum().item()), dtype=torch.int32, device=start_pos.device)
        compressed_ids = flat_range - expanded_offsets + expanded_starts

        total_q = int(cu_seqlens_q[-1].item())
        max_len = min(total_q, total_q // ratio + start_pos.shape[0])
        position_ids_cmp = torch.full((max_len,), pad_value, dtype=torch.int32, device=start_pos.device)
        valid_len = min(int(compressed_ids.numel()), max_len)
        if valid_len > 0:
            position_ids_cmp[:valid_len] = compressed_ids[:valid_len]
        return compressed_len, position_ids_cmp

    def get_cmp_slot_mapping(
        self,
        layer_idx: int,
        start_pos: torch.Tensor,
        seq_used_q: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        compressed_len: Optional[torch.Tensor] = None,
        position_ids_cmp: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        ratio = self.config.compress_ratios[layer_idx]
        block_table = self.cache_data[layer_idx].get(f"c{ratio}a_cmp_kv_block_table")
        if block_table is None:
            return None
        if compressed_len is None or position_ids_cmp is None:
            if cu_seqlens_q is None:
                total_q = int(seq_used_q.sum().item())
                cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device=start_pos.device)
            compressed_len, position_ids_cmp = self.get_compressed_position_ids(
                start_pos, seq_used_q, cu_seqlens_q, ratio
            )

        row_indices = torch.repeat_interleave(
            torch.arange(start_pos.shape[0], dtype=torch.int32, device=start_pos.device),
            compressed_len,
        )
        total_len = int(position_ids_cmp.shape[0])
        slot_mapping = torch.full((total_len,), -1, dtype=torch.int32, device=start_pos.device)
        if row_indices.numel() == 0:
            return slot_mapping

        valid_len = min(int(row_indices.numel()), total_len)
        row_indices = row_indices[:valid_len].to(torch.long)
        indices = position_ids_cmp[:valid_len]
        block_idx = (indices // self.block_size).to(torch.long)
        offset = indices % self.block_size
        slot_mapping[:valid_len] = block_table[row_indices, block_idx] * self.block_size + offset
        return slot_mapping

    def get_compressed_rope_position_ids(
        self,
        start_pos: torch.Tensor,
        seq_used_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        ratio: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        compressed_len, position_ids_cmp = self.get_compressed_position_ids(
            start_pos, seq_used_q, cu_seqlens_q, ratio, pad_value=1
        )
        return compressed_len, (position_ids_cmp * ratio).to(dtype=torch.long).unsqueeze(0)

    def get_win_slot_mapping(
        self,
        start_pos: torch.Tensor,
        seq_used_q: torch.Tensor,
        pad_to_window: bool = False,
    ) -> torch.Tensor:
        block_table = self._calc_ring_block_table(self.win_cache_size, start_pos.shape[0])
        slots = []
        for b in range(start_pos.shape[0]):
            if pad_to_window:
                seq_len = self.sliding_window
                base_pos = max(0, int(start_pos[b].item()) + int(seq_used_q[b].item()) - self.sliding_window)
            else:
                seq_len = int(seq_used_q[b].item())
                base_pos = int(start_pos[b].item())
            for t in range(seq_len):
                pos = base_pos + t
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                slots.append(block_table[b, block_idx] * self.block_size + offset)
        if not slots:
            return torch.empty((0,), dtype=torch.int32, device=self.device)
        return torch.stack(slots).to(dtype=torch.int32)

    def get_full_kv_gather_indices(
        self,
        start_pos: torch.Tensor,
        seq_used_q: torch.Tensor,
    ) -> torch.Tensor:
        total_len = start_pos.to(torch.int32) + seq_used_q.to(torch.int32)
        gather_start = torch.clamp(total_len - self.sliding_window, min=0)
        token_indices = torch.arange(self.sliding_window, dtype=torch.int32, device=self.device)
        return gather_start.unsqueeze(1) + token_indices.unsqueeze(0)

    def build_full_kv_for_prefill(
        self,
        kv: torch.Tensor,
        context_lens: torch.Tensor,
        cu_q_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = kv.shape[0]
        full_kv = self._create_cache(self._get_block_num(self.max_seq_len), self.head_dim, kv.dtype)
        block_table = self._calc_full_block_table(self.max_seq_len, batch_size)
        kv_flat = kv.reshape(-1, self.head_dim)
        slot_mapping = torch.full((kv_flat.shape[0],), -1, dtype=torch.int32, device=kv.device)
        for b in range(batch_size):
            context_len = int(context_lens[b].item())
            q_start = int(cu_q_lens[b].item())
            q_end = int(cu_q_lens[b + 1].item())
            for t in range(q_end - q_start):
                pos = context_len + t
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                if block_idx < block_table.shape[1]:
                    slot_mapping[q_start + t] = block_table[b, block_idx] * self.block_size + offset
        valid_mask = slot_mapping >= 0
        if valid_mask.any():
            torch.ops.custom.scatter_nd_update_asc(
                full_kv.view(-1, self.head_dim),
                slot_mapping[valid_mask].reshape(-1, 1),
                kv_flat[valid_mask],
            )
        return full_kv, block_table

    def _create_state_cache(self, state_block_num, compress_ratio, cache_dim):
        overlap_num = 2 if compress_ratio == 4 else 1
        return torch.zeros(
            (state_block_num, self.block_size, 2, overlap_num, cache_dim),
            dtype=torch.float32, device=self.device,
        )

    def _allocate_blocks(self, num_blocks: int) -> torch.Tensor:
        if num_blocks > self.num_free_blocks:
            raise ValueError(f"PagedDummyCache: Out of memory!")
        allocated = self.free_blocks[self.num_free_blocks - num_blocks: self.num_free_blocks]
        self.num_free_blocks -= num_blocks
        return allocated

    def update(self, kv: torch.Tensor, layer_idx: int, cu_q_lens: Optional[torch.Tensor] = None) -> None:
        batch_size = kv.shape[0]
        new_seq_len = kv.shape[1]

        if cu_q_lens is None:
            cu_q_lens = torch.arange(
                0, (batch_size + 1) * new_seq_len, step=new_seq_len,
                device=kv.device, dtype=torch.int32,
            )

        current_seq_lens = self.seq_lens[layer_idx]
        for i in range(batch_size):
            context_len = current_seq_lens[i].item()
            old_num_blocks = (context_len + self.block_size - 1) // self.block_size
            new_total_len = context_len + new_seq_len
            new_num_blocks = (new_total_len + self.block_size - 1) // self.block_size
            if new_num_blocks > old_num_blocks:
                num_to_allocate = new_num_blocks - old_num_blocks
                newly_allocated = self._allocate_blocks(num_to_allocate)
                self.block_tables[layer_idx, i, old_num_blocks:new_num_blocks] = newly_allocated

        kv_flat = kv.reshape(-1, self.head_dim)
        slot_mapping = torch.full((kv_flat.shape[0],), -1, dtype=torch.int32, device=kv.device)
        for b in range(batch_size):
            context_len = current_seq_lens[b].item()
            q_start = cu_q_lens[b].item()
            q_end = cu_q_lens[b + 1].item()
            q_len = q_end - q_start
            for t in range(q_len):
                pos = context_len + t
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                if block_idx < self.block_tables.shape[2]:
                    phys_block = self.block_tables[layer_idx, b, block_idx].item()
                    if phys_block >= 0:
                        slot_mapping[q_start + t] = phys_block * self.block_size + offset

        valid_mask = slot_mapping >= 0
        if valid_mask.any():
            cache_flat = self.kv_cache.view(-1, self.head_dim)
            torch.ops.custom.scatter_nd_update_asc(
                cache_flat, slot_mapping[valid_mask].reshape(-1, 1), kv_flat[valid_mask]
            )
        self.seq_lens[layer_idx] += new_seq_len

    def update_win_kv(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        slot_mapping: Optional[torch.Tensor] = None,
        gather_indices: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> None:
        win_cache = self.cache_data[layer_idx]["win_kv"]
        if win_cache is None:
            return
        if slot_mapping is None:
            raise ValueError("update_win_kv requires Golden-equivalent slot_mapping.")
        batch_size, seq_len, _ = kv.shape
        if gather_indices is not None:
            if start_pos is None:
                start_pos = torch.zeros(batch_size, dtype=torch.int32, device=kv.device)
            gathered_kv = []
            for b in range(batch_size):
                local_idx = gather_indices[b].to(kv.device) - start_pos[b].to(torch.int32)
                local_idx = torch.where(
                    (local_idx >= 0) & (local_idx < seq_len),
                    local_idx,
                    torch.zeros_like(local_idx),
                )
                gathered_kv.append(kv[b].index_select(0, local_idx.to(torch.long)))
            kv_flat = torch.stack(gathered_kv, dim=0).reshape(-1, self.head_dim)
        else:
            kv_flat = kv.reshape(-1, self.head_dim)
        win_flat = win_cache.view(-1, self.head_dim)
        torch.ops.custom.scatter_nd_update_asc(win_flat, slot_mapping.reshape(-1, 1), kv_flat)

    def update_sfa_cmp_kv(self, kv: torch.Tensor, layer_idx: int, slot_mapping: Optional[torch.Tensor] = None) -> None:
        sfa_cmp_cache = self.cache_data[layer_idx]["sfa_cmp_kv"]
        if sfa_cmp_cache is None:
            return
        if kv.shape[1] == 0:
            return
        batch_size, seq_len, _ = kv.shape
        kv_flat = kv.reshape(-1, self.head_dim)
        cmp_flat = sfa_cmp_cache.view(-1, self.head_dim)
        if slot_mapping is not None:
            torch.ops.custom.scatter_nd_update_asc(cmp_flat, slot_mapping.reshape(-1, 1), kv_flat)
        else:
            ratio = self.config.compress_ratios[layer_idx]
            cmp_context_len = int(self.seq_lens[layer_idx][0].item()) // ratio
            for t in range(seq_len):
                pos = cmp_context_len + t
                block_idx = pos // self.block_size + 1
                offset = pos % self.block_size
                if block_idx < sfa_cmp_cache.shape[0]:
                    sfa_cmp_cache[block_idx, offset, 0, :] = kv_flat[t]

    def update_li_cmp_kv(self, kv: torch.Tensor, layer_idx: int, slot_mapping: Optional[torch.Tensor] = None) -> None:
        li_cmp_cache = self.cache_data[layer_idx]["li_cmp_kv"]
        scale_cache = self.cache_data[layer_idx]["li_key_dequant_scale"]
        if li_cmp_cache is None:
            return
        if kv.shape[1] == 0:
            return
        batch_size, seq_len, _ = kv.shape
        kv_flat = kv.reshape(-1, self.index_head_dim).contiguous()
        kv_quant, k_scale = torch_npu.npu_dynamic_quant(kv_flat)
        k_scale = k_scale.to(torch.float16)
        cmp_flat = li_cmp_cache.view(-1, self.index_head_dim)
        scale_flat = scale_cache.view(-1, scale_cache.shape[-1])
        if slot_mapping is not None:
            torch.ops.custom.scatter_nd_update_asc(
                scale_flat,
                slot_mapping.reshape(-1, 1),
                k_scale.view(-1, scale_cache.shape[-1]),
            )
            torch.ops.custom.scatter_nd_update_asc(
                cmp_flat,
                slot_mapping.reshape(-1, 1),
                kv_quant.view(-1, li_cmp_cache.shape[-1]),
            )
        else:
            ratio = self.config.compress_ratios[layer_idx]
            cmp_context_len = int(self.seq_lens[layer_idx][0].item()) // ratio
            for t in range(seq_len):
                pos = cmp_context_len + t
                block_idx = pos // self.block_size + 1
                offset = pos % self.block_size
                if block_idx < li_cmp_cache.shape[0]:
                    li_cmp_cache[block_idx, offset, 0, :] = kv_quant[t]
                    scale_cache[block_idx, offset, 0, 0] = k_scale[t]

    def get_kv_for_decode(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_slen = self.seq_lens[layer_idx].max().item()
        max_blocks = (max_slen + self.block_size - 1) // self.block_size
        return self.kv_cache, self.block_tables[layer_idx, :, :max_blocks]

    def get_win_kv_for_decode(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        win_kv = self.cache_data[layer_idx]["win_kv"]
        block_table = self._calc_ring_block_table(self.win_cache_size, self.batch_size)
        return win_kv, block_table

    def get_win_kv(self, layer_idx: int):
        return self.cache_data[layer_idx]["win_kv"]

    def get_sfa_cmp_kv(self, layer_idx: int):
        return self.cache_data[layer_idx]["sfa_cmp_kv"]

    def get_sfa_kv_state(self, layer_idx: int):
        return self.cache_data[layer_idx]["sfa_kv_state"]

    def get_li_cmp_kv(self, layer_idx: int):
        return self.cache_data[layer_idx]["li_cmp_kv"]

    def get_li_kv_state(self, layer_idx: int):
        return self.cache_data[layer_idx]["li_kv_state"]

    def get_li_key_dequant_scale(self, layer_idx: int):
        return self.cache_data[layer_idx]["li_key_dequant_scale"]

    def get_c4a_cmp_kv_block_table(self, layer_idx: int):
        return self.cache_data[layer_idx]["c4a_cmp_kv_block_table"]

    def get_cmp_kv_block_table(self, layer_idx: int):
        ratio = self.config.compress_ratios[layer_idx]
        return self.cache_data[layer_idx].get(f"c{ratio}a_cmp_kv_block_table")

    def get_cmp_kv_for_decode(self, layer_idx: int):
        sfa_cmp = self.cache_data[layer_idx]["sfa_cmp_kv"]
        if sfa_cmp is not None:
            return sfa_cmp, None
        return None, None

    def update_cmp(self, kv: torch.Tensor, layer_idx: int) -> None:
        self.update_sfa_cmp_kv(kv, layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> torch.Tensor:
        return self.seq_lens[layer_idx].clone()


def _yarn_get_mscale(scale=1.0, mscale=1.0):
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV4RotaryEmbedding(nn.Module):

    def __init__(self, config: DeepseekV4Config, device: Optional[str] = None, base: Optional[float] = None):
        super().__init__()
        dim = config.qk_rope_head_dim
        base = config.rope_theta if base is None else base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        rope_scaling = config.rope_scaling
        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            scale = rope_scaling.get("factor", 1.0)
            self.attention_scaling = _yarn_get_mscale(scale, 1.0)
        else:
            self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = freqs.repeat_interleave(2, dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4Compressor(nn.Module):

    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: Optional[int] = None,
                 is_indexer: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim if head_dim is not None else config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap
        self.is_indexer = is_indexer
        self.debug_layer_idx = None
        if self.is_indexer:
            self.register_buffer("hadamard_matrix", _get_had_pow2(self.head_dim), persistent=False)

        self.wkv = MojoGemm(in_features=self.hidden_size, out_features=self.coff * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.wgate = MojoGemm(in_features=self.hidden_size, out_features=self.coff * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.norm = MojoRMSNorm(norm_size=self.head_dim, eps=config.rms_norm_eps)
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * self.head_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                state_cache: Optional[torch.Tensor] = None,
                state_block_table: Optional[torch.Tensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None,
                seq_used_q: Optional[torch.Tensor] = None,
                start_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if state_cache is None or state_block_table is None:
            raise ValueError("DeepseekV4Compressor requires state_cache and state_block_table.")
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=x.device,
            )
        if seq_used_q is None:
            seq_used_q = torch.full((batch_size,), seq_len, dtype=torch.int32, device=x.device)
        if start_pos is None:
            start_pos = torch.zeros(batch_size, dtype=torch.int32, device=x.device)

        x_flat = x.to(torch.bfloat16).reshape(-1, self.hidden_size).contiguous()
        cmp_flat = torch.ops.custom.compressor(
            x=x_flat,
            wkv=self.wkv.weight,
            wgate=self.wgate.weight,
            state_cache=state_cache.flatten(-3),
            ape=self.ape,
            norm_weight=self.norm.weight,
            rope_cos=cos.reshape(-1, self.rope_head_dim),
            rope_sin=sin.reshape(-1, self.rope_head_dim),
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=seq_used_q,
            start_pos=start_pos,
            coff=self.coff,
            norm_eps=self.norm.variance_epsilon,
            rotary_mode=2,
            cache_mode=1,
        )
        output_len = cos.reshape(-1, self.rope_head_dim).shape[0]
        cmp_flat = cmp_flat[:output_len]
        raw_cmp_flat = cmp_flat
        if self.is_indexer and cmp_flat.numel() > 0:
            cmp_flat = _rotate_activation(cmp_flat, self.hadamard_matrix)
        cmp_out = cmp_flat.view(1, output_len, self.head_dim)
        if os.getenv("DSV4_DEBUG_COMPRESSOR", "0") == "1":
            valid_len = int((((start_pos + seq_used_q) // self.compress_ratio) - (start_pos // self.compress_ratio)).sum().item())
            base = (
                f"[DSV4_COMPRESSOR_OUTPUT_DEBUG] side=mojo, layer={self.debug_layer_idx}, "
                f"is_indexer={self.is_indexer}, "
                f"compress_ratio={self.compress_ratio}, head_dim={self.head_dim}, "
                f"output: shape={tuple(cmp_out.shape)}, dtype={cmp_out.dtype}, device={cmp_out.device}"
            )
            if cmp_out.numel() > 0:
                detached = cmp_out.detach()
                base += f", min={detached.min().item()}, max={detached.max().item()}"
            else:
                base += ", empty=True"
            raw_valid_out = raw_cmp_flat[:valid_len]
            base += (
                f" | raw_valid_output: shape={tuple(raw_valid_out.shape)}, dtype={raw_valid_out.dtype}, "
                f"device={raw_valid_out.device}"
            )
            if raw_valid_out.numel() > 0:
                detached_raw_valid = raw_valid_out.detach()
                base += f", min={detached_raw_valid.min().item()}, max={detached_raw_valid.max().item()}"
            else:
                base += ", empty=True"
            valid_out = cmp_flat[:valid_len]
            base += (
                f" | valid_output: shape={tuple(valid_out.shape)}, dtype={valid_out.dtype}, "
                f"device={valid_out.device}"
            )
            if valid_out.numel() > 0:
                detached_valid = valid_out.detach()
                base += f", min={detached_valid.min().item()}, max={detached_valid.max().item()}"
            else:
                base += ", empty=True"
            print(base, flush=True)
        return cmp_out


class DeepseekV4Indexer(nn.Module):

    def __init__(self, config: DeepseekV4Config, compress_ratio: int = 4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.partial_slice = [self.head_dim - self.rope_head_dim, self.head_dim]
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = MojoQuantGemm(
            in_features=self.q_lora_rank,
            out_features=self.n_heads * self.head_dim,
            trans_weight=True,
        )
        self.weights_proj = MojoGemm(in_features=self.hidden_size, out_features=self.n_heads, bias=False)
        self.compressor = DeepseekV4Compressor(config, compress_ratio, head_dim=self.head_dim, is_indexer=True)
        self.compress_rotary_emb = DeepseekV4RotaryEmbedding(config, base=config.compress_rope_theta)
        self.register_buffer("hadamard_matrix", _get_had_pow2(self.head_dim), persistent=False)

    def forward(self, x, qr, cos, sin, past_key_values=None, layer_idx=0,
                cu_seqlens_q=None, seq_lens=None, start_pos: Optional[torch.Tensor] = None,
                state_block_table: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        weights = self.weights_proj(x.to(torch.bfloat16).reshape(-1, self.hidden_size))
        weights = weights.view(batch_size, seq_len, self.n_heads) * (self.softmax_scale * self.n_heads ** -0.5)

        li_state_cache = None
        if past_key_values is not None:
            li_state_cache = past_key_values.get_li_kv_state(layer_idx)
        if start_pos is None:
            start_pos = torch.zeros(batch_size, dtype=torch.int32, device=x.device)
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=x.device,
            )
        seq_used_q = torch.full((batch_size,), seq_len, dtype=torch.int32, device=x.device)
        if state_block_table is None and past_key_values is not None:
            state_block_table = past_key_values.get_cmp_state_block_table(layer_idx, start_pos, seq_used_q, True)

        compressed_len = None
        position_ids_cmp = None
        if past_key_values is not None:
            compressed_len, position_ids_cmp = past_key_values.get_compressed_rope_position_ids(
                start_pos, seq_used_q, cu_seqlens_q, self.compress_ratio
            )
            cmp_cos, cmp_sin = self.compress_rotary_emb(x, position_ids_cmp)
        else:
            cmp_cos = cos[:, ::self.compress_ratio, :]
            cmp_sin = sin[:, ::self.compress_ratio, :]
        self.compressor.debug_layer_idx = layer_idx
        li_kv = self.compressor(
            x, cmp_cos, cmp_sin,
            state_cache=li_state_cache,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens_q,
            seq_used_q=seq_used_q,
            start_pos=start_pos,
        )

        if past_key_values is not None:
            cmp_slot_mapping = past_key_values.get_cmp_slot_mapping(
                layer_idx,
                start_pos,
                seq_used_q,
                cu_seqlens_q=cu_seqlens_q,
                compressed_len=compressed_len,
                position_ids_cmp=(position_ids_cmp.squeeze(0) // self.compress_ratio).to(torch.int32),
            )
            past_key_values.update_li_cmp_kv(li_kv, layer_idx, cmp_slot_mapping)

        qr_flat = qr.reshape(-1, self.q_lora_rank).to(torch.bfloat16)
        qr_quant, qr_scale = _dynamic_quant_per_token(qr_flat)
        q = self.wq_b(qr_quant, qr_scale)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = _apply_partial_rotary(q, cos, sin, self.partial_slice)
        q = _rotate_activation(q, self.hadamard_matrix)

        if past_key_values is not None:
            li_cmp_kv = past_key_values.get_li_cmp_kv(layer_idx)
            li_key_dequant_scale = past_key_values.get_li_key_dequant_scale(layer_idx)
            c4a_block_table = past_key_values.get_c4a_cmp_kv_block_table(layer_idx)

            q_flat = q.flatten(0, 1)
            q_quant, q_scale = torch_npu.npu_dynamic_quant(q_flat)
            q_scale = q_scale.to(torch.float16)

            actual_seq_q = cu_seqlens_q[1:] if cu_seqlens_q is not None else torch.tensor([seq_len], dtype=torch.int32, device=x.device)
            actual_seq_k = seq_lens if seq_lens is not None else torch.tensor([seq_len], dtype=torch.int32, device=x.device)

            li_metadata = torch.ops.custom.npu_quant_lightning_indexer_metadata(
                layout_key='PA_BSND',
                sparse_count=self.index_topk,
                sparse_mode=3,
                layout_query="TND",
                cmp_ratio=self.compress_ratio,
                key_quant_mode=0,
                query_quant_mode=0,
                num_heads_q=self.n_heads,
                num_heads_k=1,
                head_dim=self.head_dim,
                actual_seq_lengths_query=actual_seq_q,
                actual_seq_lengths_key=actual_seq_k,
            )

            topk_idxs, _ = torch.ops.custom.npu_quant_lightning_indexer(
                query=q_quant, key=li_cmp_kv, weights=weights.flatten(0, 1).to(torch.float16),
                query_dequant_scale=q_scale,
                key_dequant_scale=li_key_dequant_scale.squeeze(-2),
                actual_seq_lengths_query=actual_seq_q,
                actual_seq_lengths_key=actual_seq_k,
                block_table=c4a_block_table, layout_key='PA_BSND',
                sparse_count=self.index_topk, sparse_mode=3,
                layout_query="TND", cmp_ratio=self.compress_ratio,
                key_quant_mode=0, query_quant_mode=0,
                metadata=li_metadata,
            )
            return topk_idxs.view(q_flat.shape[0], -1, self.index_topk)

        return None


def _dynamic_quant_per_token(x: torch.Tensor):
    return torch_npu.npu_dynamic_quant(x)


class DeepseekV4Attention(nn.Module):

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.head_dim = config.head_dim
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.sliding_window = config.sliding_window
        self.partial_slice = [self.head_dim - self.qk_rope_head_dim, self.head_dim]
        self.scaling = self.head_dim ** (-0.5)

        raw_ratio = config.compress_ratios[layer_idx] if layer_idx < len(config.compress_ratios) else 0
        self.compress_ratio = raw_ratio if raw_ratio > 1 else 1
        self._is_c1a = raw_ratio == 0

        self.wq_a = MojoGemm(in_features=config.hidden_size, out_features=config.q_lora_rank, bias=False)
        self.q_norm = MojoRMSNorm(norm_size=config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_a_quant = MojoDynamicQuant(input_size=config.q_lora_rank)
        nn.init.ones_(self.q_a_quant.inv_smooth_scale)
        self.wq_b = MojoQuantGemm(in_features=config.q_lora_rank, out_features=self.num_heads * self.head_dim, trans_weight=True)
        self.q_b_norm = MojoRMSNorm(eps=config.rms_norm_eps, norm_size=self.head_dim)

        self.wkv = MojoGemm(in_features=config.hidden_size, out_features=self.head_dim, bias=False, dtype=torch.bfloat16)
        self.kv_norm = MojoRMSNorm(eps=config.rms_norm_eps, norm_size=self.head_dim)

        self.wo_a = MojoGemm(in_features=self.num_heads * self.head_dim // self.o_groups, out_features=self.o_groups * self.o_lora_rank, bias=False)
        self.wo_b = MojoQuantGemm(in_features=self.o_groups * self.o_lora_rank, out_features=config.hidden_size, trans_weight=True)

        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))

        if raw_ratio > 1:
            self.sfa_compressor = DeepseekV4Compressor(config, raw_ratio, head_dim=self.head_dim)
            self.compress_rotary_emb = DeepseekV4RotaryEmbedding(config, base=config.compress_rope_theta)
            self.indexer = DeepseekV4Indexer(config, raw_ratio) if raw_ratio == 4 else None
        else:
            self.sfa_compressor = None
            self.compress_rotary_emb = None
            self.indexer = None

    def _debug_sparse_attn_inputs(
        self,
        q,
        kv_cache,
        block_tables,
        seq_lens,
        batch_size,
        seq_length,
        compress_ratio,
        cu_q_lens=None,
        cmp_kv_cache=None,
        cmp_block_tables=None,
        cmp_sparse_indices=None,
    ):
        if os.getenv("DSV4_DEBUG_SPARSE_ATTN", "0") != "1":
            return

        rank = int(os.getenv("RANK", os.getenv("RANK_ID", "0")))

        def tensor_stats(name, tensor):
            if tensor is None:
                return f"{name}=None"
            shape = tuple(tensor.shape)
            base = f"{name}: shape={shape}, dtype={tensor.dtype}, device={tensor.device}"
            if tensor.numel() == 0:
                return f"{base}, empty=True"
            detached = tensor.detach()
            min_val = detached.min().item()
            max_val = detached.max().item()
            return f"{base}, min={min_val}, max={max_val}"

        def index_check(name, indices, block_count=None, logical_limit=None):
            if indices is None:
                return f"{name}_check=None"
            if indices.numel() == 0:
                return f"{name}_check: empty=True"
            detached = indices.detach()
            min_val = int(detached.min().item())
            max_val = int(detached.max().item())
            checks = [f"min={min_val}", f"max={max_val}", f"has_negative={min_val < 0}"]
            if block_count is not None:
                checks.append(f"block_count={block_count}")
                checks.append(f"physical_oob={max_val >= block_count}")
            if logical_limit is not None:
                checks.append(f"logical_limit={logical_limit}")
                checks.append(f"logical_oob={max_val >= logical_limit}")
            return f"{name}_check: " + ", ".join(checks)

        def sparse_index_check(name, indices, actual_seq_k, ratio, bsz):
            if indices is None:
                return f"{name}_check=None"
            if indices.numel() == 0:
                return f"{name}_check: empty=True"
            detached = indices.detach()
            padding_count = int((detached == -1).sum().item())
            invalid_negative_count = int((detached < -1).sum().item())
            valid = detached[detached >= 0]
            checks = [
                f"padding_count={padding_count}",
                f"invalid_negative_count={invalid_negative_count}",
            ]
            if valid.numel() == 0:
                checks.append("valid_empty=True")
                return f"{name}_check: " + ", ".join(checks)

            valid_min = int(valid.min().item())
            valid_max = int(valid.max().item())
            checks.extend([f"valid_min={valid_min}", f"valid_max={valid_max}"])
            if actual_seq_k is not None and actual_seq_k.numel() > 0 and ratio > 1:
                actual_seq_k_max = int(actual_seq_k.detach().max().item())
                ceil_limit = (actual_seq_k_max + ratio - 1) // ratio
                golden_limit = actual_seq_k_max // ratio + bsz
                checks.extend([
                    f"actual_seq_k_max={actual_seq_k_max}",
                    f"ceil_logical_limit={ceil_limit}",
                    f"ceil_logical_oob={valid_max >= ceil_limit}",
                    f"golden_logical_limit={golden_limit}",
                    f"golden_logical_oob={valid_max >= golden_limit}",
                ])
            return f"{name}_check: " + ", ".join(checks)

        ori_block_count = kv_cache.shape[0] if kv_cache is not None and kv_cache.dim() > 0 else None
        cmp_block_count = cmp_kv_cache.shape[0] if cmp_kv_cache is not None and cmp_kv_cache.dim() > 0 else None

        debug_items = [
            f"[DSV4_SPARSE_ATTN_DEBUG] rank={rank}, layer={self.layer_idx}, batch_size={batch_size}, "
            f"seq_length={seq_length}, compress_ratio={compress_ratio}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, sliding_window={self.sliding_window}",
            tensor_stats("q", q),
            tensor_stats("kv_cache", kv_cache),
            tensor_stats("block_tables", block_tables),
            index_check("block_tables", block_tables, block_count=ori_block_count),
            tensor_stats("seq_lens", seq_lens),
            tensor_stats("cu_q_lens", cu_q_lens),
            tensor_stats("cmp_kv_cache", cmp_kv_cache),
            tensor_stats("cmp_block_tables", cmp_block_tables),
            index_check("cmp_block_tables", cmp_block_tables, block_count=cmp_block_count),
            tensor_stats("cmp_sparse_indices", cmp_sparse_indices),
            sparse_index_check("cmp_sparse_indices", cmp_sparse_indices, seq_lens, compress_ratio, batch_size),
        ]
        print(" | ".join(debug_items), flush=True)

    def _debug_compressor_inputs(
        self,
        *,
        x,
        kv,
        cos,
        sin,
        state_cache,
        state_block_table,
        cmp_slot_mapping,
        cmp_cache,
        cu_seqlens,
        seq_used_q,
        start_pos,
        is_prefill,
    ):
        if os.getenv("DSV4_DEBUG_COMPRESSOR", "0") != "1":
            return

        rank = int(os.getenv("RANK", os.getenv("RANK_ID", "0")))

        def tensor_stats(name, tensor):
            if tensor is None:
                return f"{name}=None"
            shape = tuple(tensor.shape)
            base = f"{name}: shape={shape}, dtype={tensor.dtype}, device={tensor.device}"
            if tensor.numel() == 0:
                return f"{base}, empty=True"
            detached = tensor.detach()
            return f"{base}, min={detached.min().item()}, max={detached.max().item()}"

        state_blocks = state_cache.shape[0] if state_cache is not None and state_cache.dim() > 0 else None
        cmp_blocks = cmp_cache.shape[0] if cmp_cache is not None and cmp_cache.dim() > 0 else None

        def index_check(name, indices, block_count):
            if indices is None:
                return f"{name}_check=None"
            if indices.numel() == 0:
                return f"{name}_check: empty=True"
            detached = indices.detach()
            min_val = int(detached.min().item())
            max_val = int(detached.max().item())
            checks = [f"min={min_val}", f"max={max_val}", f"has_negative={min_val < 0}"]
            if block_count is not None:
                checks.extend([f"block_count={block_count}", f"physical_oob={max_val >= block_count}"])
            return f"{name}_check: " + ", ".join(checks)

        debug_items = [
            f"[DSV4_COMPRESSOR_DEBUG] rank={rank}, layer={self.layer_idx}, is_prefill={is_prefill}, "
            f"compress_ratio={self.compress_ratio}, head_dim={self.head_dim}",
            tensor_stats("x", x),
            tensor_stats("kv", kv),
            tensor_stats("cos", cos),
            tensor_stats("sin", sin),
            tensor_stats("state_cache", state_cache),
            tensor_stats("state_block_table", state_block_table),
            index_check("state_block_table", state_block_table, state_blocks),
            tensor_stats("cmp_slot_mapping", cmp_slot_mapping),
            index_check("cmp_slot_mapping", cmp_slot_mapping, cmp_blocks),
            tensor_stats("cmp_cache", cmp_cache),
            tensor_stats("cu_seqlens", cu_seqlens),
            tensor_stats("seq_used_q", seq_used_q),
            tensor_stats("start_pos", start_pos),
        ]
        print(" | ".join(debug_items), flush=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, seq_length = hidden_states.shape[:2]

        context_lens = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
        )

        h_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.bfloat16)

        qa = self.wq_a(h_flat)
        qa = qa.view(batch_size, seq_length, -1)
        qa = self.q_norm(qa)
        qa_flat = qa.reshape(-1, qa.shape[-1]).to(torch.bfloat16)
        qa_quant, qa_scale = self.q_a_quant(qa_flat)
        q = self.wq_b(qa_quant, qa_scale)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        q = self.q_b_norm(q)

        kv = self.wkv(h_flat)
        kv = self.kv_norm(kv)
        kv = kv.view(batch_size, seq_length, self.head_dim)

        cos, sin = position_embeddings
        q = _apply_partial_rotary(q, cos, sin, self.partial_slice)
        kv = _apply_partial_rotary(
            kv.view(batch_size, seq_length, 1, self.head_dim), cos, sin, self.partial_slice
        ).view(batch_size, seq_length, self.head_dim)

        if past_key_values is None:
            raise ValueError("Paged Attention requires a PagedDummyCache instance.")

        cmp_sparse_indices = None
        if self.sfa_compressor is not None:
            sfa_state_cache = past_key_values.get_sfa_kv_state(self.layer_idx)
            start_pos = context_lens.to(dtype=torch.int32)
            seq_used_q = torch.full((batch_size,), seq_length, dtype=torch.int32, device=hidden_states.device)
            cu_seqlens_q = torch.arange(
                0, (batch_size + 1) * seq_length, step=seq_length,
                dtype=torch.int32, device=hidden_states.device,
            )
            compressed_len, cmp_rope_position_ids = past_key_values.get_compressed_rope_position_ids(
                start_pos, seq_used_q, cu_seqlens_q, self.compress_ratio
            )
            cmp_cos, cmp_sin = self.compress_rotary_emb(hidden_states, cmp_rope_position_ids)
            state_block_table = past_key_values.get_cmp_state_block_table(
                self.layer_idx, start_pos, seq_used_q, is_prefill
            )
            cmp_slot_mapping = past_key_values.get_cmp_slot_mapping(
                self.layer_idx,
                start_pos,
                seq_used_q,
                cu_seqlens_q=cu_seqlens_q,
                compressed_len=compressed_len,
                position_ids_cmp=(cmp_rope_position_ids.squeeze(0) // self.compress_ratio).to(torch.int32),
            )
            self._debug_compressor_inputs(
                x=hidden_states,
                kv=kv,
                cos=cmp_cos,
                sin=cmp_sin,
                state_cache=sfa_state_cache,
                state_block_table=state_block_table,
                cmp_slot_mapping=cmp_slot_mapping,
                cmp_cache=past_key_values.get_sfa_cmp_kv(self.layer_idx),
                cu_seqlens=cu_seqlens_q,
                seq_used_q=seq_used_q,
                start_pos=start_pos,
                is_prefill=is_prefill,
            )
            self.sfa_compressor.debug_layer_idx = self.layer_idx
            cmp_kv = self.sfa_compressor(
                hidden_states, cmp_cos, cmp_sin,
                state_cache=sfa_state_cache,
                state_block_table=state_block_table,
                cu_seqlens=cu_seqlens_q,
                seq_used_q=seq_used_q,
                start_pos=start_pos,
            )
            past_key_values.update_sfa_cmp_kv(cmp_kv, self.layer_idx, cmp_slot_mapping)

            if self.indexer is not None:
                current_seq_lens = context_lens.to(dtype=torch.int32) + seq_length
                cmp_sparse_indices = self.indexer.forward(
                    hidden_states, qa, cos, sin,
                    past_key_values=past_key_values, layer_idx=self.layer_idx,
                    cu_seqlens_q=cu_seqlens_q,
                    seq_lens=current_seq_lens,
                    start_pos=start_pos,
                    state_block_table=state_block_table,
                )

        if self._is_c1a:
            o = self._c1a_attention(q, kv, past_key_values, context_lens, is_prefill)
        else:
            o = self._sparse_attention(q, kv, past_key_values, context_lens, cmp_sparse_indices, is_prefill)

        o = self._attn_post(o, position_embeddings)
        return o, None

    def _run_attn(self, q, kv_cache, block_tables, seq_lens, batch_size, seq_length,
                  compress_ratio, cu_q_lens=None, cmp_kv_cache=None, cmp_block_tables=None,
                  cmp_sparse_indices=None):
        has_cmp_kv = compress_ratio > 1
        metadata_kwargs = {
            "cu_seqlens_q": cu_q_lens,
            "seqused_kv": seq_lens,
            "batch_size": batch_size,
            "cmp_ratio": compress_ratio,
            "ori_mask_mode": 4,
            "cmp_mask_mode": 3,
            "ori_win_left": self.sliding_window - 1,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
            "num_heads_q": self.num_heads,
            "num_heads_kv": 1,
            "head_dim": self.head_dim,
            "has_ori_kv": True,
            "has_cmp_kv": has_cmp_kv,
        }
        if has_cmp_kv and compress_ratio == 4:
            metadata_kwargs["cmp_topk"] = self.config.index_topk

        metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**metadata_kwargs)

        self._debug_sparse_attn_inputs(
            q, kv_cache, block_tables, seq_lens, batch_size, seq_length,
            compress_ratio, cu_q_lens, cmp_kv_cache, cmp_block_tables, cmp_sparse_indices,
        )

        o = torch.ops.custom.npu_sparse_attn_sharedkv(
            q=q, ori_kv=kv_cache,
            cmp_kv=cmp_kv_cache if has_cmp_kv else None,
            cmp_sparse_indices=cmp_sparse_indices if has_cmp_kv else None,
            cu_seqlens_q=cu_q_lens, seqused_kv=seq_lens,
            cmp_block_table=cmp_block_tables if has_cmp_kv and cmp_block_tables is not None else None,
            ori_block_table=block_tables, cmp_ratio=compress_ratio,
            ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=self.sliding_window - 1, ori_win_right=0,
            layout_q="TND", layout_kv="PA_ND",
            sinks=self.attn_sink, metadata=metadata,
            softmax_scale=self.scaling,
        )[0]
        return o.view(batch_size, seq_length, self.num_heads, self.head_dim)

    def _c1a_attention(self, q, kv, past_key_values, context_lens, is_prefill: bool):
        batch_size, seq_length = q.shape[:2]
        q_tnd = q.contiguous().view(-1, self.num_heads, self.head_dim)
        current_seq_lens = context_lens + seq_length
        q_lens = torch.full((batch_size,), seq_length, dtype=torch.int32, device=q.device)
        cu_q_lens = torch.cat([torch.tensor([0], device=q.device, dtype=torch.int32), q_lens.cumsum(0, dtype=torch.int32)])
        if is_prefill:
            kv_cache, block_tables = past_key_values.build_full_kv_for_prefill(kv, context_lens, cu_q_lens)
            win_slot_mapping = past_key_values.get_win_slot_mapping(context_lens, q_lens, pad_to_window=True)
            full_kv_gather_indices = past_key_values.get_full_kv_gather_indices(context_lens, q_lens)
            past_key_values.update_win_kv(kv, self.layer_idx, win_slot_mapping, full_kv_gather_indices, context_lens)
        else:
            win_slot_mapping = past_key_values.get_win_slot_mapping(context_lens, q_lens)
            past_key_values.update_win_kv(kv, self.layer_idx, win_slot_mapping)
            past_key_values.update(kv, self.layer_idx)
            kv_cache, block_tables = past_key_values.get_win_kv_for_decode(self.layer_idx)
        out = self._run_attn(q_tnd, kv_cache, block_tables, current_seq_lens, batch_size, seq_length, 1, cu_q_lens)
        if is_prefill:
            past_key_values.update(kv, self.layer_idx, cu_q_lens)
        return out

    def _sparse_attention(self, q, kv, past_key_values, context_lens, cmp_sparse_indices=None, is_prefill: bool = True):
        batch_size, seq_length = q.shape[:2]
        q_tnd = q.contiguous().view(-1, self.num_heads, self.head_dim)
        current_seq_lens = context_lens + seq_length
        q_lens = torch.full((batch_size,), seq_length, dtype=torch.int32, device=q.device)
        cu_q_lens = torch.cat([torch.tensor([0], device=q.device, dtype=torch.int32), q_lens.cumsum(0, dtype=torch.int32)])
        if is_prefill:
            kv_cache, block_tables = past_key_values.build_full_kv_for_prefill(kv, context_lens, cu_q_lens)
            win_slot_mapping = past_key_values.get_win_slot_mapping(context_lens, q_lens, pad_to_window=True)
            full_kv_gather_indices = past_key_values.get_full_kv_gather_indices(context_lens, q_lens)
            past_key_values.update_win_kv(kv, self.layer_idx, win_slot_mapping, full_kv_gather_indices, context_lens)
        else:
            win_slot_mapping = past_key_values.get_win_slot_mapping(context_lens, q_lens)
            past_key_values.update_win_kv(kv, self.layer_idx, win_slot_mapping)
            past_key_values.update(kv, self.layer_idx)
            kv_cache, block_tables = past_key_values.get_win_kv_for_decode(self.layer_idx)
        cmp_kv_cache = past_key_values.get_sfa_cmp_kv(self.layer_idx)
        cmp_block_tables = past_key_values.get_cmp_kv_block_table(self.layer_idx) if self.compress_ratio > 1 else None
        out = self._run_attn(q_tnd, kv_cache, block_tables, current_seq_lens, batch_size, seq_length,
                             self.compress_ratio, cu_q_lens, cmp_kv_cache, cmp_block_tables, cmp_sparse_indices)
        if is_prefill:
            past_key_values.update(kv, self.layer_idx, cu_q_lens)
        return out

    def _attn_post(self, o, position_embeddings):
        batch_size, seq_length = o.shape[:2]
        cos = position_embeddings[0]
        sin = -position_embeddings[1]

        o = _apply_partial_rotary(o, cos, sin, self.partial_slice)

        o = o.reshape(batch_size * seq_length, self.o_groups, -1).to(torch.bfloat16)
        wo_a_weight = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        o_t = o.transpose(0, 1).float()
        wo_a_weight_t = wo_a_weight.transpose(1, 2).float()
        wo_a_out = torch.bmm(o_t, wo_a_weight_t)
        wo_a_out = wo_a_out.transpose(0, 1).reshape(batch_size * seq_length, -1).to(torch.bfloat16)
        wo_a_quant, wo_a_scale = _dynamic_quant_per_token(wo_a_out)
        wo_b_out = self.wo_b(wo_a_quant, wo_a_scale)
        return wo_b_out.view(batch_size, seq_length, -1)


class DeepseekV4SharedExpert(nn.Module):

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.swiglu_limit = config.swiglu_limit

        self.gate_proj = MojoQuantGemm(in_features=self.hidden_size, out_features=self.intermediate_size, trans_weight=True)
        self.up_proj = MojoQuantGemm(in_features=self.hidden_size, out_features=self.intermediate_size, trans_weight=True)
        self.down_proj = MojoQuantGemm(in_features=self.intermediate_size, out_features=self.hidden_size, trans_weight=True)
        self.gate_quant = MojoDynamicQuant(input_size=self.hidden_size)
        self.up_quant = MojoDynamicQuant(input_size=self.hidden_size)
        self.intermediate_quant = MojoDynamicQuant(input_size=self.intermediate_size)
        nn.init.ones_(self.gate_quant.inv_smooth_scale)
        nn.init.ones_(self.up_quant.inv_smooth_scale)
        nn.init.ones_(self.intermediate_quant.inv_smooth_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size).to(torch.bfloat16)
        x_quant, activation_scale = self.gate_quant(x_flat)

        gate_up_weight = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0).transpose(0, 1).contiguous()
        gate_up_weight_scale = torch.cat(
            (self.gate_proj.weight_scale, self.up_proj.weight_scale), dim=0
        ).contiguous()
        merged_x = torch_npu.npu_quant_matmul(
            x_quant,
            gate_up_weight,
            gate_up_weight_scale.view(-1).float(),
            pertoken_scale=None,
            bias=None,
            output_dtype=torch.int32,
        )

        swiglu_limit_args = {}
        if self.swiglu_limit is not None and self.swiglu_limit > 0:
            swiglu_limit_args.update(
                {
                    "swiglu_mode": 1,
                    "clamp_limit": self.swiglu_limit,
                    "glu_alpha": 1,
                    "glu_bias": 0,
                }
            )
        intermediate_quant, intermediate_scale = torch_npu.npu_dequant_swiglu_clamp_quant(
            merged_x,
            weight_scale=gate_up_weight_scale.view(-1).float(),
            quant_scale=self.intermediate_quant.inv_smooth_scale.to(dtype=torch.float32),
            quant_mode=1,
            activate_left=True,
            activation_scale=activation_scale.view(-1),
            **swiglu_limit_args,
        )
        intermediate_scale = intermediate_scale.unsqueeze(-1)
        return self.down_proj(intermediate_quant, intermediate_scale).view(*orig_shape)


class DeepseekV4MoE(nn.Module):

    def __init__(self, config: DeepseekV4Config, layer_idx: int, ep_size: int = 1, ep_rank: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.is_hash = layer_idx < config.num_hash_layers

        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_per_rank = self.n_routed_experts // self.ep_size
        self.ep_start = self.ep_rank * self.experts_per_rank
        self.ep_end = self.ep_start + self.experts_per_rank
        self.ep_group = None
        self.hccl_comm_dict = {}
        self.dispatch_kwargs = None
        self.combine_kwargs = None
        self.gmm_quant_mode = "w8a8int8"
        self.dispatch_quant_mode = {
            "w16a16": 0,
            "w8a8int8": 2,
        }

        self.dispatch = MojoMoEDispatch(num_experts=config.n_routed_experts)
        self.experts = MojoQuantExperts(
            num_experts=self.experts_per_rank,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            weight_dtype=torch.int8,
        )
        self.combine = MojoMoECombine(multiply_by_gates=True)
        nn.init.ones_(self.experts.up_proj_quantize.inv_smooth_scale)
        nn.init.ones_(self.experts.down_proj_quantize.inv_smooth_scale)

        self.register_buffer(
            "smooth_scale_1",
            torch.ones(self.experts_per_rank, config.hidden_size, dtype=torch.float32),
        )
        self.register_buffer(
            "smooth_scale_2",
            torch.ones(self.experts_per_rank, config.moe_intermediate_size, dtype=torch.float32),
        )

        self.shared_experts = DeepseekV4SharedExpert(config)

        self.gate = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size, dtype=torch.float32))

        if self.is_hash:
            self.e_score_correction_bias = None
            self.tid2eid = nn.Parameter(
                torch.randint(high=config.n_routed_experts, size=(config.vocab_size, self.top_k), dtype=torch.int32),
                requires_grad=False,
            )
        else:
            self.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
            self.tid2eid = None

    def _gate_topk(self, logits, input_ids=None):
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sqrtsoftplus":
            scores = F.softplus(logits).sqrt()
        else:
            raise NotImplementedError(f"Unsupported scoring_func: {self.scoring_func}")

        if self.topk_method == "noaux_tc":
            scoring_func_mapping = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}
            topk_weight, topk_idx, _ = torch.ops.custom.npu_moe_gating_top_k(
                logits, k=self.top_k, bias=self.e_score_correction_bias,
                input_ids=input_ids if self.is_hash else None,
                tid2eid=self.tid2eid, k_group=1, group_count=1,
                group_select_mode=1, renorm=0,
                norm_type=scoring_func_mapping[self.scoring_func],
                routed_scaling_factor=self.routed_scaling_factor,
                eps=float(1e-20), out_flag=False,
            )
            return topk_idx.to(torch.int32), topk_weight
        elif self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        else:
            raise NotImplementedError(f"Unsupported topk_method: {self.topk_method}")

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True)
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx.to(torch.int32), topk_weight

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                is_prefill: bool = True) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size).to(torch.bfloat16)

        logits = F.linear(hidden_states_flat.float(), self.gate)
        topk_idx, topk_weight = self._gate_topk(logits, input_ids)

        shared_out = self.shared_experts(residuals)
        shared_out_flat = shared_out.view(-1, self.hidden_size).to(torch.bfloat16)

        if self.ep_size > 1 and self.ep_group is not None:
            if is_prefill:
                routed_out = self._moe_infer_ep(
                    hidden_states_flat, topk_idx, topk_weight, shared_expert_out=shared_out_flat)
                return routed_out.view(*orig_shape)
            else:
                routed_out = self._moe_infer_ep_decode(
                    hidden_states_flat, topk_idx, topk_weight, shared_expert_out=shared_out_flat)
                return routed_out.view(*orig_shape)

        sorted_hidden, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states_flat, topk_weight, topk_idx
        )
        expert_outputs = self.experts(sorted_hidden, tokens_per_expert)
        output_buffer = torch.zeros_like(hidden_states_flat, memory_format=torch.contiguous_format)
        routed_out = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        return routed_out.view(*orig_shape) + shared_out

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        moe_ep_group_name = self.hccl_comm_dict.get("moe_ep_group_mc2_name", None)
        quant_mode = self.dispatch_quant_mode.get(self.gmm_quant_mode, self.dispatch_quant_mode["w16a16"])
        enable_smooth_scale = quant_mode == self.dispatch_quant_mode["w8a8int8"]
        self.dispatch_kwargs = {
            "x_active_mask": None,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.n_routed_experts,
            "global_bs": 0,
            "scales": self.smooth_scale_1 if enable_smooth_scale else None,
            "quant_mode": quant_mode,
            "group_ep": moe_ep_group_name,
            "ep_world_size": self.ep_size,
            "ep_rank_id": global_rank,
            "group_tp": moe_ep_group_name,
            "tp_world_size": 1,
            "tp_rank_id": 0,
        }
        self.combine_kwargs = {
            "x_active_mask": None,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.n_routed_experts,
            "global_bs": 0,
            "group_ep": moe_ep_group_name,
            "ep_world_size": self.ep_size,
            "ep_rank_id": global_rank,
            "group_tp": moe_ep_group_name,
            "tp_world_size": 1,
            "tp_rank_id": 0,
        }

    def forward_expert_gmm(self, x, expert_tokens, pertoken_scale=None, group_list_type=1):
        experts_mod = self.experts
        hidden_size = x.size(-1)

        if pertoken_scale is not None and pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            x = x.view(-1, hidden_size)

        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        fc1_out = torch_npu.npu_grouped_matmul(
            [x], [experts_mod.up_proj_weight],
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.int32,
            group_type=0,
            group_list_type=group_list_type,
            tuning_config=[0],
        )[0]

        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            fc1_out,
            weight_scale=experts_mod.up_proj_weight_scale,
            quant_scale=self.smooth_scale_2,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1,
            activation_scale=pertoken_scale,
        )

        fc2_out = torch_npu.npu_grouped_matmul(
            [intermediate_h], [experts_mod.down_proj_weight],
            scale=[experts_mod.down_proj_weight_scale],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=group_list_type,
            tuning_config=[0],
        )[0]

        return fc2_out

    def process_expert_weights(self):
        if self.ep_size <= 1:
            return
        experts_mod = self.experts
        experts_mod.up_proj_weight.data = experts_mod.up_proj_weight.data.transpose(1, 2).contiguous()
        experts_mod.down_proj_weight.data = experts_mod.down_proj_weight.data.transpose(1, 2).contiguous()
        torch_npu.npu.config.allow_internal_format = True
        experts_mod.up_proj_weight.data = torch_npu.npu_format_cast(experts_mod.up_proj_weight.data.contiguous(), 29)
        experts_mod.down_proj_weight.data = torch_npu.npu_format_cast(experts_mod.down_proj_weight.data.contiguous(), 29)
        experts_mod.up_proj_weight_scale.data = experts_mod.up_proj_weight_scale.data.to(torch.float)
        self.smooth_scale_1.data = self.smooth_scale_1.data.to(torch.float)
        self.smooth_scale_2.data = self.smooth_scale_2.data.to(torch.float)

    def _moe_infer_ep(self, hidden_states_flat, topk_idx, topk_weight, shared_expert_out=None):
        moe_ep_group = self.ep_group
        n_tokens = hidden_states_flat.shape[0]

        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = \
            torch_npu.npu_moe_init_routing_v2(
                hidden_states_flat,
                expert_idx=topk_idx,
                active_num=n_tokens * self.top_k,
                expert_num=self.n_routed_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.n_routed_experts],
                quant_mode=1,
                scale=self.smooth_scale_1,
            )

        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)

        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        combine_tokens = combine_tokens.view(2, self.ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]

        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)

        gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
        dist.all_to_all_single(gathered_pertoken_scale, pertoken_scale, output_splits, input_splits, group=moe_ep_group)

        hidden_states_ordered, gathered_pertoken_scale, gathered_ids_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(
                gathered_tokens,
                tokens_per_expert_group.view(self.ep_size, -1),
                per_token_scales=gathered_pertoken_scale,
            )

        expert_out = self.forward_expert_gmm(
            hidden_states_ordered, tokens_per_local_expert,
            pertoken_scale=gathered_pertoken_scale,
            group_list_type=1,
        )

        new_x = torch.index_select(expert_out, 0, gathered_ids_unsort.float().argsort().int())

        combined_tokens = new_x.new_empty(expanded_x.shape[0], new_x.shape[1])
        dist.all_to_all_single(combined_tokens, new_x, input_splits, output_splits, group=moe_ep_group)

        routed_out = torch_npu.npu_moe_finalize_routing(
            combined_tokens,
            skip1=shared_expert_out,
            skip2=None,
            bias=None,
            scales=topk_weight.to(combined_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )
        return routed_out

    def _moe_infer_ep_decode(self, hidden_states_flat, topk_idx, topk_weight, shared_expert_out=None):
        if self.dispatch_kwargs is None:
            self.set_mc2_kwargs()

        dispatch_output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states_flat,
            expert_ids=topk_idx,
            **self.dispatch_kwargs,
        )
        expand_x, dynamic_scale, expand_idx, expert_token_num = dispatch_output[:4]
        ep_recv_counts = dispatch_output[4] if len(dispatch_output) > 4 else None
        tp_recv_counts = dispatch_output[5] if len(dispatch_output) > 5 else None

        expert_out = self.forward_expert_gmm(expand_x, expert_token_num, pertoken_scale=dynamic_scale, group_list_type=1)

        combine_input = {
            "expand_x": expert_out,
            "shared_expert_x": shared_expert_out,
            "expert_ids": topk_idx,
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weight.to(torch.float32),
        }
        if ep_recv_counts is not None:
            combine_input["ep_send_counts"] = ep_recv_counts
        if tp_recv_counts is not None:
            combine_input["tp_send_counts"] = tp_recv_counts

        routed_out = torch_npu.npu_moe_distribute_combine_v2(
            **combine_input,
            **self.combine_kwargs,
        )
        return routed_out


class OpKernel:

    @staticmethod
    def hc_pre(hidden_states, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, norm_eps, hc_eps):
        y, post, comb = torch.ops.custom.npu_hc_pre(
            hidden_states,
            hc_fn.float() if hc_fn.dtype != torch.float32 else hc_fn,
            hc_scale.float() if hc_scale.dtype != torch.float32 else hc_scale,
            hc_base.float() if hc_base.dtype != torch.float32 else hc_base,
            hc_mult=hc_mult, hc_sinkhorn_iters=sinkhorn_iters,
            norm_eps=norm_eps, hc_eps=hc_eps,
        )
        return y, post, comb

    @staticmethod
    def hc_post(hidden_states, residual, post, comb):
        return torch.ops.custom.npu_hc_post(hidden_states, residual, post, comb)


class DeepseekV4DecoderLayer(nn.Module):

    def __init__(self, config: DeepseekV4Config, layer_idx: int, ep_size: int = 1, ep_rank: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        self.self_attn = DeepseekV4Attention(config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV4MoE(config, layer_idx, ep_size=ep_size, ep_rank=ep_rank)

        hc_dim = self.hc_mult * config.hidden_size
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        torch.set_default_dtype(origin_dtype)

        self.attn_norm = MojoRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)
        self.ffn_norm = MojoRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale,
            self.hc_attn_base, self.hc_mult, self.hc_sinkhorn_iters,
            self.norm_eps, self.hc_eps
        )
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            position_embeddings=position_embeddings,
            is_prefill=is_prefill,
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale,
            self.hc_ffn_base, self.hc_mult, self.hc_sinkhorn_iters,
            self.norm_eps, self.hc_eps
        )
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids=input_ids, is_prefill=is_prefill)
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        return hidden_states


class DeepseekV4Model(nn.Module):

    def __init__(self, config: DeepseekV4Config, ep_size: int = 1, ep_rank: int = 0):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepseekV4DecoderLayer(config, layer_idx, ep_size=ep_size, ep_rank=ep_rank) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = MojoRMSNorm(eps=config.rms_norm_eps, norm_size=config.hidden_size)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config=config)
        self.compress_rotary_emb = DeepseekV4RotaryEmbedding(config=config, base=config.compress_rope_theta)

        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        hc_dim = config.hc_mult * config.hidden_size
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))
        torch.set_default_dtype(origin_dtype)

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: Optional[bool] = None,
        is_prefill: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, PagedDummyCache]:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if past_key_values is None:
            past_key_values = PagedDummyCache(
                self.config,
                batch_size=batch_size,
                device=str(device),
                block_size=128,
                max_seq_len=max(seq_len * 4, 4096),
                pa_max_length=self.config.pa_max_length,
                next_n=self.config.next_n,
            )

        past_len = int(past_key_values.get_seq_length(0).max().item())
        position_ids = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)
        cmp_cos, cmp_sin = self.compress_rotary_emb(hidden_states, position_ids)
        compress_position_embeddings = (cmp_cos, cmp_sin)

        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_position_embeddings = (
                compress_position_embeddings
                if self.config.compress_ratios[layer_idx] > 1
                else position_embeddings
            )
            hidden_states = decoder_layer(
                hidden_states, attention_mask=attention_mask,
                position_embeddings=layer_position_embeddings, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache,
                input_ids=input_ids, is_prefill=is_prefill, **kwargs,
            )

        hidden_states = self._hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class DeepseekV4ForCausalLM(nn.Module):

    def __init__(self, config, num_layers=None, ep_size=1, ep_rank=0):
        super().__init__()
        if not isinstance(config, DeepseekV4Config):
            config = DeepseekV4Config._from_hf_config(config)
        if num_layers is not None and num_layers < config.num_hidden_layers:
            config = DeepseekV4Config(**{k: getattr(config, k) for k in [
                "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "moe_intermediate_size",
                "n_shared_experts", "n_routed_experts", "num_experts_per_tok",
                "routed_scaling_factor", "n_group", "topk_group",
                "first_k_dense_replace", "norm_topk_prob", "scoring_func", "topk_method",
                "head_dim", "q_lora_rank", "qk_rope_head_dim", "o_lora_rank", "o_groups",
                "sliding_window", "hc_mult", "hc_sinkhorn_iters",
                "hc_eps", "index_n_heads", "index_head_dim", "index_topk",
                "num_hash_layers", "attention_bias", "attention_dropout",
                "rms_norm_eps", "hidden_act", "rope_theta", "compress_rope_theta",
                "max_position_embeddings", "swiglu_limit", "rope_scaling",
            ]})
            config.num_hidden_layers = num_layers
            config.compress_ratios = config.compress_ratios[:num_layers]
        self.config = config
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.model = DeepseekV4Model(config, ep_size=ep_size, ep_rank=ep_rank)
        self.lm_head = MojoGemm(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)
        self.hccl_comm_dict = {}

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: Optional[bool] = None,
        is_prefill: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, PagedDummyCache]:
        hidden_states, past_key_values = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            is_prefill=is_prefill, **kwargs,
        )
        hidden_states_flat = hidden_states.view(-1, self.config.hidden_size).to(torch.bfloat16)
        logits = self.lm_head(hidden_states_flat)
        logits = logits.view(*hidden_states.shape[:-1], self.config.vocab_size)
        return logits, past_key_values

    def init_parallel_comm_group(self):
        if not dist.is_initialized():
            logging.warning("dist not initialized, skip init_parallel_comm_group")
            return
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        if self.ep_size > 1:
            hccl_buffer_size = int(os.environ.get("HCCL_BUFFSIZE", "200"))
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
            moe_ep_group = dist.new_group(
                ranks=list(range(world_size)),
                pg_options=options,
            )
            self.hccl_comm_dict["moe_ep_group"] = moe_ep_group

            mc2_buffer_size = int(os.environ.get("MC2_BUFFSIZE", str(max(200, self.config.moe_intermediate_size * self.config.hidden_size * self.ep_size // (1024 * 1024) + 100))))
            options_mc2 = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            options_mc2.hccl_config = {"hccl_buffer_size": mc2_buffer_size}
            moe_ep_group_mc2 = dist.new_group(
                ranks=list(range(world_size)),
                pg_options=options_mc2,
            )
            moe_ep_group_mc2_name = moe_ep_group_mc2._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
            self.hccl_comm_dict["moe_ep_group_mc2"] = moe_ep_group_mc2
            self.hccl_comm_dict["moe_ep_group_mc2_name"] = moe_ep_group_mc2_name
            logging.info(f"Created moe_ep_group and moe_ep_group_mc2: world_size={world_size}, ep_size={self.ep_size}, rank={global_rank}")
        else:
            self.hccl_comm_dict["moe_ep_group"] = None
            self.hccl_comm_dict["moe_ep_group_mc2"] = None
            self.hccl_comm_dict["moe_ep_group_mc2_name"] = None

    def set_ep_group(self):
        moe_ep_group = self.hccl_comm_dict.get("moe_ep_group")
        self.moe_ep_group = moe_ep_group
        for layer in self.model.layers:
            layer.mlp.ep_group = moe_ep_group
            layer.mlp.hccl_comm_dict = self.hccl_comm_dict

    @staticmethod
    def _align_weight(weight, target):
        if weight.shape != target.shape:
            if weight.dim() == 2 and target.dim() == 1 and weight.shape[1] == 1:
                weight = weight.squeeze(-1)
            elif weight.dim() == 1 and target.dim() == 2 and target.shape[1] == 1:
                weight = weight.unsqueeze(-1)
        if weight.dtype != target.dtype:
            if target.dtype == torch.bfloat16 and weight.dtype == torch.float32:
                weight = weight.to(torch.bfloat16)
            elif target.dtype == torch.float32 and weight.dtype == torch.bfloat16:
                weight = weight.to(torch.float32)
            elif target.dtype == torch.int32 and weight.dtype == torch.int64:
                weight = weight.to(torch.int32)
        return weight

    @staticmethod
    def _init_default_weights(model):
        for module_name, module in model.named_modules():
            cls_name = type(module).__name__
            if "RMSNorm" in cls_name:
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data.fill_(1.0)
            elif "DynamicQuant" in cls_name:
                if hasattr(module, 'inv_smooth_scale') and module.inv_smooth_scale is not None:
                    module.inv_smooth_scale.data.fill_(1.0)

    @staticmethod
    def load_weights(model, weight_dir: str):
        from safetensors.torch import load_file

        DeepseekV4ForCausalLM._init_default_weights(model)

        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        name_mapping = DeepseekV4ForCausalLM._build_name_mapping(model)
        needed_checkpoint_keys = set(name_mapping.keys())

        max_layer_idx = model.config.num_hidden_layers - 1
        needed_checkpoint_keys = {
            k for k in needed_checkpoint_keys
            if not k.startswith("layers.") or int(k.split(".")[1]) <= max_layer_idx
        }

        for layer_idx in range(model.config.num_hidden_layers):
            pfx = f"layers.{layer_idx}.ffn.experts"
            ep_start = model.ep_rank * (model.config.n_routed_experts // model.ep_size)
            ep_end = ep_start + model.config.n_routed_experts // model.ep_size
            for eid in range(ep_start, ep_end):
                for w in ["w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"]:
                    needed_checkpoint_keys.add(f"{pfx}.{eid}.{w}")

        file_to_keys = {}
        for ck_key in needed_checkpoint_keys:
            if ck_key in weight_map:
                sf = weight_map[ck_key]
                file_to_keys.setdefault(sf, []).append(ck_key)

        params_dict = dict(model.named_parameters())
        buffers_dict = dict(model.named_buffers())

        expert_weights = {}
        for sf_file in sorted(file_to_keys.keys()):
            fp = os.path.join(weight_dir, sf_file)
            data = load_file(fp)
            for ck_key in file_to_keys[sf_file]:
                if ck_key not in data:
                    continue
                weight = data[ck_key]
                if "experts." in ck_key and ".ffn." in ck_key and "shared_experts" not in ck_key:
                    expert_weights[ck_key] = weight
                    continue
                model_name = name_mapping.get(ck_key)
                if model_name is None:
                    continue
                if model_name in params_dict:
                    param = params_dict[model_name]
                    weight = DeepseekV4ForCausalLM._align_weight(weight, param)
                    if param.shape == weight.shape:
                        param.data.copy_(weight)
                elif model_name in buffers_dict:
                    buf = buffers_dict[model_name]
                    weight = DeepseekV4ForCausalLM._align_weight(weight, buf)
                    if buf.shape == weight.shape:
                        buf.data.copy_(weight)
            del data

        DeepseekV4ForCausalLM._load_expert_weights(model, expert_weights)

        for layer_idx in range(model.config.num_hidden_layers):
            mlp = model.model.layers[layer_idx].mlp
            if hasattr(mlp, 'process_expert_weights'):
                mlp.process_expert_weights()

        if model.ep_size > 1 and dist.is_initialized():
            moe_ep_group = model.hccl_comm_dict.get("moe_ep_group")
            if moe_ep_group is not None:
                for layer_idx in range(model.config.num_hidden_layers):
                    mlp = model.model.layers[layer_idx].mlp
                    if hasattr(mlp, 'smooth_scale_1') and mlp.smooth_scale_1 is not None:
                        all_smooth_scale_1 = mlp.smooth_scale_1.data.new_empty(
                            mlp.smooth_scale_1.data.shape[0] * model.ep_size,
                            mlp.smooth_scale_1.data.shape[1])
                        dist.all_gather_into_tensor(
                            all_smooth_scale_1, mlp.smooth_scale_1.data,
                            group=moe_ep_group)
                        mlp.smooth_scale_1.data = all_smooth_scale_1

    @staticmethod
    def _build_name_mapping(model):
        mapping = {}
        mapping["embed.weight"] = "model.embed_tokens.weight"
        mapping["head.weight"] = "lm_head.weight"
        mapping["norm.weight"] = "model.norm.weight"
        mapping["hc_head_fn"] = "model.hc_head_fn"
        mapping["hc_head_base"] = "model.hc_head_base"
        mapping["hc_head_scale"] = "model.hc_head_scale"

        for layer_idx in range(model.config.num_hidden_layers):
            pfx = f"layers.{layer_idx}"
            mpfx = f"model.layers.{layer_idx}"

            mapping[f"{pfx}.hc_attn_fn"] = f"{mpfx}.hc_attn_fn"
            mapping[f"{pfx}.hc_attn_base"] = f"{mpfx}.hc_attn_base"
            mapping[f"{pfx}.hc_attn_scale"] = f"{mpfx}.hc_attn_scale"
            mapping[f"{pfx}.hc_ffn_fn"] = f"{mpfx}.hc_ffn_fn"
            mapping[f"{pfx}.hc_ffn_base"] = f"{mpfx}.hc_ffn_base"
            mapping[f"{pfx}.hc_ffn_scale"] = f"{mpfx}.hc_ffn_scale"
            mapping[f"{pfx}.attn_norm.weight"] = f"{mpfx}.attn_norm.weight"
            mapping[f"{pfx}.ffn_norm.weight"] = f"{mpfx}.ffn_norm.weight"

            mapping[f"{pfx}.attn.wq_a.weight"] = f"{mpfx}.self_attn.wq_a.weight"
            mapping[f"{pfx}.attn.q_norm.weight"] = f"{mpfx}.self_attn.q_norm.weight"
            mapping[f"{pfx}.attn.wq_b.weight"] = f"{mpfx}.self_attn.wq_b.weight"
            mapping[f"{pfx}.attn.wq_b.scale"] = f"{mpfx}.self_attn.wq_b.weight_scale"
            mapping[f"{pfx}.attn.wkv.weight"] = f"{mpfx}.self_attn.wkv.weight"
            mapping[f"{pfx}.attn.kv_norm.weight"] = f"{mpfx}.self_attn.kv_norm.weight"
            mapping[f"{pfx}.attn.wo_a.weight"] = f"{mpfx}.self_attn.wo_a.weight"
            mapping[f"{pfx}.attn.wo_b.weight"] = f"{mpfx}.self_attn.wo_b.weight"
            mapping[f"{pfx}.attn.wo_b.scale"] = f"{mpfx}.self_attn.wo_b.weight_scale"
            mapping[f"{pfx}.attn.attn_sink"] = f"{mpfx}.self_attn.attn_sink"

            mapping[f"{pfx}.attn.compressor.wkv.weight"] = f"{mpfx}.self_attn.sfa_compressor.wkv.weight"
            mapping[f"{pfx}.attn.compressor.wgate.weight"] = f"{mpfx}.self_attn.sfa_compressor.wgate.weight"
            mapping[f"{pfx}.attn.compressor.ape"] = f"{mpfx}.self_attn.sfa_compressor.ape"
            mapping[f"{pfx}.attn.compressor.norm.weight"] = f"{mpfx}.self_attn.sfa_compressor.norm.weight"

            mapping[f"{pfx}.attn.indexer.wq_b.weight"] = f"{mpfx}.self_attn.indexer.wq_b.weight"
            mapping[f"{pfx}.attn.indexer.wq_b.scale"] = f"{mpfx}.self_attn.indexer.wq_b.weight_scale"
            mapping[f"{pfx}.attn.indexer.weights_proj.weight"] = f"{mpfx}.self_attn.indexer.weights_proj.weight"
            mapping[f"{pfx}.attn.indexer.compressor.wkv.weight"] = f"{mpfx}.self_attn.indexer.compressor.wkv.weight"
            mapping[f"{pfx}.attn.indexer.compressor.wgate.weight"] = f"{mpfx}.self_attn.indexer.compressor.wgate.weight"
            mapping[f"{pfx}.attn.indexer.compressor.ape"] = f"{mpfx}.self_attn.indexer.compressor.ape"
            mapping[f"{pfx}.attn.indexer.compressor.norm.weight"] = f"{mpfx}.self_attn.indexer.compressor.norm.weight"

            mapping[f"{pfx}.ffn.gate.weight"] = f"{mpfx}.mlp.gate"
            mapping[f"{pfx}.ffn.gate.bias"] = f"{mpfx}.mlp.e_score_correction_bias"
            mapping[f"{pfx}.ffn.gate.tid2eid"] = f"{mpfx}.mlp.tid2eid"

            mapping[f"{pfx}.ffn.shared_experts.w1.weight"] = f"{mpfx}.mlp.shared_experts.gate_proj.weight"
            mapping[f"{pfx}.ffn.shared_experts.w1.scale"] = f"{mpfx}.mlp.shared_experts.gate_proj.weight_scale"
            mapping[f"{pfx}.ffn.shared_experts.w3.weight"] = f"{mpfx}.mlp.shared_experts.up_proj.weight"
            mapping[f"{pfx}.ffn.shared_experts.w3.scale"] = f"{mpfx}.mlp.shared_experts.up_proj.weight_scale"
            mapping[f"{pfx}.ffn.shared_experts.w2.weight"] = f"{mpfx}.mlp.shared_experts.down_proj.weight"
            mapping[f"{pfx}.ffn.shared_experts.w2.scale"] = f"{mpfx}.mlp.shared_experts.down_proj.weight_scale"

        return mapping

    @staticmethod
    def _load_expert_weights(model, expert_weights):
        ep_size = model.ep_size
        ep_rank = model.ep_rank
        experts_per_rank = model.config.n_routed_experts // ep_size
        ep_start = ep_rank * experts_per_rank
        intermediate_size = model.config.moe_intermediate_size
        hidden_size = model.config.hidden_size

        for layer_idx in range(model.config.num_hidden_layers):
            experts_mod = model.model.layers[layer_idx].mlp.experts

            up_proj_w = torch.empty(experts_per_rank, intermediate_size * 2, hidden_size, dtype=torch.int8)
            up_proj_s = torch.empty(experts_per_rank, intermediate_size * 2, dtype=torch.bfloat16)
            down_proj_w = torch.empty(experts_per_rank, hidden_size, intermediate_size, dtype=torch.int8)
            down_proj_s = torch.empty(experts_per_rank, hidden_size, dtype=torch.bfloat16)

            for local_eid in range(experts_per_rank):
                global_eid = ep_start + local_eid
                w1_key = f"layers.{layer_idx}.ffn.experts.{global_eid}.w1.weight"
                w3_key = f"layers.{layer_idx}.ffn.experts.{global_eid}.w3.weight"
                w2_key = f"layers.{layer_idx}.ffn.experts.{global_eid}.w2.weight"
                s1_key = f"layers.{layer_idx}.ffn.experts.{global_eid}.w1.scale"
                s3_key = f"layers.{layer_idx}.ffn.experts.{global_eid}.w3.scale"
                s2_key = f"layers.{layer_idx}.ffn.experts.{global_eid}.w2.scale"

                if w1_key in expert_weights:
                    up_proj_w[local_eid, :intermediate_size, :] = expert_weights[w1_key]
                if w3_key in expert_weights:
                    up_proj_w[local_eid, intermediate_size:, :] = expert_weights[w3_key]
                if w2_key in expert_weights:
                    down_proj_w[local_eid] = expert_weights[w2_key]
                if s1_key in expert_weights:
                    up_proj_s[local_eid, :intermediate_size] = expert_weights[s1_key].squeeze(-1).to(torch.bfloat16)
                if s3_key in expert_weights:
                    up_proj_s[local_eid, intermediate_size:] = expert_weights[s3_key].squeeze(-1).to(torch.bfloat16)
                if s2_key in expert_weights:
                    down_proj_s[local_eid] = expert_weights[s2_key].squeeze(-1).to(torch.bfloat16)

            experts_mod.up_proj_weight.copy_(up_proj_w)
            experts_mod.up_proj_weight_scale.data.copy_(up_proj_s)
            experts_mod.down_proj_weight.copy_(down_proj_w)
            experts_mod.down_proj_weight_scale.data.copy_(down_proj_s)
