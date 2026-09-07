import json
import math
import os
import time

from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mojo_opset import MojoBatchGemm
from mojo_opset import MojoDequantSwiGLUClampQuant
from mojo_opset import MojoDynamicQuant
from mojo_opset import MojoGemm
from mojo_opset import MojoHcPost
from mojo_opset import MojoHcPre
from mojo_opset import MojoInplacePartialRotaryMul
from mojo_opset import MojoMoECombine
from mojo_opset import MojoMoEDispatch
from mojo_opset import MojoMoEDynamicQuant
from mojo_opset import MojoMoEGatingTopK
from mojo_opset import MojoQuantGemm
from mojo_opset import MojoQuantGroupGemm
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoScatterNdUpdate

_DYNAMIC_QUANT_PER_TOKEN = MojoDynamicQuant()
_NPU_FRACTAL_NZ = 29

_DSV4_LAYER_PROFILE = os.getenv("DSV4_LAYER_PROFILE", "0") == "1"
_DSV4_LAYER_PROFILE_STATS = {}


def _profile_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _profile_sync():
    npu = getattr(torch, "npu", None)
    if _DSV4_LAYER_PROFILE and npu is not None:
        npu.synchronize()


def _profile_record(layer_idx: int, name: str, elapsed_ms: float):
    if not _DSV4_LAYER_PROFILE or not _profile_rank0():
        return
    layer_stats = _DSV4_LAYER_PROFILE_STATS.setdefault(int(layer_idx), {})
    layer_stats.setdefault(name, []).append(float(elapsed_ms))


class _ProfileTimer:
    def __init__(self, layer_idx: int, name: str):
        self.layer_idx = layer_idx
        self.name = name
        self.start = 0.0

    def __enter__(self):
        if _DSV4_LAYER_PROFILE:
            _profile_sync()
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if _DSV4_LAYER_PROFILE:
            _profile_sync()
            _profile_record(self.layer_idx, self.name, (time.perf_counter() - self.start) * 1000.0)
        return False


def _profile_timer(layer_idx: int, name: str) -> _ProfileTimer:
    return _ProfileTimer(layer_idx, name)


def reset_dsv4_layer_profile():
    _DSV4_LAYER_PROFILE_STATS.clear()


def get_dsv4_layer_profile():
    return {
        str(layer_idx): {
            name: {
                "count": len(values),
                "total_ms": sum(values),
                "avg_ms": sum(values) / len(values) if values else 0.0,
                "values_ms": values,
            }
            for name, values in stats.items()
        }
        for layer_idx, stats in sorted(_DSV4_LAYER_PROFILE_STATS.items())
    }


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


def _apply_partial_rotary(x, cos, sin, rotary_mul):
    """Pack ``x[B, S, N, D]`` and per-token cos/sin for partial RoPE."""
    output = rotary_mul(
        x.flatten(0, 1).unsqueeze(2),
        cos.reshape(-1, 1, 1, cos.shape[-1]),
        sin.reshape(-1, 1, 1, sin.shape[-1]),
    )
    return output.view_as(x)


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
        self.rope_scaling = kwargs.get(
            "rope_scaling",
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "type": "yarn",
            },
        )

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
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "moe_intermediate_size",
            "n_shared_experts",
            "n_routed_experts",
            "num_experts_per_tok",
            "routed_scaling_factor",
            "n_group",
            "topk_group",
            "first_k_dense_replace",
            "norm_topk_prob",
            "scoring_func",
            "topk_method",
            "head_dim",
            "q_lora_rank",
            "qk_rope_head_dim",
            "o_lora_rank",
            "o_groups",
            "sliding_window",
            "compress_ratios",
            "next_n",
            "pa_max_length",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "num_hash_layers",
            "attention_bias",
            "attention_dropout",
            "rms_norm_eps",
            "hidden_act",
            "rope_theta",
            "compress_rope_theta",
            "max_position_embeddings",
            "swiglu_limit",
        ]:
            if hasattr(hf_config, attr):
                kwargs[attr] = getattr(hf_config, attr)
        if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
            kwargs["rope_scaling"] = dict(hf_config.rope_scaling)
        return cls(**kwargs)


class PagedDummyCache:
    def __init__(
        self,
        config: DeepseekV4Config,
        batch_size: int,
        device: str,
        block_size: int = 128,
        max_seq_len: int = 4096,
        pa_max_length: Optional[int] = None,
        next_n: Optional[int] = None,
    ):
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
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.block_tables = torch.full(
            (self.num_layers, self.batch_size, max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.seq_lens = torch.zeros(
            (self.num_layers, self.batch_size),
            dtype=torch.int32,
            device=self.device,
        )
        self.scatter_nd_update = MojoScatterNdUpdate()
        self.li_dynamic_quant = MojoDynamicQuant()

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
                    torch.arange(
                        0, self.batch_size * cmp_block_num_per_batch, dtype=torch.int32, device=self.device
                    ).view(self.batch_size, -1)
                    + 1
                )
            elif ratio == 128:
                cmp_block_num = self._get_block_num(self.pa_max_length // ratio)
                overlap_num = 1
                state_block_num = self._get_block_num(overlap_num * ratio)
                cache_dict["sfa_cmp_kv"] = self._create_cache(cmp_block_num, self.head_dim, torch.bfloat16)
                cache_dict["sfa_kv_state"] = self._create_state_cache(state_block_num, ratio, self.head_dim)
                cmp_block_num_per_batch = (cmp_block_num - 1) // self.batch_size
                cache_dict["c128a_cmp_kv_block_table"] = (
                    torch.arange(
                        0, self.batch_size * cmp_block_num_per_batch, dtype=torch.int32, device=self.device
                    ).view(self.batch_size, -1)
                    + 1
                )

            self.cache_data[layer_idx] = cache_dict

    def _get_block_num(self, cache_size):
        return math.ceil(cache_size / self.block_size) * self.batch_size + 1

    def _create_cache(self, block_num, dim, dtype):
        return torch.zeros(
            (block_num, self.block_size, 1, dim),
            dtype=dtype,
            device=self.device,
        )

    def _calc_full_block_table(self, cache_size: int, batch_size: int) -> torch.Tensor:
        block_num_per_batch = math.ceil(cache_size / self.block_size)
        return (
            torch.arange(0, batch_size * block_num_per_batch, dtype=torch.int32, device=self.device).view(
                batch_size, -1
            )
            + 1
        )

    def _calc_ring_block_table(self, cache_size: int, batch_size: int) -> torch.Tensor:
        block_num_per_batch = math.ceil(cache_size / self.block_size)
        block_table_len = math.ceil(self.pa_max_length / self.block_size)
        block_table_offset = (
            torch.arange(0, batch_size * block_num_per_batch, dtype=torch.int32, device=self.device).view(
                batch_size, -1
            )
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
            torch.arange(0, batch_size * block_num_per_batch, dtype=torch.int32, device=self.device).view(
                batch_size, -1
            )
            + 1
        )
        repeat_num = math.ceil(block_table_len / block_num_per_batch)
        block_table_offset = block_table_offset.repeat(1, repeat_num)[:, :block_table_len]
        block_pos_ids = torch.arange(block_table_len, dtype=torch.int32, device=self.device).repeat(batch_size, 1)
        actual_seq_len = start_pos + seq_used_q
        actual_block_start = (start_pos // self.block_size).view(batch_size, 1)
        actual_block_end = ((actual_seq_len - 1) // self.block_size).view(batch_size, 1)
        if is_prefill:
            return torch.where(
                block_pos_ids == actual_block_end, block_table_offset, torch.zeros_like(block_table_offset)
            )
        block_table = torch.where(
            block_pos_ids >= actual_block_start, block_table_offset, torch.zeros_like(block_table_offset)
        )
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
        start_pos = start_pos.to(device=self.device, dtype=torch.int32)
        seq_used_q = seq_used_q.to(device=self.device, dtype=torch.int32)
        batch_size = start_pos.shape[0]
        block_table = self._calc_ring_block_table(self.win_cache_size, start_pos.shape[0])
        if pad_to_window:
            seq_len = self.sliding_window
            base_pos = torch.clamp(start_pos + seq_used_q - self.sliding_window, min=0)
            valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        else:
            seq_len = int(seq_used_q.max().item()) if batch_size > 0 else 0
            if seq_len == 0:
                return torch.empty((0,), dtype=torch.int32, device=self.device)
            base_pos = start_pos
            valid_mask = torch.arange(seq_len, dtype=torch.int32, device=self.device).unsqueeze(
                0
            ) < seq_used_q.unsqueeze(1)

        offsets_in_seq = torch.arange(seq_len, dtype=torch.int32, device=self.device).unsqueeze(0)
        positions = base_pos.unsqueeze(1) + offsets_in_seq
        block_idx = (positions // self.block_size).to(torch.long)
        block_offset = positions % self.block_size
        row_idx = torch.arange(batch_size, device=self.device, dtype=torch.long).unsqueeze(1).expand_as(block_idx)
        slots = block_table[row_idx, block_idx] * self.block_size + block_offset
        return slots[valid_mask].to(dtype=torch.int32)

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
        actual_q_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = kv.shape[0]
        seq_len = kv.shape[1]
        full_kv = self._create_cache(self._get_block_num(self.max_seq_len), self.head_dim, kv.dtype)
        block_table = self._calc_full_block_table(self.max_seq_len, batch_size)
        q_lens = (cu_q_lens[1:] - cu_q_lens[:-1]).to(device=kv.device, dtype=torch.int32)
        if actual_q_lens is not None:
            q_lens = actual_q_lens.to(device=kv.device, dtype=torch.int32)
        token_offsets = torch.arange(seq_len, dtype=torch.int32, device=kv.device).unsqueeze(0)
        valid_mask = token_offsets < q_lens.unsqueeze(1)
        positions = context_lens.to(device=kv.device, dtype=torch.int32).unsqueeze(1) + token_offsets
        block_idx = (positions // self.block_size).to(torch.long)
        valid_mask = valid_mask & (block_idx < block_table.shape[1])

        if bool(valid_mask.any().item()):
            row_idx = torch.arange(batch_size, device=kv.device, dtype=torch.long).unsqueeze(1).expand_as(block_idx)
            offset = positions % self.block_size
            slot_mapping = (
                block_table[row_idx, block_idx.clamp(max=block_table.shape[1] - 1)] * self.block_size + offset
            )
            kv_flat = kv[valid_mask].reshape(-1, self.head_dim)
            slot_mapping = slot_mapping[valid_mask].to(dtype=torch.int32)
            self.scatter_nd_update(
                full_kv.view(-1, self.head_dim),
                slot_mapping.reshape(-1, 1),
                kv_flat,
            )
        return full_kv, block_table

    def _create_state_cache(self, state_block_num, compress_ratio, cache_dim):
        overlap_num = 2 if compress_ratio == 4 else 1
        return torch.zeros(
            (state_block_num, self.block_size, 2, overlap_num, cache_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def update(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        cu_q_lens: Optional[torch.Tensor] = None,
        actual_q_lens: Optional[torch.Tensor] = None,
    ) -> None:
        batch_size = kv.shape[0]
        new_seq_len = kv.shape[1]

        if cu_q_lens is None:
            cu_q_lens = torch.arange(
                0,
                (batch_size + 1) * new_seq_len,
                step=new_seq_len,
                device=kv.device,
                dtype=torch.int32,
            )

        current_seq_lens = self.seq_lens[layer_idx]
        q_lens = (cu_q_lens[1:] - cu_q_lens[:-1]).to(device=kv.device, dtype=torch.int32)
        if actual_q_lens is not None:
            q_lens = actual_q_lens.to(device=kv.device, dtype=torch.int32)
        new_total_lens = current_seq_lens + q_lens

        max_blocks_per_seq = self.block_tables.shape[2]
        logical_blocks = torch.arange(max_blocks_per_seq, dtype=torch.int32, device=kv.device).unsqueeze(0)
        required_blocks = (new_total_lens + self.block_size - 1) // self.block_size
        required_mask = logical_blocks < required_blocks.unsqueeze(1)

        # Use a deterministic physical block layout, equivalent to golden's static PA table.
        layer_base = layer_idx * batch_size * max_blocks_per_seq
        batch_base = torch.arange(batch_size, dtype=torch.int32, device=kv.device).unsqueeze(1) * max_blocks_per_seq
        deterministic_blocks = layer_base + batch_base + logical_blocks
        self.block_tables[layer_idx] = torch.where(
            required_mask,
            deterministic_blocks,
            self.block_tables[layer_idx],
        )

        token_offsets = torch.arange(new_seq_len, dtype=torch.int32, device=kv.device).unsqueeze(0)
        valid_mask = token_offsets < q_lens.unsqueeze(1)
        positions = current_seq_lens.unsqueeze(1) + token_offsets
        block_idx = (positions // self.block_size).to(torch.long)
        valid_mask = valid_mask & (block_idx < max_blocks_per_seq)

        row_idx = torch.arange(batch_size, device=kv.device, dtype=torch.long).unsqueeze(1).expand_as(block_idx)
        block_idx_safe = block_idx.clamp(max=max_blocks_per_seq - 1)
        phys_block = self.block_tables[layer_idx][row_idx, block_idx_safe]
        slot_mapping = (phys_block * self.block_size + (positions % self.block_size)).to(dtype=torch.int32)
        kv_flat = kv[valid_mask].reshape(-1, self.head_dim)
        slot_mapping = slot_mapping[valid_mask]
        cache_flat = self.kv_cache.view(-1, self.head_dim)
        self.scatter_nd_update(cache_flat, slot_mapping.reshape(-1, 1), kv_flat)
        self.seq_lens[layer_idx] = new_total_lens.to(self.seq_lens.dtype)

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
            local_idx = gather_indices.to(kv.device, dtype=torch.int32) - start_pos.to(
                kv.device, dtype=torch.int32
            ).unsqueeze(1)
            local_idx = torch.where(
                (local_idx >= 0) & (local_idx < seq_len),
                local_idx,
                torch.zeros_like(local_idx),
            ).to(torch.long)
            row_idx = torch.arange(batch_size, device=kv.device, dtype=torch.long).unsqueeze(1).expand_as(local_idx)
            kv_flat = kv[row_idx, local_idx].reshape(-1, self.head_dim)
        else:
            kv_flat = kv.reshape(-1, self.head_dim)
        win_flat = win_cache.view(-1, self.head_dim)
        self.scatter_nd_update(win_flat, slot_mapping.reshape(-1, 1), kv_flat)

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
            self.scatter_nd_update(cmp_flat, slot_mapping.reshape(-1, 1), kv_flat)
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
        kv_quant, k_scale = self.li_dynamic_quant(kv_flat)
        k_scale = k_scale.squeeze(-1).to(torch.float16)
        cmp_flat = li_cmp_cache.view(-1, self.index_head_dim)
        scale_flat = scale_cache.view(-1, scale_cache.shape[-1])
        if slot_mapping is not None:
            self.scatter_nd_update(
                scale_flat,
                slot_mapping.reshape(-1, 1),
                k_scale.view(-1, scale_cache.shape[-1]),
            )
            self.scatter_nd_update(
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
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
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


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int) -> torch.Tensor:
    prefix = x[..., :-rope_head_dim]
    rope = x[..., -rope_head_dim:].float()
    rope = (rope * cos[..., :rope_head_dim].float() + _rotate_interleaved(rope) * sin[..., :rope_head_dim].float()).to(
        x.dtype
    )
    return torch.cat((prefix, rope), dim=-1) if prefix.shape[-1] else rope


def _gather_paged_rows(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    if cache.dim() == 3:
        cache = cache.unsqueeze(2)
    logical_positions = logical_positions.to(device=cache.device, dtype=torch.long)
    block_size = cache.shape[1]
    logical_blocks = torch.div(logical_positions, block_size, rounding_mode="floor")
    physical_blocks = block_table_row.to(device=cache.device).index_select(0, logical_blocks).long()
    offsets = torch.remainder(logical_positions, block_size)
    return cache[physical_blocks, offsets]


def compress_kv(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_head_dim: int,
    cmp_ratio: int,
    state_block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    coff: int = 1,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Compress packed ``x[T, H]`` and update the paged compression state."""
    boundaries = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    ranges = list(zip(boundaries, boundaries[1:]))
    batch_size = len(ranges)
    start_values = (
        [0] * batch_size
        if start_pos is None
        else [int(value) for value in start_pos.detach().cpu().tolist()]
    )
    active_lengths = (
        [end - begin for begin, end in ranges]
        if seqused is None
        else [int(value) for value in seqused.detach().cpu().tolist()]
    )

    head_dim = int(norm_weight.numel())
    projected_dim = coff * head_dim
    kv_projected = F.linear(x.float(), wkv.float())
    gate_projected = F.linear(x.float(), wgate.float())
    block_size = state_cache.shape[1]

    local_state: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def mapped_block(batch_idx: int, absolute_pos: int, for_write: bool) -> Optional[int]:
        logical_block = absolute_pos // block_size
        table_row = state_block_table[batch_idx]
        block_id = int(table_row[logical_block].item()) if logical_block < table_row.numel() else 0
        if block_id > 0:
            return block_id
        if for_write:
            return None
        positive = torch.unique(table_row[table_row > 0])
        return int(positive.item()) if positive.numel() == 1 else None

    def load_state(batch_idx: int, absolute_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        local = local_state.get((batch_idx, absolute_pos))
        if local is not None:
            return local
        block_id = mapped_block(batch_idx, absolute_pos, False)
        if block_id is None:
            raise RuntimeError("compressed state is not mapped")
        cached = state_cache[block_id, absolute_pos % block_size]
        return cached[:projected_dim], cached[projected_dim:]

    packed_outputs: List[torch.Tensor] = []
    for batch_idx, ((begin, _), active_len, absolute_start) in enumerate(zip(ranges, active_lengths, start_values)):
        for relative_pos in range(active_len):
            flat_idx = begin + relative_pos
            absolute_pos = absolute_start + relative_pos
            kv_value = kv_projected[flat_idx]
            score_value = gate_projected[flat_idx] + ape[absolute_pos % cmp_ratio].float()
            local_state[(batch_idx, absolute_pos)] = (kv_value, score_value)

            block_id = mapped_block(batch_idx, absolute_pos, True)
            if block_id is not None:
                state_cache[block_id, absolute_pos % block_size, :projected_dim] = kv_value
                state_cache[block_id, absolute_pos % block_size, projected_dim:] = score_value

            if (absolute_pos + 1) % cmp_ratio:
                continue

            group_start = absolute_pos + 1 - cmp_ratio
            if coff == 1:
                group = [load_state(batch_idx, pos) for pos in range(group_start, absolute_pos + 1)]
                kv_group = torch.stack([item[0] for item in group])
                score_group = torch.stack([item[1] for item in group])
            else:
                previous: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for pos in range(group_start - cmp_ratio, group_start):
                    if pos < 0:
                        previous.append(
                            (kv_value.new_zeros(head_dim), score_value.new_full((head_dim,), float("-inf")))
                        )
                    else:
                        prev_kv, prev_score = load_state(batch_idx, pos)
                        previous.append((prev_kv[:head_dim], prev_score[:head_dim]))
                current = [load_state(batch_idx, pos) for pos in range(group_start, absolute_pos + 1)]
                kv_group = torch.cat(
                    (
                        torch.stack([item[0] for item in previous]),
                        torch.stack([item[0][head_dim:] for item in current]),
                    ),
                    dim=0,
                )
                score_group = torch.cat(
                    (
                        torch.stack([item[1] for item in previous]),
                        torch.stack([item[1][head_dim:] for item in current]),
                    ),
                    dim=0,
                )

            pooled = (kv_group * torch.softmax(score_group, dim=0)).sum(dim=0).to(x.dtype)
            pooled_fp = pooled.float()
            pooled = (
                pooled_fp
                * torch.rsqrt(pooled_fp.square().mean(dim=-1, keepdim=True) + float(norm_eps))
                * norm_weight.float()
            ).to(x.dtype)
            packed_outputs.append(pooled)

    rope_cos = rope_cos.reshape(-1, rope_cos.shape[-1])
    rope_sin = rope_sin.reshape(-1, rope_sin.shape[-1])
    output = x.new_zeros((rope_cos.shape[0], head_dim))
    for output_idx, value in enumerate(packed_outputs):
        output[output_idx] = _apply_rope(value, rope_cos[output_idx], rope_sin[output_idx], rope_head_dim)
    return output


class DeepseekV4Compressor(nn.Module):
    def __init__(
        self, config: DeepseekV4Config, compress_ratio: int, head_dim: Optional[int] = None, is_indexer: bool = False
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim if head_dim is not None else config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap
        self.is_indexer = is_indexer
        if self.is_indexer:
            self.register_buffer("hadamard_matrix", _get_had_pow2(self.head_dim), persistent=False)

        self.wkv = MojoGemm(
            in_features=self.hidden_size, out_features=self.coff * self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.wgate = MojoGemm(
            in_features=self.hidden_size, out_features=self.coff * self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.norm = MojoRMSNorm(norm_size=self.head_dim, eps=config.rms_norm_eps)
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * self.head_dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        state_cache: Optional[torch.Tensor] = None,
        state_block_table: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        seq_used_q: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if state_cache is None or state_block_table is None:
            raise ValueError("DeepseekV4Compressor requires state_cache and state_block_table.")
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=x.device,
            )
        if seq_used_q is None:
            seq_used_q = torch.full((batch_size,), seq_len, dtype=torch.int32, device=x.device)
        if start_pos is None:
            start_pos = torch.zeros(batch_size, dtype=torch.int32, device=x.device)

        x_flat = x.to(torch.bfloat16).reshape(-1, self.hidden_size).contiguous()
        cmp_flat = compress_kv(
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
        )
        output_len = cos.reshape(-1, self.rope_head_dim).shape[0]
        cmp_flat = cmp_flat[:output_len]
        if self.is_indexer and cmp_flat.numel() > 0:
            cmp_flat = _rotate_activation(cmp_flat, self.hadamard_matrix)
        cmp_out = cmp_flat.view(1, output_len, self.head_dim)
        return cmp_out


def quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    sparse_count: int,
    cmp_ratio: int,
) -> torch.Tensor:
    """Select compressed-KV positions for packed INT8 ``query[T, N, D]``."""
    query_ends = [int(value) for value in actual_seq_lengths_query.detach().cpu().tolist()]
    query_ranges = list(zip([0] + query_ends[:-1], query_ends))
    key_lengths = [int(value) for value in actual_seq_lengths_key.detach().cpu().tolist()]
    num_query_heads = query.shape[1]
    num_key_heads = key.shape[2]
    query_heads_per_key = num_query_heads // num_key_heads
    query_scales = query_dequant_scale.reshape(-1, num_query_heads).float()
    query_weights = weights.reshape(-1, num_query_heads).float()
    if key_dequant_scale.dim() == 4:
        key_dequant_scale = key_dequant_scale.squeeze(-1)

    indices = torch.full(
        (query.shape[0], num_key_heads, sparse_count),
        -1,
        dtype=torch.int32,
        device=query.device,
    )
    for batch_idx, (query_begin, query_end) in enumerate(query_ranges):
        query_len = query_end - query_begin
        absolute_query_start = key_lengths[batch_idx] - query_len
        for local_query_idx, flat_query_idx in enumerate(range(query_begin, query_end)):
            absolute_query_pos = absolute_query_start + local_query_idx
            valid_key_count = (absolute_query_pos + 1) // int(cmp_ratio)
            if valid_key_count <= 0:
                continue

            logical_positions = torch.arange(valid_key_count, device=query.device, dtype=torch.long)
            key_rows = _gather_paged_rows(key, block_table[batch_idx], logical_positions).float()
            scale_rows = _gather_paged_rows(
                key_dequant_scale.unsqueeze(-1), block_table[batch_idx], logical_positions
            ).squeeze(-1)
            key_rows = key_rows * scale_rows.float().unsqueeze(-1)
            q_rows = query[flat_query_idx].float() * query_scales[flat_query_idx].unsqueeze(-1)

            for key_head_idx in range(num_key_heads):
                head_begin = key_head_idx * query_heads_per_key
                head_end = head_begin + query_heads_per_key
                correlations = torch.matmul(
                    q_rows[head_begin:head_end],
                    key_rows[:, key_head_idx].transpose(0, 1),
                )
                scores = (
                    torch.relu(correlations) * query_weights[flat_query_idx, head_begin:head_end].unsqueeze(-1)
                ).sum(dim=0)
                selected_count = min(int(sparse_count), valid_key_count)
                order = torch.argsort(scores, descending=True, stable=True)[:selected_count]
                indices[flat_query_idx, key_head_idx, :selected_count] = order.to(torch.int32)

    return indices


class DeepseekV4Indexer(nn.Module):
    def __init__(self, config: DeepseekV4Config, compress_ratio: int = 4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.rotary_mul = MojoInplacePartialRotaryMul(
            partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim]
        )
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = MojoQuantGemm(
            in_features=self.q_lora_rank,
            out_features=self.n_heads * self.head_dim,
            trans_weight=True,
        )
        self.weights_proj = MojoGemm(in_features=self.hidden_size, out_features=self.n_heads, bias=False)
        self.compressor = DeepseekV4Compressor(config, compress_ratio, head_dim=self.head_dim, is_indexer=True)
        self.compress_rotary_emb = DeepseekV4RotaryEmbedding(config, base=config.compress_rope_theta)
        self.register_buffer("hadamard_matrix", _get_had_pow2(self.head_dim), persistent=False)

    def prepare_quant_proj_weights(self) -> None:
        prepare_quant_gemm_weight(self.wq_b, scale_fp32=True)

    def forward(
        self,
        x,
        qr,
        cos,
        sin,
        past_key_values=None,
        layer_idx=0,
        cu_seqlens_q=None,
        seq_lens=None,
        start_pos: Optional[torch.Tensor] = None,
        state_block_table: Optional[torch.Tensor] = None,
        seq_used_q: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = x.shape

        with _profile_timer(layer_idx, "indexer_weights_proj"):
            weights = self.weights_proj(x.to(torch.bfloat16).reshape(-1, self.hidden_size))
            weights = weights.view(-1, self.n_heads) * (self.softmax_scale * self.n_heads**-0.5)

        li_state_cache = None
        if past_key_values is not None:
            li_state_cache = past_key_values.get_li_kv_state(layer_idx)
        logical_batch_size = int(seq_used_q.shape[0]) if seq_used_q is not None else batch_size
        if start_pos is None:
            start_pos = torch.zeros(logical_batch_size, dtype=torch.int32, device=x.device)
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=x.device,
            )
        if seq_used_q is None:
            seq_used_q = torch.full((logical_batch_size,), seq_len, dtype=torch.int32, device=x.device)
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
            cmp_cos = cos[:, :: self.compress_ratio, :]
            cmp_sin = sin[:, :: self.compress_ratio, :]
        with _profile_timer(layer_idx, "indexer_compressor"):
            li_kv = self.compressor(
                x,
                cmp_cos,
                cmp_sin,
                state_cache=li_state_cache,
                state_block_table=state_block_table,
                cu_seqlens=cu_seqlens_q,
                seq_used_q=seq_used_q,
                start_pos=start_pos,
            )

        if past_key_values is not None:
            with _profile_timer(layer_idx, "indexer_li_cache_update"):
                cmp_slot_mapping = past_key_values.get_cmp_slot_mapping(
                    layer_idx,
                    start_pos,
                    seq_used_q,
                    cu_seqlens_q=cu_seqlens_q,
                    compressed_len=compressed_len,
                    position_ids_cmp=(position_ids_cmp.squeeze(0) // self.compress_ratio).to(torch.int32),
                )
                past_key_values.update_li_cmp_kv(li_kv, layer_idx, cmp_slot_mapping)

        with _profile_timer(layer_idx, "indexer_q_proj_rope"):
            qr_flat = qr.reshape(-1, self.q_lora_rank).to(torch.bfloat16)
            qr_quant, qr_scale = _dynamic_quant_per_token(qr_flat)
            q = self.wq_b(qr_quant, qr_scale)
            q = q.view(1, -1, self.n_heads, self.head_dim)
            q = _apply_partial_rotary(q, cos, sin, self.rotary_mul)
            q = _rotate_activation(q, self.hadamard_matrix)

        if past_key_values is not None:
            li_cmp_kv = past_key_values.get_li_cmp_kv(layer_idx)
            li_key_dequant_scale = past_key_values.get_li_key_dequant_scale(layer_idx)
            c4a_block_table = past_key_values.get_c4a_cmp_kv_block_table(layer_idx)

            q_flat = q.flatten(0, 1)
            with _profile_timer(layer_idx, "indexer_q_quant"):
                q_quant, q_scale = _dynamic_quant_per_token(q_flat)
                q_scale = q_scale.to(torch.float16)

            actual_seq_q = (
                cu_seqlens_q[1:]
                if cu_seqlens_q is not None
                else torch.tensor([seq_len], dtype=torch.int32, device=x.device)
            )
            actual_seq_k = (
                seq_lens if seq_lens is not None else torch.tensor([seq_len], dtype=torch.int32, device=x.device)
            )

            with _profile_timer(layer_idx, "indexer_li_kernel"):
                topk_idxs = quant_lightning_indexer(
                    query=q_quant,
                    key=li_cmp_kv,
                    weights=weights.to(torch.float16),
                    query_dequant_scale=q_scale,
                    key_dequant_scale=li_key_dequant_scale.squeeze(-2),
                    actual_seq_lengths_query=actual_seq_q,
                    actual_seq_lengths_key=actual_seq_k,
                    block_table=c4a_block_table,
                    sparse_count=self.index_topk,
                    cmp_ratio=self.compress_ratio,
                )
            topk_idxs = topk_idxs.view(q_flat.shape[0], -1, self.index_topk)
            return topk_idxs

        return None


def _dynamic_quant_per_token(x: torch.Tensor):
    quant, scale = _DYNAMIC_QUANT_PER_TOKEN(x)
    return quant, scale.squeeze(-1)


def _to_nz(weight: torch.Tensor) -> torch.Tensor:
    if weight.device.type != "npu":
        return weight

    import torch_npu

    torch_npu.npu.config.allow_internal_format = True
    return torch_npu.npu_format_cast(weight.contiguous(), _NPU_FRACTAL_NZ)


def prepare_quant_gemm_weight(module, *, scale_fp32: bool = False) -> None:
    if scale_fp32 and module.weight_scale.dtype != torch.float32:
        module.weight_scale.data = module.weight_scale.data.float()
    if module.weight.device.type != "npu":
        return

    weight = module.weight.t() if module.trans_weight else module.weight
    module.weight = _to_nz(weight.contiguous())
    module.trans_weight = False
    module.weight_shape = tuple(weight.shape)


def sparse_attn_shared_kv(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
    ori_win_left: int,
    *,
    cmp_kv: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    cmp_ratio: int = 1,
) -> torch.Tensor:
    """Run DeepSeek V4 shared-KV attention for packed ``q[T, N, D]``."""
    boundaries = [int(value) for value in cu_seqlens_q.detach().cpu().tolist()]
    query_ranges = list(zip(boundaries, boundaries[1:]))
    key_lengths = [int(value) for value in seqused_kv.detach().cpu().tolist()]
    num_heads, head_dim = q.shape[1:]
    kv_heads = ori_kv.shape[2] if ori_kv.dim() == 4 else 1
    head_to_kv = torch.div(
        torch.arange(num_heads, device=q.device),
        num_heads // kv_heads,
        rounding_mode="floor",
    ).long()
    scale = float(softmax_scale) if softmax_scale else 1.0 / math.sqrt(head_dim)
    output = torch.zeros_like(q)

    for batch_idx, (query_begin, query_end) in enumerate(query_ranges):
        total_kv = key_lengths[batch_idx]
        absolute_query_start = total_kv - (query_end - query_begin)
        for local_query_idx, flat_query_idx in enumerate(range(query_begin, query_end)):
            absolute_query_pos = absolute_query_start + local_query_idx
            ori_begin = max(0, absolute_query_pos - int(ori_win_left))
            ori_positions = torch.arange(ori_begin, absolute_query_pos + 1, device=q.device, dtype=torch.long)
            candidate_kv = _gather_paged_rows(ori_kv, ori_block_table[batch_idx], ori_positions)

            if cmp_kv is not None:
                valid_cmp_count = (absolute_query_pos + 1) // int(cmp_ratio)
                if cmp_sparse_indices is None:
                    cmp_positions = torch.arange(valid_cmp_count, device=q.device, dtype=torch.long)
                else:
                    row = cmp_sparse_indices[flat_query_idx]
                    if row.dim() == 2:
                        row = row[0]
                    cmp_positions = row.to(device=q.device, dtype=torch.long)
                    cmp_positions = cmp_positions[(cmp_positions >= 0) & (cmp_positions < valid_cmp_count)]
                if cmp_positions.numel():
                    compressed = _gather_paged_rows(cmp_kv, cmp_block_table[batch_idx], cmp_positions)
                    candidate_kv = torch.cat((candidate_kv, compressed), dim=0)

            kv_by_query_head = candidate_kv.index_select(1, head_to_kv).float()
            scores = torch.einsum("hd,khd->hk", q[flat_query_idx].float(), kv_by_query_head) * scale
            if sinks is not None:
                probabilities = torch.softmax(torch.cat((scores, sinks.float().unsqueeze(-1)), dim=-1), dim=-1)[
                    ..., :-1
                ]
            else:
                probabilities = torch.softmax(scores, dim=-1)
            output[flat_query_idx] = torch.einsum("hk,khd->hd", probabilities, kv_by_query_head).to(q.dtype)
    return output


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
        self.rotary_mul = MojoInplacePartialRotaryMul(
            partial_slice=[self.head_dim - self.qk_rope_head_dim, self.head_dim]
        )
        self.scaling = self.head_dim ** (-0.5)

        raw_ratio = config.compress_ratios[layer_idx] if layer_idx < len(config.compress_ratios) else 0
        self.compress_ratio = raw_ratio if raw_ratio > 1 else 1
        self._is_c1a = raw_ratio == 0

        self.wq_a = MojoGemm(in_features=config.hidden_size, out_features=config.q_lora_rank, bias=False)
        self.q_norm = MojoRMSNorm(norm_size=config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_a_quant = MojoDynamicQuant(input_size=config.q_lora_rank)
        nn.init.ones_(self.q_a_quant.inv_smooth_scale)
        self.wq_b = MojoQuantGemm(
            in_features=config.q_lora_rank, out_features=self.num_heads * self.head_dim, trans_weight=True
        )
        self.q_b_norm = MojoRMSNorm(eps=config.rms_norm_eps, norm_size=self.head_dim)

        self.wkv = MojoGemm(
            in_features=config.hidden_size, out_features=self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.kv_norm = MojoRMSNorm(eps=config.rms_norm_eps, norm_size=self.head_dim)

        self.wo_a = MojoBatchGemm(
            num_groups=self.o_groups,
            in_features=self.num_heads * self.head_dim // self.o_groups,
            out_features=self.o_lora_rank,
        )
        self.wo_b = MojoQuantGemm(
            in_features=self.o_groups * self.o_lora_rank, out_features=config.hidden_size, trans_weight=True
        )

        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))

        if raw_ratio > 1:
            self.sfa_compressor = DeepseekV4Compressor(config, raw_ratio, head_dim=self.head_dim)
            self.compress_rotary_emb = DeepseekV4RotaryEmbedding(config, base=config.compress_rope_theta)
            self.indexer = DeepseekV4Indexer(config, raw_ratio) if raw_ratio == 4 else None
        else:
            self.sfa_compressor = None
            self.compress_rotary_emb = None
            self.indexer = None

    def prepare_quant_proj_weights(self) -> None:
        prepare_quant_gemm_weight(self.wq_b, scale_fp32=True)
        if isinstance(self.wo_b, MojoQuantGemm):
            prepare_quant_gemm_weight(self.wo_b)

    @staticmethod
    def _run_projection(module: nn.Module, input: torch.Tensor) -> torch.Tensor:
        if isinstance(module, MojoQuantGemm):
            input_quant, input_scale = _dynamic_quant_per_token(input)
            return module(input_quant, input_scale)
        return module(input)

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
        q_lens = kwargs.get("q_lens")
        if q_lens is None:
            q_lens = torch.full((batch_size,), seq_length, dtype=torch.int32, device=hidden_states.device)
        else:
            q_lens = q_lens.to(dtype=torch.int32, device=hidden_states.device)
        cu_seqlens_q = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
                torch.arange(1, batch_size + 1, dtype=torch.int32, device=hidden_states.device) * seq_length,
            ]
        )

        context_lens = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
        )

        with _profile_timer(self.layer_idx, "attention_setup"):
            h_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.bfloat16)

            qa = self._run_projection(self.wq_a, h_flat)
            qa = qa.view(batch_size, seq_length, -1)
            qa = self.q_norm(qa)
            qa_flat = qa.reshape(-1, qa.shape[-1]).to(torch.bfloat16)
            qa_quant, qa_scale = self.q_a_quant(qa_flat)
            q = self.wq_b(qa_quant, qa_scale)
            q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
            q = self.q_b_norm(q)

            kv = self._run_projection(self.wkv, h_flat)
            kv = self.kv_norm(kv)
            kv = kv.view(batch_size, seq_length, self.head_dim)

            cos, sin = position_embeddings
            q = _apply_partial_rotary(q, cos, sin, self.rotary_mul)
            kv = _apply_partial_rotary(
                kv.view(batch_size, seq_length, 1, self.head_dim), cos, sin, self.rotary_mul
            ).view(batch_size, seq_length, self.head_dim)

        if past_key_values is None:
            raise ValueError("Paged Attention requires a PagedDummyCache instance.")

        cmp_sparse_indices = None
        if self.sfa_compressor is not None:
            sfa_state_cache = past_key_values.get_sfa_kv_state(self.layer_idx)
            start_pos = context_lens.to(dtype=torch.int32)
            seq_used_q = q_lens
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
            compressor_hidden_states = hidden_states
            with _profile_timer(self.layer_idx, "compressor"):
                cmp_kv = self.sfa_compressor(
                    compressor_hidden_states,
                    cmp_cos,
                    cmp_sin,
                    state_cache=sfa_state_cache,
                    state_block_table=state_block_table,
                    cu_seqlens=cu_seqlens_q,
                    seq_used_q=seq_used_q,
                    start_pos=start_pos,
                )
            with _profile_timer(self.layer_idx, "cache_update"):
                past_key_values.update_sfa_cmp_kv(cmp_kv, self.layer_idx, cmp_slot_mapping)

            if self.indexer is not None:
                if is_prefill:
                    current_seq_lens = torch.full(
                        (batch_size,), seq_length, dtype=torch.int32, device=hidden_states.device
                    )
                else:
                    current_seq_lens = context_lens.to(dtype=torch.int32) + q_lens
                indexer_hidden_states = hidden_states
                indexer_qa = qa
                indexer_cos = cos
                indexer_sin = sin
                with _profile_timer(self.layer_idx, "indexer"):
                    cmp_sparse_indices = self.indexer.forward(
                        indexer_hidden_states,
                        indexer_qa,
                        indexer_cos,
                        indexer_sin,
                        past_key_values=past_key_values,
                        layer_idx=self.layer_idx,
                        cu_seqlens_q=cu_seqlens_q,
                        seq_lens=current_seq_lens,
                        start_pos=start_pos,
                        state_block_table=state_block_table,
                        seq_used_q=q_lens,
                    )

        if self._is_c1a:
            o = self._c1a_attention(q, kv, past_key_values, context_lens, is_prefill, q_lens)
        else:
            o = self._sparse_attention(q, kv, past_key_values, context_lens, cmp_sparse_indices, is_prefill, q_lens)

        with _profile_timer(self.layer_idx, "attention_post"):
            o = self._attn_post(o, position_embeddings)
        return o, None

    def _run_attn(
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
        has_cmp_kv = compress_ratio > 1
        with _profile_timer(self.layer_idx, "attn_core"):
            o = sparse_attn_shared_kv(
                q=q,
                ori_kv=kv_cache,
                cmp_kv=cmp_kv_cache if has_cmp_kv else None,
                cmp_sparse_indices=cmp_sparse_indices if has_cmp_kv else None,
                cu_seqlens_q=cu_q_lens,
                seqused_kv=seq_lens,
                cmp_block_table=cmp_block_tables if has_cmp_kv and cmp_block_tables is not None else None,
                ori_block_table=block_tables,
                cmp_ratio=compress_ratio,
                ori_win_left=self.sliding_window - 1,
                sinks=self.attn_sink,
                softmax_scale=self.scaling,
            )
        return o.view(batch_size, seq_length, self.num_heads, self.head_dim)

    def _c1a_attention(self, q, kv, past_key_values, context_lens, is_prefill: bool, q_lens: torch.Tensor):
        batch_size, seq_length = q.shape[:2]
        q_padded = q.contiguous().view(-1, self.num_heads, self.head_dim)
        cu_q_lens_padded = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=q.device),
                torch.arange(1, batch_size + 1, dtype=torch.int32, device=q.device) * seq_length,
            ]
        )
        if is_prefill:
            current_seq_lens = torch.full((batch_size,), seq_length, dtype=torch.int32, device=q.device)
        else:
            current_seq_lens = context_lens + q_lens
        if is_prefill:
            with _profile_timer(self.layer_idx, "cache_update"):
                kv_cache, block_tables = past_key_values.build_full_kv_for_prefill(
                    kv, context_lens, cu_q_lens_padded, actual_q_lens=q_lens
                )
                win_slot_mapping = past_key_values.get_win_slot_mapping(context_lens, q_lens, pad_to_window=True)
                full_kv_gather_indices = past_key_values.get_full_kv_gather_indices(context_lens, q_lens)
                past_key_values.update_win_kv(
                    kv, self.layer_idx, win_slot_mapping, full_kv_gather_indices, context_lens
                )
        else:
            win_slot_mapping = past_key_values.get_win_slot_mapping(context_lens, q_lens)
            past_key_values.update_win_kv(kv, self.layer_idx, win_slot_mapping)
            past_key_values.update(kv, self.layer_idx)
            kv_cache, block_tables = past_key_values.get_win_kv_for_decode(self.layer_idx)
        out = self._run_attn(
            q_padded, kv_cache, block_tables, current_seq_lens, batch_size, seq_length, 1, cu_q_lens_padded
        )
        if is_prefill:
            with _profile_timer(self.layer_idx, "cache_update"):
                past_key_values.update(kv, self.layer_idx, cu_q_lens_padded, actual_q_lens=q_lens)
                past_key_values.seq_lens[self.layer_idx] = (context_lens + q_lens).to(past_key_values.seq_lens.dtype)
        return out

    def _sparse_attention(
        self,
        q,
        kv,
        past_key_values,
        context_lens,
        cmp_sparse_indices=None,
        is_prefill: bool = True,
        q_lens: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length = q.shape[:2]
        if q_lens is None:
            q_lens = torch.full((batch_size,), seq_length, dtype=torch.int32, device=q.device)
        q_padded = q.contiguous().view(-1, self.num_heads, self.head_dim)
        cu_q_lens_padded = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=q.device),
                torch.arange(1, batch_size + 1, dtype=torch.int32, device=q.device) * seq_length,
            ]
        )
        if is_prefill:
            current_seq_lens = torch.full((batch_size,), seq_length, dtype=torch.int32, device=q.device)
        else:
            current_seq_lens = context_lens + q_lens
        if is_prefill:
            with _profile_timer(self.layer_idx, "cache_update"):
                kv_cache, block_tables = past_key_values.build_full_kv_for_prefill(
                    kv, context_lens, cu_q_lens_padded, actual_q_lens=q_lens
                )
                win_slot_mapping = past_key_values.get_win_slot_mapping(context_lens, q_lens, pad_to_window=True)
                full_kv_gather_indices = past_key_values.get_full_kv_gather_indices(context_lens, q_lens)
                past_key_values.update_win_kv(
                    kv, self.layer_idx, win_slot_mapping, full_kv_gather_indices, context_lens
                )
        else:
            win_slot_mapping = past_key_values.get_win_slot_mapping(context_lens, q_lens)
            past_key_values.update_win_kv(kv, self.layer_idx, win_slot_mapping)
            past_key_values.update(kv, self.layer_idx)
            kv_cache, block_tables = past_key_values.get_win_kv_for_decode(self.layer_idx)
        cmp_kv_cache = past_key_values.get_sfa_cmp_kv(self.layer_idx)
        cmp_block_tables = past_key_values.get_cmp_kv_block_table(self.layer_idx) if self.compress_ratio > 1 else None
        out = self._run_attn(
            q_padded,
            kv_cache,
            block_tables,
            current_seq_lens,
            batch_size,
            seq_length,
            self.compress_ratio,
            cu_q_lens_padded,
            cmp_kv_cache,
            cmp_block_tables,
            cmp_sparse_indices,
        )
        if is_prefill:
            with _profile_timer(self.layer_idx, "cache_update"):
                past_key_values.update(kv, self.layer_idx, cu_q_lens_padded, actual_q_lens=q_lens)
                past_key_values.seq_lens[self.layer_idx] = (context_lens + q_lens).to(past_key_values.seq_lens.dtype)
        return out

    def _attn_post(self, o, position_embeddings):
        batch_size, seq_length = o.shape[:2]
        cos = position_embeddings[0]
        sin = -position_embeddings[1]

        self.rotary_mul(
            o.flatten(0, 1).unsqueeze(2),
            cos.reshape(-1, 1, 1, self.qk_rope_head_dim),
            sin.reshape(-1, 1, 1, self.qk_rope_head_dim),
        )
        o = o.reshape(batch_size * seq_length, self.o_groups, -1).to(torch.bfloat16)
        wo_a_out = self.wo_a(o)
        wo_a_out = wo_a_out.reshape(batch_size * seq_length, -1).to(torch.bfloat16)
        wo_b_out = self._run_projection(self.wo_b, wo_a_out)
        return wo_b_out.view(batch_size, seq_length, -1)


class DeepseekV4QuantExperts(nn.Module):
    """Rank-local DeepSeek-V4 W8A8 experts."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.up_proj_quantize = MojoMoEDynamicQuant(num_experts, hidden_size)
        self.up_proj = MojoQuantGroupGemm(
            num_groups=num_experts,
            in_features=hidden_size,
            out_features=intermediate_size * 2,
            output_dtype=torch.int32,
        )
        self.dequant_swiglu_quant = MojoDequantSwiGLUClampQuant(
            expert_num=num_experts, hidden_size=intermediate_size, clamp_limit=swiglu_limit
        )
        self.down_proj = MojoQuantGroupGemm(
            num_groups=num_experts, in_features=intermediate_size, out_features=hidden_size
        )
        nn.init.ones_(self.up_proj_quantize.inv_smooth_scale)
        nn.init.ones_(self.dequant_swiglu_quant.quant_scale)
        self._weights_prepared = False

    def prepare_for_inference(self) -> None:
        if self._weights_prepared or self.up_proj.weight.device.type != "npu":
            return
        self.up_proj.weight = _to_nz(self.up_proj.weight)
        self.down_proj.weight = _to_nz(self.down_proj.weight)
        self.down_proj.weight_scale = self.down_proj.weight_scale.to(torch.bfloat16)
        self._weights_prepared = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        self.prepare_for_inference()
        tokens_per_expert = tokens_per_expert.to(torch.int64)
        hidden_int8, hidden_scale = self.up_proj_quantize(hidden_states, tokens_per_expert)
        accumulator = self.up_proj(hidden_int8, tokens_per_expert)
        intermediate_int8, intermediate_scale = self.dequant_swiglu_quant(
            accumulator, hidden_scale, tokens_per_expert
        )
        return self.down_proj(intermediate_int8, tokens_per_expert, intermediate_scale)


class DeepseekV4SharedExpert(DeepseekV4QuantExperts):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__(
            num_experts=1,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            swiglu_limit=config.swiglu_limit,
        )
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size).to(torch.bfloat16)
        self.prepare_for_inference()
        tokens_per_expert = torch.tensor([hidden_states.shape[0]], dtype=torch.int64, device=hidden_states.device)
        with _profile_timer(self.layer_idx, "shared_expert_gate_up"):
            hidden_int8, hidden_scale = self.up_proj_quantize(hidden_states, tokens_per_expert)
            accumulator = self.up_proj(hidden_int8, tokens_per_expert)
        with _profile_timer(self.layer_idx, "shared_expert_swiglu"):
            intermediate_int8, intermediate_scale = self.dequant_swiglu_quant(
                accumulator, hidden_scale, tokens_per_expert
            )
        with _profile_timer(self.layer_idx, "shared_expert_down"):
            output = self.down_proj(intermediate_int8, tokens_per_expert, intermediate_scale)
        return output.view(*original_shape)


class DeepseekV4MoE(nn.Module):
    """DeepSeek-V4 routed and shared experts with correctness-first EP.

    Every rank evaluates the global router, computes only its local expert slice,
    scatters partial outputs into token order, and then sums those partials over
    the EP group.  This mirrors mojo_opset's generic expert-parallel contract
    without carrying the prototype's MC2/all-to-all performance stack.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int, ep_size: int = 1, ep_rank: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = int(config.n_shared_experts or 0)
        self.norm_topk_prob = config.norm_topk_prob
        if self.n_shared_experts != 1:
            raise NotImplementedError(
                f"DeepSeek-V4 V1 requires n_shared_experts=1, got {self.n_shared_experts}."
            )
        if not self.norm_topk_prob:
            raise NotImplementedError("DeepSeek-V4 V1 requires norm_topk_prob=true.")
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        scoring_func_mapping = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}
        if config.scoring_func not in scoring_func_mapping:
            raise NotImplementedError(f"Unsupported scoring_func: {config.scoring_func}")
        if config.topk_method != "noaux_tc":
            raise NotImplementedError(
                f"DeepSeek-V4 V1 only supports topk_method='noaux_tc', got {config.topk_method!r}."
            )
        self.norm_type = scoring_func_mapping[config.scoring_func]
        self.swiglu_limit = config.swiglu_limit
        self.is_hash = layer_idx < config.num_hash_layers

        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_per_rank = self.n_routed_experts // self.ep_size
        self.ep_start = self.ep_rank * self.experts_per_rank
        self.ep_end = self.ep_start + self.experts_per_rank
        self.ep_group = None

        self.dispatch = MojoMoEDispatch(num_experts=self.n_routed_experts)
        self.experts = DeepseekV4QuantExperts(
            num_experts=self.experts_per_rank,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            swiglu_limit=self.swiglu_limit,
        )
        self.combine = MojoMoECombine(multiply_by_gates=True)

        self.shared_experts = DeepseekV4SharedExpert(config, layer_idx)
        self.moe_gating_top_k = MojoMoEGatingTopK(
            num_experts=self.n_routed_experts,
            top_k=self.top_k,
            norm_type=self.norm_type,
            routed_scaling_factor=self.routed_scaling_factor,
            bias=not self.is_hash,
        )
        self.gate = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size, dtype=torch.float32))
        if self.is_hash:
            self.tid2eid = nn.Parameter(
                torch.randint(
                    high=config.n_routed_experts,
                    size=(config.vocab_size, self.top_k),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
        else:
            self.tid2eid = None

    def _gate_topk(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor]):
        if self.is_hash and input_ids is None:
            raise ValueError("input_ids is required for DeepSeek-V4 hash-layer routing.")
        if self.is_hash:
            if self.norm_type == 0:
                router_scores = torch.softmax(logits.float(), dim=-1)
            elif self.norm_type == 1:
                router_scores = torch.sigmoid(logits.float())
            else:
                router_scores = torch.sqrt(F.softplus(logits.float()))
            topk_idx = self.tid2eid.index_select(0, input_ids.to(dtype=torch.int64)).to(dtype=torch.int32)
            topk_weight = torch.gather(router_scores, 1, topk_idx.to(dtype=torch.int64))
            if self.norm_type != 0:
                topk_weight /= topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = (topk_weight * self.routed_scaling_factor).to(dtype=logits.dtype)
            return topk_idx, topk_weight

        topk_weight, topk_idx, _ = self.moe_gating_top_k(logits)
        return topk_idx.to(torch.int32), topk_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        del is_prefill
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size).to(torch.bfloat16)
        flat_ids = input_ids.reshape(-1).contiguous() if input_ids is not None else None

        logits = F.linear(flat.float(), self.gate)
        topk_idx, topk_weight = self._gate_topk(logits, flat_ids)
        sorted_hidden, tokens_per_expert, sorted_gates, token_indices = self.dispatch(flat, topk_weight, topk_idx)

        if self.ep_size > 1:
            if self.ep_group is None:
                raise RuntimeError("EP group has not been initialized; call init_parallel_comm_group/set_ep_group.")
            cumsum = tokens_per_expert.cumsum(0)
            token_start = 0 if self.ep_start == 0 else int(cumsum[self.ep_start - 1].item())
            token_end = int(cumsum[self.ep_end - 1].item())
            sorted_hidden = sorted_hidden[token_start:token_end]
            sorted_gates = sorted_gates[token_start:token_end]
            token_indices = token_indices[token_start:token_end]
            tokens_per_expert = tokens_per_expert[self.ep_start : self.ep_end]

        expert_outputs = self.experts(sorted_hidden, tokens_per_expert)
        output_buffer = torch.zeros_like(flat, memory_format=torch.contiguous_format)
        routed = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        if self.ep_size > 1:
            dist.all_reduce(routed, op=dist.ReduceOp.SUM, group=self.ep_group)

        if self.n_shared_experts:
            routed = routed + self.shared_experts(flat)
        return routed.view(*original_shape)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int, ep_size: int = 1, ep_rank: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult

        self.self_attn = DeepseekV4Attention(config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV4MoE(config, layer_idx, ep_size=ep_size, ep_rank=ep_rank)

        self.attn_norm = MojoRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)
        self.ffn_norm = MojoRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=torch.bfloat16)
        self.hc_attn = MojoHcPre(
            hidden_size=config.hidden_size,
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            norm_eps=config.rms_norm_eps,
            hc_eps=config.hc_eps,
        )
        self.hc_ffn = MojoHcPre(
            hidden_size=config.hidden_size,
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            norm_eps=config.rms_norm_eps,
            hc_eps=config.hc_eps,
        )
        self.hc_post = MojoHcPost()

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
        if _DSV4_LAYER_PROFILE:
            _profile_sync()
            layer_start = time.perf_counter()
        residual = hidden_states
        hidden_states, post, comb = self.hc_attn(hidden_states)
        hidden_states = self.attn_norm(hidden_states)
        with _profile_timer(self.layer_idx, "attention_total"):
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                is_prefill=is_prefill,
                **kwargs,
            )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = self.hc_ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states)
        with _profile_timer(self.layer_idx, "moe"):
            hidden_states = self.mlp(hidden_states, input_ids=input_ids, is_prefill=is_prefill)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        if _DSV4_LAYER_PROFILE:
            _profile_sync()
            _profile_record(self.layer_idx, "layer_total", (time.perf_counter() - layer_start) * 1000.0)
        return hidden_states


class DeepseekV4Model(nn.Module):
    def __init__(self, config: DeepseekV4Config, ep_size: int = 1, ep_rank: int = 0):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(config, layer_idx, ep_size=ep_size, ep_rank=ep_rank)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
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

        if attention_mask is not None and is_prefill:
            q_lens = attention_mask.to(device=device, dtype=torch.int32).sum(dim=-1)
            position_ids = (attention_mask.to(device=device, dtype=torch.long).cumsum(dim=-1) - 1).clamp(min=0)
            position_ids = position_ids.masked_fill(~attention_mask.to(dtype=torch.bool), 1)
        else:
            q_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
            past_lens = past_key_values.get_seq_length(0).to(device=device, dtype=torch.long)
            offsets = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = past_lens.unsqueeze(1) + offsets

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)
        cmp_cos, cmp_sin = self.compress_rotary_emb(hidden_states, position_ids)
        compress_position_embeddings = (cmp_cos, cmp_sin)

        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_position_embeddings = (
                compress_position_embeddings if self.config.compress_ratios[layer_idx] > 1 else position_embeddings
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=layer_position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                input_ids=input_ids,
                is_prefill=is_prefill,
                q_lens=q_lens,
                **kwargs,
            )

        hidden_states = self._hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(self, config, num_layers=None, ep_size=1, ep_rank=0):
        super().__init__()
        if ep_size < 1:
            raise ValueError(f"ep_size must be positive, got {ep_size}.")
        if not 0 <= ep_rank < ep_size:
            raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}.")
        if not isinstance(config, DeepseekV4Config):
            config = DeepseekV4Config._from_hf_config(config)
        if num_layers is not None and num_layers < config.num_hidden_layers:
            config.num_hidden_layers = num_layers
            config.compress_ratios = config.compress_ratios[:num_layers]
        if config.n_routed_experts % ep_size:
            raise ValueError(f"n_routed_experts={config.n_routed_experts} must be divisible by ep_size={ep_size}.")
        self.config = config
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.model = DeepseekV4Model(config, ep_size=ep_size, ep_rank=ep_rank)
        self.lm_head = MojoGemm(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )
        self.moe_ep_group = None

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
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            is_prefill=is_prefill,
            **kwargs,
        )
        if is_prefill and attention_mask is not None:
            q_lens = attention_mask.to(dtype=torch.int32).sum(dim=-1)
            gather_index = (q_lens - 1).view(-1, 1, 1).expand(-1, 1, hidden_states.shape[-1])
            hidden_states = torch.gather(hidden_states, 1, gather_index)
        hidden_states_flat = hidden_states.reshape(-1, self.config.hidden_size).to(torch.bfloat16)
        logits = self.lm_head(hidden_states_flat)
        return logits.view(*hidden_states.shape[:-1], self.config.vocab_size).float(), past_key_values

    def init_parallel_comm_group(self):
        if self.ep_size == 1:
            self.moe_ep_group = None
            return
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before EP groups are created.")
        world_size = dist.get_world_size()
        if world_size != self.ep_size:
            raise ValueError(f"V1 requires world_size == ep_size, got {world_size} and {self.ep_size}.")
        rank = dist.get_rank()
        if rank != self.ep_rank:
            raise ValueError(f"Distributed rank {rank} does not match configured ep_rank {self.ep_rank}.")
        self.moe_ep_group = dist.group.WORLD

    def set_ep_group(self):
        for layer in self.model.layers:
            layer.mlp.ep_group = self.moe_ep_group

    @staticmethod
    def load_weights(model, weight_dir: str):
        from .loader import load_weights

        return load_weights(model, weight_dir)
