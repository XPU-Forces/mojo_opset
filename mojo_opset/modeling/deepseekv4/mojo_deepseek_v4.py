import math
import json
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mojo_opset import MojoGemm
from mojo_opset import MojoQuantGemm
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoDynamicQuant
from mojo_opset import MojoStorePagedKVCache
from mojo_opset import MojoMoEDispatch
from mojo_opset import MojoQuantExperts
from mojo_opset import MojoMoECombine


_HAS_INPLACE_PARTIAL_ROTARY = hasattr(torch.ops.custom, "inplace_partial_rotary_mul")
_HAS_NPU_SPARSE_ATTN_SHAREDKV = hasattr(torch.ops.custom, "npu_sparse_attn_sharedkv")
_HAS_NPU_HC_PRE = hasattr(torch.ops.custom, "npu_hc_pre")
_HAS_NPU_HC_POST = hasattr(torch.ops.custom, "npu_hc_post")
_HAS_COMPRESSOR = hasattr(torch.ops.custom, "compressor")
_HAS_QUANT_LIGHTNING_INDEXER = hasattr(torch.ops.custom, "npu_quant_lightning_indexer")
_HAS_NPU_MOE_GATING_TOP_K = hasattr(torch.ops.custom, "npu_moe_gating_top_k")


def _fallback_partial_rotary(x, cos, sin, partial_slice):
    if isinstance(partial_slice, list):
        rope_dim_start = partial_slice[0]
        rope_dim_end = partial_slice[1]
    else:
        rope_dim_start = x.shape[-1] - partial_slice
        rope_dim_end = x.shape[-1]

    head_dim = x.shape[-1]
    rope_dim = rope_dim_end - rope_dim_start
    half_rope = rope_dim // 2

    x1 = x[..., :rope_dim_start]
    x2 = x[..., rope_dim_start:rope_dim_end]
    x3 = x[..., rope_dim_end:]

    while cos.dim() < x2.dim():
        cos = cos.unsqueeze(-2)
    while sin.dim() < x2.dim():
        sin = sin.unsqueeze(-2)

    cos_slice = cos[..., :half_rope]
    sin_slice = sin[..., :half_rope]

    x2_even = x2[..., 0::2]
    x2_odd = x2[..., 1::2]
    rotated_even = x2_even * cos_slice - x2_odd * sin_slice
    rotated_odd = x2_even * sin_slice + x2_odd * cos_slice

    rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
    parts = [x1, rotated]
    if x3.numel() > 0 and x3.shape[-1] > 0:
        parts.append(x3)
    return torch.cat(parts, dim=-1)


def _apply_partial_rotary(x, cos, sin, partial_slice):
    if _HAS_INPLACE_PARTIAL_ROTARY:
        try:
            torch.ops.custom.inplace_partial_rotary_mul(
                x, cos, sin,
                rotary_mode="interleave",
                partial_slice=partial_slice,
            )
            return x
        except RuntimeError:
            pass
    return _fallback_partial_rotary(x, cos, sin, partial_slice)


def _fallback_sparse_attn_sharedkv(q, kv_cache, block_tables, seq_lens, scaling, attn_sink=None,
                                    cmp_kv_cache=None, cmp_block_tables=None, compress_ratio=1,
                                    cmp_sparse_indices=None):
    head_dim = q.shape[-1]
    num_heads = q.shape[-2] if q.dim() >= 2 else 1
    batch_size = seq_lens.shape[0]
    is_tnd = q.dim() == 3

    all_outputs = []
    token_offset = 0
    for b in range(batch_size):
        slen = int(seq_lens[b].item())

        if is_tnd:
            q_len = max(1, q.shape[0] // batch_size)
        else:
            q_len = 1

        if slen <= 0:
            all_outputs.append(torch.zeros(q_len, num_heads, head_dim, device=q.device, dtype=q.dtype))
            if is_tnd:
                token_offset += q_len
            continue

        if block_tables.dim() == 2:
            kv_blocks = block_tables[b]
            valid_blocks = kv_blocks[kv_blocks >= 0]
            if valid_blocks.numel() > 0:
                k_all = kv_cache[valid_blocks]
                if k_all.dim() == 4 and k_all.shape[2] == 1:
                    k_all = k_all.squeeze(2)
                k_all = k_all.reshape(-1, head_dim)[:slen]
            else:
                k_all = torch.zeros(slen, head_dim, device=q.device, dtype=kv_cache.dtype)
        else:
            k_all = kv_cache.view(-1, head_dim)[:slen]

        if is_tnd:
            q_b = q[token_offset:token_offset + q_len]
            token_offset += q_len
        else:
            q_b = q[b:b + 1]

        k_b = k_all.unsqueeze(0).unsqueeze(0).expand(q_len, num_heads, -1, -1)
        v_b = k_b.clone()

        scores = torch.matmul(q_b.float().unsqueeze(2), k_b.transpose(-2, -1).float()) * scaling

        if attn_sink is not None:
            scores[:, :, :, :1] = scores[:, :, :, :1] + attn_sink.view(1, num_heads, 1, 1).float()

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_b.float()).squeeze(2)
        all_outputs.append(attn_out.to(q.dtype))

    return torch.cat(all_outputs, dim=0)


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
            "sliding_window", "compress_ratios", "hc_mult", "hc_sinkhorn_iters",
            "hc_eps", "index_n_heads", "index_head_dim", "index_topk",
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
                 block_size: int = 128, max_seq_len: int = 4096):
        self.num_layers = config.num_hidden_layers
        self.device = device
        self.block_size = block_size
        self.batch_size = batch_size
        self.head_dim = config.head_dim
        self.config = config
        self.sliding_window = config.sliding_window
        self.index_head_dim = config.index_head_dim

        max_blocks_per_seq = (max_seq_len + self.block_size - 1) // self.block_size
        total_blocks = self.batch_size * max_blocks_per_seq * self.num_layers

        self.kv_cache = torch.zeros(
            (total_blocks, 1, self.block_size, self.head_dim),
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
        self.store_paged_kv = MojoStorePagedKVCache()

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
            }
            win_block_num = self._get_block_num(self.sliding_window)
            cache_dict["win_kv"] = self._create_cache(win_block_num, self.head_dim, torch.bfloat16)

            if ratio == 4:
                cmp_block_num = self._get_block_num(max_seq_len // ratio)
                overlap_num = 2
                state_block_num = self._get_block_num((1 + overlap_num) * ratio)
                cache_dict["sfa_cmp_kv"] = self._create_cache(cmp_block_num, self.head_dim, torch.bfloat16)
                cache_dict["sfa_kv_state"] = self._create_state_cache(state_block_num, ratio, self.head_dim)
                cache_dict["li_cmp_kv"] = self._create_cache(cmp_block_num, self.index_head_dim, torch.bfloat16)
                cache_dict["li_kv_state"] = self._create_state_cache(state_block_num, ratio, self.index_head_dim)
            elif ratio == 128:
                cmp_block_num = self._get_block_num(max_seq_len // ratio)
                overlap_num = 1
                state_block_num = self._get_block_num(overlap_num * ratio)
                cache_dict["sfa_cmp_kv"] = self._create_cache(cmp_block_num, self.head_dim, torch.bfloat16)
                cache_dict["sfa_kv_state"] = self._create_state_cache(state_block_num, ratio, self.head_dim)

            self.cache_data[layer_idx] = cache_dict

    def _get_block_num(self, cache_size):
        return math.ceil(cache_size / self.block_size) * self.batch_size + 1

    def _create_cache(self, block_num, dim, dtype):
        return torch.zeros(
            (block_num, self.block_size, 1, dim),
            dtype=dtype, device=self.device,
        )

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

        kv_flat = kv.reshape(-1, 1, self.head_dim)
        self.store_paged_kv(
            kv_flat, kv_flat, self.kv_cache, self.kv_cache,
            self.block_tables[layer_idx], cu_q_lens, current_seq_lens,
        )
        self.seq_lens[layer_idx] += new_seq_len

    def update_win_kv(self, kv: torch.Tensor, layer_idx: int, slot_mapping: Optional[torch.Tensor] = None) -> None:
        win_cache = self.cache_data[layer_idx]["win_kv"]
        if win_cache is None:
            return
        batch_size, seq_len, _ = kv.shape
        kv_flat = kv.reshape(-1, self.head_dim)
        win_flat = win_cache.view(-1, self.head_dim)
        if slot_mapping is not None:
            sm = slot_mapping.reshape(-1, 1).expand(-1, self.head_dim)
            win_flat.scatter_(0, sm, kv_flat)
        else:
            context_len = int(self.seq_lens[layer_idx][0].item())
            for t in range(seq_len):
                pos = context_len + t
                block_idx = pos // self.block_size + 1
                offset = pos % self.block_size
                if block_idx < win_cache.shape[0]:
                    win_cache[block_idx, offset, 0, :] = kv_flat[t]

    def update_sfa_cmp_kv(self, kv: torch.Tensor, layer_idx: int, slot_mapping: Optional[torch.Tensor] = None) -> None:
        sfa_cmp_cache = self.cache_data[layer_idx]["sfa_cmp_kv"]
        if sfa_cmp_cache is None:
            return
        batch_size, seq_len, _ = kv.shape
        kv_flat = kv.reshape(-1, self.head_dim)
        cmp_flat = sfa_cmp_cache.view(-1, self.head_dim)
        if slot_mapping is not None:
            sm = slot_mapping.reshape(-1, 1).expand(-1, self.head_dim)
            cmp_flat.scatter_(0, sm, kv_flat)
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
        if li_cmp_cache is None:
            return
        batch_size, seq_len, _ = kv.shape
        kv_flat = kv.reshape(-1, self.index_head_dim)
        cmp_flat = li_cmp_cache.view(-1, self.index_head_dim)
        if slot_mapping is not None:
            sm = slot_mapping.reshape(-1, 1).expand(-1, self.index_head_dim)
            cmp_flat.scatter_(0, sm, kv_flat)
        else:
            ratio = self.config.compress_ratios[layer_idx]
            cmp_context_len = int(self.seq_lens[layer_idx][0].item()) // ratio
            for t in range(seq_len):
                pos = cmp_context_len + t
                block_idx = pos // self.block_size + 1
                offset = pos % self.block_size
                if block_idx < li_cmp_cache.shape[0]:
                    li_cmp_cache[block_idx, offset, 0, :] = kv_flat[t]

    def get_kv_for_decode(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_slen = self.seq_lens[layer_idx].max().item()
        max_blocks = (max_slen + self.block_size - 1) // self.block_size
        return self.kv_cache, self.block_tables[layer_idx, :, :max_blocks]

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

    def __init__(self, config: DeepseekV4Config, device: Optional[str] = None):
        super().__init__()
        dim = config.qk_rope_head_dim
        base = config.rope_theta
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
            emb = torch.cat((freqs, freqs), dim=-1)
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

        self.wkv = MojoGemm(in_features=self.hidden_size, out_features=self.coff * self.head_dim, bias=False)
        self.wgate = MojoGemm(in_features=self.hidden_size, out_features=self.coff * self.head_dim, bias=False)
        self.norm = MojoRMSNorm(norm_size=self.head_dim, eps=config.rms_norm_eps)
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * self.head_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                state_cache: Optional[torch.Tensor] = None,
                start_pos: int = 0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_bf16 = x.to(torch.bfloat16)

        kv = self.wkv(x_bf16.reshape(-1, self.hidden_size)).view(batch_size, seq_len, self.coff * self.head_dim)
        gate = torch.sigmoid(self.wgate(x_bf16.reshape(-1, self.hidden_size)).view(batch_size, seq_len, self.coff * self.head_dim))
        kv = kv * gate

        compressed_len = (seq_len + self.compress_ratio - 1) // self.compress_ratio
        kv_compressed = torch.zeros(batch_size, compressed_len, self.head_dim, device=x.device, dtype=torch.bfloat16)

        ape = self.ape.float()

        for i in range(compressed_len):
            start = i * self.compress_ratio
            end = min(start + self.compress_ratio, seq_len)
            chunk_len = end - start

            if self.overlap and i > 0:
                overlap_start = max(0, start - self.compress_ratio)
                overlap_end = min(overlap_start + self.compress_ratio, seq_len)
                main_part = kv[:, start:end, :self.head_dim]
                overlap_part = kv[:, overlap_start:overlap_end, self.head_dim:]
                combined = torch.cat([main_part, overlap_part], dim=1)
                ape_main = ape[:chunk_len, :self.head_dim]
                ape_overlap = ape[:overlap_end - overlap_start, self.head_dim:]
                ape_combined = torch.cat([ape_main, ape_overlap], dim=0)
                ape_weights = torch.softmax(ape_combined, dim=0)
                kv_compressed[:, i, :] = (combined.float() * ape_weights.unsqueeze(0)).sum(dim=1).to(torch.bfloat16)
            else:
                ape_weights = torch.softmax(ape[:chunk_len, :self.head_dim], dim=0)
                kv_compressed[:, i, :] = (kv[:, start:end, :self.head_dim].float() * ape_weights.unsqueeze(0)).sum(dim=1).to(torch.bfloat16)

        kv_compressed = self.norm(kv_compressed)

        partial_slice = [self.head_dim - self.rope_head_dim, self.head_dim]
        cos_cmp = cos[:, :compressed_len, :]
        sin_cmp = sin[:, :compressed_len, :]
        kv_compressed = _apply_partial_rotary(
            kv_compressed.unsqueeze(2), cos_cmp, sin_cmp, partial_slice
        ).squeeze(2)

        if state_cache is not None:
            self._update_state(state_cache, kv_compressed, start_pos)

        return kv_compressed

    def _update_state(self, state_cache: torch.Tensor, kv_compressed: torch.Tensor, start_pos: int):
        batch_size, cmp_len, _ = kv_compressed.shape
        ratio = self.compress_ratio
        overlap_num = 2 if ratio == 4 else 1
        cmp_start_pos = start_pos // ratio

        for b in range(batch_size):
            for t in range(cmp_len):
                global_pos = cmp_start_pos + t
                block_idx = global_pos // state_cache.shape[1] + 1
                offset = global_pos % state_cache.shape[1]
                if block_idx < state_cache.shape[0]:
                    for ov in range(overlap_num):
                        state_cache[block_idx, offset, 0, ov, :] = kv_compressed[b, t, :].float()
                    if overlap_num > 1 and t > 0:
                        state_cache[block_idx, offset, 1, :, :] = kv_compressed[b, t - 1, :].float()


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

    def forward(self, x, qr, cos, sin, past_key_values=None, layer_idx=0,
                cu_seqlens_q=None, seq_lens=None):
        batch_size, seq_len, _ = x.shape

        weights = self.weights_proj(x.to(torch.bfloat16).reshape(-1, self.hidden_size))
        weights = weights.view(batch_size, seq_len, self.n_heads) * (self.softmax_scale * self.n_heads ** -0.5)

        li_state_cache = None
        if past_key_values is not None:
            li_state_cache = past_key_values.get_li_kv_state(layer_idx)

        cmp_cos = cos[:, ::self.compress_ratio, :]
        cmp_sin = sin[:, ::self.compress_ratio, :]
        li_kv = self.compressor(x, cmp_cos, cmp_sin, state_cache=li_state_cache)

        if past_key_values is not None:
            past_key_values.update_li_cmp_kv(li_kv, layer_idx)

        qr_flat = qr.reshape(-1, self.q_lora_rank).to(torch.bfloat16)
        qr_quant, qr_scale = _dynamic_quant_per_token(qr_flat)
        q = self.wq_b(qr_quant, qr_scale)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = _apply_partial_rotary(q, cos, sin, self.partial_slice)

        if _HAS_QUANT_LIGHTNING_INDEXER and past_key_values is not None:
            try:
                import torch_npu
                li_cmp_kv = past_key_values.get_li_cmp_kv(layer_idx)

                q_flat = q.flatten(0, 1)
                q_quant, q_scale = torch_npu.npu_dynamic_quant(q_flat)
                q_scale = q_scale.to(torch.float16)

                actual_seq_q = cu_seqlens_q[1:] if cu_seqlens_q is not None else torch.tensor([seq_len], dtype=torch.int32, device=x.device)
                actual_seq_k = seq_lens if seq_lens is not None else torch.tensor([seq_len], dtype=torch.int32, device=x.device)

                topk_idxs, _ = torch.ops.custom.npu_quant_lightning_indexer(
                    query=q_quant, key=li_cmp_kv, weights=weights.flatten(0, 1).to(torch.float16),
                    query_dequant_scale=q_scale,
                    key_dequant_scale=torch.ones(1, dtype=torch.float16, device=x.device),
                    actual_seq_lengths_query=actual_seq_q,
                    actual_seq_lengths_key=actual_seq_k,
                    block_table=None, layout_key='PA_BSND',
                    sparse_count=self.index_topk, sparse_mode=3,
                    layout_query="TND", cmp_ratio=self.compress_ratio,
                    key_quant_mode=0, query_quant_mode=0,
                )
                return topk_idxs.view(q_flat.shape[0], -1, self.index_topk)
            except RuntimeError:
                pass

        return self._fallback_indexer(q, li_kv, weights, batch_size, seq_len)

    def _fallback_indexer(self, q, li_cmp_kv, weights, batch_size, seq_len):
        return None


def _dynamic_quant_per_token(x: torch.Tensor):
    x_fp = x.float()
    scale = x_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    scale = torch.where(scale < 1e-6, 1.0, scale)
    output = torch.clamp(torch.round(x_fp / scale), -128, 127)
    return output.to(torch.int8), scale.squeeze(-1)


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

        self.wkv = MojoGemm(in_features=config.hidden_size, out_features=self.head_dim, bias=False)
        self.kv_norm = MojoRMSNorm(eps=config.rms_norm_eps, norm_size=self.head_dim)

        self.wo_a = MojoGemm(in_features=self.num_heads * self.head_dim // self.o_groups, out_features=self.o_groups * self.o_lora_rank, bias=False)
        self.wo_b = MojoQuantGemm(in_features=self.o_groups * self.o_lora_rank, out_features=config.hidden_size, trans_weight=True)

        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))

        if raw_ratio > 1:
            self.sfa_compressor = DeepseekV4Compressor(config, raw_ratio, head_dim=self.head_dim)
            self.indexer = DeepseekV4Indexer(config, raw_ratio) if raw_ratio == 4 else None
        else:
            self.sfa_compressor = None
            self.indexer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: bool = True,
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
            cmp_cos = cos[:, ::self.compress_ratio, :]
            cmp_sin = sin[:, ::self.compress_ratio, :]

            sfa_state_cache = past_key_values.get_sfa_kv_state(self.layer_idx)
            start_pos = int(context_lens[0].item()) if context_lens.numel() > 0 else 0
            cmp_kv = self.sfa_compressor(hidden_states, cmp_cos, cmp_sin,
                                         state_cache=sfa_state_cache, start_pos=start_pos)
            past_key_values.update_sfa_cmp_kv(cmp_kv, self.layer_idx)

            if self.indexer is not None:
                cmp_sparse_indices = self.indexer.forward(
                    hidden_states, qa, cmp_cos, cmp_sin,
                    past_key_values=past_key_values, layer_idx=self.layer_idx,
                )

        if self._is_c1a:
            o = self._c1a_attention(q, kv, past_key_values, context_lens)
        else:
            o = self._sparse_attention(q, kv, past_key_values, context_lens, cmp_sparse_indices)

        o = self._attn_post(o, position_embeddings)
        return o, None

    def _run_attn(self, q, kv_cache, block_tables, seq_lens, batch_size, seq_length,
                  compress_ratio, cu_q_lens=None, cmp_kv_cache=None, cmp_block_tables=None,
                  cmp_sparse_indices=None):
        if _HAS_NPU_SPARSE_ATTN_SHAREDKV:
            try:
                o = torch.ops.custom.npu_sparse_attn_sharedkv(
                    q=q, ori_kv=kv_cache,
                    cmp_kv=cmp_kv_cache if cmp_kv_cache is not None else kv_cache,
                    cmp_sparse_indices=cmp_sparse_indices,
                    cu_seqlens_q=cu_q_lens, seqused_kv=seq_lens,
                    cmp_block_table=cmp_block_tables if cmp_block_tables is not None and compress_ratio > 1 else None,
                    ori_block_table=block_tables, cmp_ratio=compress_ratio,
                    ori_mask_mode=4, cmp_mask_mode=3,
                    ori_win_left=self.sliding_window - 1, ori_win_right=0,
                    layout_q="TND", layout_kv="PA_ND",
                    sinks=self.attn_sink, metadata=None,
                    softmax_scale=self.scaling,
                )[0]
                return o.view(batch_size, seq_length, self.num_heads, self.head_dim)
            except RuntimeError:
                pass
        o = _fallback_sparse_attn_sharedkv(
            q, kv_cache, block_tables, seq_lens, self.scaling, self.attn_sink,
            cmp_kv_cache=cmp_kv_cache, cmp_block_tables=cmp_block_tables,
            compress_ratio=compress_ratio, cmp_sparse_indices=cmp_sparse_indices,
        )
        return o.view(batch_size, seq_length, self.num_heads, self.head_dim)

    def _c1a_attention(self, q, kv, past_key_values, context_lens):
        batch_size, seq_length = q.shape[:2]
        past_key_values.update(kv, self.layer_idx)
        kv_cache, block_tables = past_key_values.get_kv_for_decode(self.layer_idx)
        q_tnd = q.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)
        current_seq_lens = context_lens + seq_length
        q_lens = torch.full((batch_size,), seq_length, dtype=torch.int32, device=q.device)
        cu_q_lens = torch.cat([torch.tensor([0], device=q.device, dtype=torch.int32), q_lens.cumsum(0, dtype=torch.int32)])
        return self._run_attn(q_tnd, kv_cache, block_tables, current_seq_lens, batch_size, seq_length, 1, cu_q_lens)

    def _sparse_attention(self, q, kv, past_key_values, context_lens, cmp_sparse_indices=None):
        batch_size, seq_length = q.shape[:2]
        past_key_values.update(kv, self.layer_idx)
        kv_cache, block_tables = past_key_values.get_kv_for_decode(self.layer_idx)
        cmp_kv_cache = past_key_values.get_sfa_cmp_kv(self.layer_idx)
        q_tnd = q.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)
        current_seq_lens = context_lens + seq_length
        q_lens = torch.full((batch_size,), seq_length, dtype=torch.int32, device=q.device)
        cu_q_lens = torch.cat([torch.tensor([0], device=q.device, dtype=torch.int32), q_lens.cumsum(0, dtype=torch.int32)])
        return self._run_attn(q_tnd, kv_cache, block_tables, current_seq_lens, batch_size, seq_length,
                              self.compress_ratio, cu_q_lens, cmp_kv_cache, None, cmp_sparse_indices)

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
        x_gate, gate_scale = self.gate_quant(x_flat)
        x_up, up_scale = self.up_quant(x_flat)
        gate_out = self.gate_proj(x_gate, gate_scale)
        up_out = self.up_proj(x_up, up_scale)
        gate_act = F.silu(gate_out.float())
        intermediate = gate_act * up_out.float()
        if self.swiglu_limit is not None and self.swiglu_limit > 0:
            intermediate = intermediate.clamp(-self.swiglu_limit, self.swiglu_limit)
        intermediate = intermediate.to(torch.bfloat16)
        intermediate_quant, intermediate_scale = self.intermediate_quant(intermediate)
        return self.down_proj(intermediate_quant, intermediate_scale).view(*orig_shape)


class DeepseekV4MoE(nn.Module):

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
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

        self.dispatch = MojoMoEDispatch(num_experts=config.n_routed_experts)
        self.experts = MojoQuantExperts(
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            weight_dtype=torch.int8,
        )
        self.combine = MojoMoECombine(multiply_by_gates=True)
        nn.init.ones_(self.experts.up_proj_quantize.inv_smooth_scale)
        nn.init.ones_(self.experts.down_proj_quantize.inv_smooth_scale)

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
            if _HAS_NPU_MOE_GATING_TOP_K:
                scoring_func_mapping = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}
                try:
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
                except RuntimeError:
                    pass

            if not self.is_hash:
                tmp_scores = scores + self.e_score_correction_bias.unsqueeze(0)
                _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            else:
                hash_idx = self.tid2eid[input_ids]
                topk_idx = hash_idx.view(scores.shape[0], -1)

            topk_weight = scores.gather(1, topk_idx)
        elif self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        else:
            raise NotImplementedError(f"Unsupported topk_method: {self.topk_method}")

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True)
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx.to(torch.int32), topk_weight

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, self.hidden_size).to(torch.bfloat16)

        logits = F.linear(hidden_states_flat.float(), self.gate)
        topk_idx, topk_weight = self._gate_topk(logits, input_ids)

        sorted_hidden, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states_flat, topk_weight, topk_idx
        )
        expert_outputs = self.experts(sorted_hidden, tokens_per_expert)
        output_buffer = torch.zeros_like(hidden_states_flat, memory_format=torch.contiguous_format)
        routed_out = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)

        shared_out = self.shared_experts(residuals)
        return routed_out.view(*orig_shape) + shared_out


class OpKernel:

    @staticmethod
    def hc_pre(hidden_states, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, norm_eps, hc_eps):
        if _HAS_NPU_HC_PRE:
            try:
                y, post, comb = torch.ops.custom.npu_hc_pre(
                    hidden_states,
                    hc_fn.float() if hc_fn.dtype != torch.float32 else hc_fn,
                    hc_scale.float() if hc_scale.dtype != torch.float32 else hc_scale,
                    hc_base.float() if hc_base.dtype != torch.float32 else hc_base,
                    hc_mult=hc_mult, hc_sinkhorn_iters=sinkhorn_iters,
                    norm_eps=norm_eps, hc_eps=hc_eps,
                )
                return y, post, comb
            except RuntimeError:
                pass
        return OpKernel._hc_pre_native(hidden_states, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, norm_eps, hc_eps)

    @staticmethod
    def _hc_pre_native(hidden_states, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, norm_eps, hc_eps):
        shape = hidden_states.shape
        x = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt

        pre, post, comb = OpKernel._hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        y = y.to(hidden_states.dtype)
        return y, post, comb

    @staticmethod
    def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps):
        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + hc_eps
        post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0))
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

        comb = comb.softmax(-1) + hc_eps
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + hc_eps)
        for _ in range(sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            comb = comb / (row_sum + hc_eps)
            col_sum = comb.sum(-2, keepdim=True)
            comb = comb / (col_sum + hc_eps)
        return pre, post, comb

    @staticmethod
    def hc_post(hidden_states, residual, post, comb):
        if _HAS_NPU_HC_POST:
            try:
                return torch.ops.custom.npu_hc_post(hidden_states, residual, post, comb)
            except RuntimeError:
                pass
        return OpKernel._hc_post_native(hidden_states, residual, post, comb)

    @staticmethod
    def _hc_post_native(hidden_states, residual, post, comb):
        y = (
            post.unsqueeze(-1) * hidden_states.float().unsqueeze(-2)
            + torch.sum(comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=2)
        )
        return y.type_as(hidden_states)


class DeepseekV4DecoderLayer(nn.Module):

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        self.self_attn = DeepseekV4Attention(config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV4MoE(config, layer_idx)

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

        self.attn_norm = MojoRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = MojoRMSNorm(config.hidden_size, config.rms_norm_eps)

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
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale,
            self.hc_ffn_base, self.hc_mult, self.hc_sinkhorn_iters,
            self.norm_eps, self.hc_eps
        )
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids=input_ids)
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        return hidden_states


class DeepseekV4Model(nn.Module):

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = MojoRMSNorm(eps=config.rms_norm_eps, norm_size=config.hidden_size)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config=config)

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
        **kwargs,
    ) -> Tuple[torch.Tensor, PagedDummyCache]:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if past_key_values is None:
            past_key_values = PagedDummyCache(self.config, batch_size=batch_size, device=str(device), block_size=128, max_seq_len=max(seq_len * 4, 4096))

        past_len = int(past_key_values.get_seq_length(0).max().item())
        position_ids = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, attention_mask=attention_mask,
                position_embeddings=position_embeddings, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache,
                input_ids=input_ids, **kwargs,
            )

        hidden_states = self._hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class DeepseekV4ForCausalLM(nn.Module):

    def __init__(self, config, num_layers=None):
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
        self.model = DeepseekV4Model(config)
        self.lm_head = MojoGemm(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, PagedDummyCache]:
        hidden_states, past_key_values = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache, **kwargs,
        )
        hidden_states_flat = hidden_states.view(-1, self.config.hidden_size).to(torch.bfloat16)
        logits = self.lm_head(hidden_states_flat)
        logits = logits.view(*hidden_states.shape[:-1], self.config.vocab_size)
        return logits, past_key_values

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
            for eid in range(model.config.n_routed_experts):
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
                if "experts." in ck_key and ".ffn." in ck_key:
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
        n_experts = model.config.n_routed_experts
        intermediate_size = model.config.moe_intermediate_size
        hidden_size = model.config.hidden_size

        for layer_idx in range(model.config.num_hidden_layers):
            experts_mod = model.model.layers[layer_idx].mlp.experts

            up_proj_w = torch.empty(n_experts, intermediate_size * 2, hidden_size, dtype=torch.int8)
            up_proj_s = torch.empty(n_experts, intermediate_size * 2, dtype=torch.bfloat16)
            down_proj_w = torch.empty(n_experts, hidden_size, intermediate_size, dtype=torch.int8)
            down_proj_s = torch.empty(n_experts, hidden_size, dtype=torch.bfloat16)

            for eid in range(n_experts):
                w1_key = f"layers.{layer_idx}.ffn.experts.{eid}.w1.weight"
                w3_key = f"layers.{layer_idx}.ffn.experts.{eid}.w3.weight"
                w2_key = f"layers.{layer_idx}.ffn.experts.{eid}.w2.weight"
                s1_key = f"layers.{layer_idx}.ffn.experts.{eid}.w1.scale"
                s3_key = f"layers.{layer_idx}.ffn.experts.{eid}.w3.scale"
                s2_key = f"layers.{layer_idx}.ffn.experts.{eid}.w2.scale"

                if w1_key in expert_weights:
                    up_proj_w[eid, :intermediate_size, :] = expert_weights[w1_key]
                if w3_key in expert_weights:
                    up_proj_w[eid, intermediate_size:, :] = expert_weights[w3_key]
                if w2_key in expert_weights:
                    down_proj_w[eid] = expert_weights[w2_key]
                if s1_key in expert_weights:
                    up_proj_s[eid, :intermediate_size] = expert_weights[s1_key].squeeze(-1).to(torch.bfloat16)
                if s3_key in expert_weights:
                    up_proj_s[eid, intermediate_size:] = expert_weights[s3_key].squeeze(-1).to(torch.bfloat16)
                if s2_key in expert_weights:
                    down_proj_s[eid] = expert_weights[s2_key].squeeze(-1).to(torch.bfloat16)

            experts_mod.up_proj_weight.copy_(up_proj_w)
            experts_mod.up_proj_weight_scale.data.copy_(up_proj_s)
            experts_mod.down_proj_weight.copy_(down_proj_w)
            experts_mod.down_proj_weight_scale.data.copy_(down_proj_s)
