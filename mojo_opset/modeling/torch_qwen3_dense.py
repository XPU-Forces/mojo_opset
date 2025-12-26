import math

from typing import Optional

import torch

from torch import nn


def silu(x):
    return nn.functional.silu(x)


class Qwen3Config:
    def __init__(self):
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.rms_norm_eps = 1e-6
        self.attention_bias = True
        self.hidden_act = "silu"
        self.attention_dropout = 0.0
        self.layer_types = ["full_attention"]
        self.num_hidden_layers = 1
        self.sliding_window = None
        self._attn_implementation = "eager"
        self.max_position_embeddings = 8192

        self.rope_theta = 10000.0
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": 10000.0,
        }
        self.head_dim = self.hidden_size // self.num_attention_heads


class PagedDummyCache:
    def __init__(self, config: Qwen3Config, batch_size: int, device: str, block_size: int = 16):
        self.num_layers = config.num_hidden_layers
        self.device = device
        self.block_size = block_size
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.batch_size = batch_size

        max_blocks_per_seq = (config.max_position_embeddings + self.block_size - 1) // self.block_size
        total_blocks = self.batch_size * max_blocks_per_seq

        self.k_cache = torch.zeros(
            (total_blocks, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.v_cache = torch.zeros(
            (total_blocks, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )

        self.block_tables = torch.zeros((self.batch_size, max_blocks_per_seq), dtype=torch.long, device=self.device)
        self.seq_lens = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        self.free_blocks = torch.arange(total_blocks, device=self.device, dtype=torch.long)
        self.num_free_blocks = total_blocks

    def _allocate_blocks(self, num_blocks: int):
        if num_blocks > self.num_free_blocks:
            raise ValueError("PagedDummyCache: Out of memory!")
        allocated = self.free_blocks[self.num_free_blocks - num_blocks : self.num_free_blocks]
        self.num_free_blocks -= num_blocks
        return allocated

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        batch_size, _, new_seq_len, _ = key_states.shape

        for i in range(batch_size):
            context_len = self.seq_lens[i].item()

            old_num_blocks = (context_len + self.block_size - 1) // self.block_size
            new_total_len = context_len + new_seq_len
            new_num_blocks = (new_total_len + self.block_size - 1) // self.block_size

            if new_num_blocks > old_num_blocks:
                num_to_allocate = new_num_blocks - old_num_blocks
                newly_allocated = self._allocate_blocks(num_to_allocate)
                self.block_tables[i, old_num_blocks:new_num_blocks] = newly_allocated

            for j in range(new_seq_len):
                logical_pos = context_len + j
                block_idx_in_table = logical_pos // self.block_size
                pos_in_block = logical_pos % self.block_size

                physical_block_id = self.block_tables[i, block_idx_in_table]

                self.k_cache[physical_block_id, :, pos_in_block, :] = key_states[i, :, j, :]
                self.v_cache[physical_block_id, :, pos_in_block, :] = value_states[i, :, j, :]

            self.seq_lens[i] = new_total_len

    def get_kv_for_prefill(self, layer_idx: int):
        return None, None

    def get_kv_for_decode(self, layer_idx: int):
        max_slen = self.seq_lens.max().item()
        max_blocks = (max_slen + self.block_size - 1) // self.block_size
        return self.k_cache, self.v_cache, self.block_tables

    def get_seq_length(self, layer_idx: int = 0):
        return self.seq_lens.clone()


class DummyGradientCheckpointingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_gradient_checkpointing = False


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.config = config
        dim = config.head_dim
        base = config.rope_theta
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "meta_device"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6, norm_type: str = "rmsnorm", gamma: Optional[torch.Tensor] = None):
        super().__init__()
        self.epsilon = float(eps)
        self.gamma = gamma

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)
        return self.gamma * hidden_states.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size, self.intermediate_size = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def paged_attention_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    total_q_tokens, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, block_size, _ = k_cache.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    total_kv_tokens = total_q_tokens

    k_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
    v_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    batch_size = len(q_lens)

    for i in range(batch_size):
        seq_len = q_lens[i].item()
        start_loc = cu_seqlens_q[i].item()
        end_loc = cu_seqlens_q[i + 1].item()

        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        for j in range(num_blocks_for_seq):
            physical_block_id = block_tables[i, j].item()

            start_pos_in_seq = j * block_size
            tokens_in_block = min(block_size, seq_len - start_pos_in_seq)

            start_loc_in_batch = start_loc + start_pos_in_seq
            end_loc_in_batch = start_loc_in_batch + tokens_in_block

            k_slice = k_cache[physical_block_id, :, :tokens_in_block, :]

            k_unpadded[start_loc_in_batch:end_loc_in_batch, :, :] = k_slice.permute(1, 0, 2)

            v_slice = v_cache[physical_block_id, :, :tokens_in_block, :]
            v_unpadded[start_loc_in_batch:end_loc_in_batch, :, :] = v_slice.permute(1, 0, 2)

    if num_q_heads != num_kv_heads:
        k_expanded = k_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        v_expanded = v_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    else:
        k_expanded = k_unpadded
        v_expanded = v_unpadded

    attn_mask = torch.ones(total_q_tokens, total_q_tokens, device=q.device, dtype=torch.bool).tril(diagonal=0)

    tok_to_seq = torch.repeat_interleave(torch.arange(batch_size, device=q.device), q_lens)

    seq_mask = tok_to_seq[:, None] == tok_to_seq[None, :]
    final_mask = attn_mask & seq_mask

    attn_scores = torch.einsum("thd,khd->thk", q, k_expanded) * softmax_scale
    attn_scores.masked_fill_(~final_mask.unsqueeze(1), -torch.inf)

    attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

    output = torch.einsum("thk,khd->thd", attn_probs, v_expanded)
    return output


def paged_attention_decode(q, k_cache, v_cache, seqlens, block_tables, softmax_scale):
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads, block_size, head_dim = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    max_len_in_batch = seqlens.max().item()

    k_ref = torch.zeros(batch_size, max_len_in_batch, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
    v_ref = torch.zeros(batch_size, max_len_in_batch, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)

    for i in range(batch_size):
        seq_len = seqlens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        for j in range(num_blocks_for_seq):
            physical_block_id = block_tables[i, j].item()

            start_pos = j * block_size
            tokens_in_block = min(block_size, seq_len - start_pos)

            k_slice = k_cache[physical_block_id, :, :tokens_in_block, :]
            v_slice = v_cache[physical_block_id, :, :tokens_in_block, :]

            k_ref[i, start_pos : start_pos + tokens_in_block, :, :] = k_slice.permute(1, 0, 2)
            v_ref[i, start_pos : start_pos + tokens_in_block, :, :] = v_slice.permute(1, 0, 2)

    _, k_len, num_k_heads, _ = k_ref.shape
    num_share_q_heads = num_q_heads // num_k_heads
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim)

    if num_share_q_heads > 1:
        k_ref = k_ref.repeat_interleave(num_share_q_heads, dim=2)
        v_ref = v_ref.repeat_interleave(num_share_q_heads, dim=2)

    attn = torch.einsum("bhd,bkhd->bhk", q, k_ref) * softmax_scale

    mask = torch.arange(k_len, device=q.device)[None, :] >= seqlens[:, None]
    attn.masked_fill_(mask[:, None, :], -torch.inf)

    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum("bhk,bkhd->bhd", attn, v_ref)
    return out


def paged_attention_forward(
    module: "Qwen3Attention",
    query_states: torch.Tensor,  # [BNSD]
    key_states: torch.Tensor,  # [BNSD]
    value_states: torch.Tensor,  # [BNSD]
    past_key_values: PagedDummyCache,
    context_lens: torch.Tensor,
    **kwargs,
):
    bsz, num_q_heads, q_len, head_dim = query_states.shape
    device = query_states.device

    past_key_values.update(key_states, value_states, module.layer_idx)

    if q_len > 1:
        q_lens = torch.full((bsz,), q_len, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), q_lens.cumsum(0)])
        total_tokens = cu_seqlens_q[-1].item()

        q = query_states.permute(0, 2, 1, 3).reshape(total_tokens, num_q_heads, head_dim)

        current_seq_lens = context_lens + q_len
        max_len_in_batch = current_seq_lens.max().item()
        max_blocks = (max_len_in_batch + past_key_values.block_size - 1) // past_key_values.block_size

        k_cache = past_key_values.k_cache
        v_cache = past_key_values.v_cache

        block_tables = past_key_values.block_tables

        attn_output_tnd = paged_attention_prefill(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            block_tables,
            softmax_scale=module.scaling,
        )

        attn_output = attn_output_tnd.reshape(bsz, q_len, num_q_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)

    else:
        q = query_states.squeeze(2)
        k_cache, v_cache, block_tables = past_key_values.get_kv_for_decode(module.layer_idx)
        current_seq_lens = context_lens + 1

        attn_output_bhd = paged_attention_decode(q, k_cache, v_cache, current_seq_lens, block_tables, module.scaling)
        attn_output = attn_output_bhd.unsqueeze(2)

    return attn_output, None


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config, self.layer_idx, self.hidden_size = config, layer_idx, config.hidden_size
        self.num_heads, self.head_dim = config.num_attention_heads, self.hidden_size // config.num_attention_heads
        self.num_key_value_heads, self.num_key_value_groups = (
            config.num_key_value_heads,
            self.num_heads // config.num_key_value_heads,
        )
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.gamma = nn.Parameter(torch.ones(self.head_dim))

        self.q_norm = (
            Qwen3RMSNorm(eps=config.rms_norm_eps, gamma=self.gamma) if hasattr(config, "q_norm") else nn.Identity()
        )
        self.k_norm = (
            Qwen3RMSNorm(eps=config.rms_norm_eps, gamma=self.gamma) if hasattr(config, "k_norm") else nn.Identity()
        )

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, use_cache, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        context_lens = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else torch.zeros(bsz, dtype=torch.long, device=hidden_states.device)
        )

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)  # [BSND]
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)  # [BSND]
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)  # [BSND]

        query_states = self.q_norm(query_states).transpose(1, 2)  # [BNSD]
        key_states = self.k_norm(key_states).transpose(1, 2)  # [BNSD]
        value_states = value_states.transpose(1, 2)  # [BNSD]
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is None:
            raise ValueError("Paged Attention requires a PagedDummyCache instance.")

        attn_output, _ = paged_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            past_key_values=past_key_values,
            context_lens=context_lens,
        )

        attn_output = query_states.reshape(bsz, q_len, self.hidden_size).contiguous()
        return self.o_proj(attn_output), None


class Qwen3DecoderLayer(DummyGradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.input_layernorm = Qwen3RMSNorm(config.rms_norm_eps, gamma=self.gamma)
        self.post_attention_layernorm = Qwen3RMSNorm(config.rms_norm_eps, gamma=self.gamma)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
