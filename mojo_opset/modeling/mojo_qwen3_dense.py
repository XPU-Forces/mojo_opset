from typing import Optional
from typing import Tuple

import torch

from torch import nn

from mojo_opset import MojoLinear
from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoRoPE
from mojo_opset import MojoSilu
from mojo_opset import MojoStorePagedKVCache


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
        # Allocate total blocks for ALL layers
        total_blocks = self.batch_size * max_blocks_per_seq * self.num_layers

        # k_cache/v_cache now holds blocks for all layers
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

        # block_tables needs to be per-layer
        self.block_tables = torch.zeros(
            (self.num_layers, self.batch_size, max_blocks_per_seq), dtype=torch.long, device=self.device
        )
        # seq_lens needs to be per-layer
        self.seq_lens = torch.zeros((self.num_layers, self.batch_size), dtype=torch.long, device=self.device)

        self.free_blocks = torch.arange(total_blocks, device=self.device, dtype=torch.long)
        self.num_free_blocks = total_blocks
        self.store_paged_kv = MojoStorePagedKVCache()

    def _allocate_blocks(self, num_blocks: int):
        if num_blocks > self.num_free_blocks:
            raise ValueError("PagedDummyCache: Out of memory!")
        allocated = self.free_blocks[self.num_free_blocks - num_blocks : self.num_free_blocks]
        self.num_free_blocks -= num_blocks
        return allocated

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        batch_size, head_num, new_seq_len, head_dim = key_states.shape

        key_states = key_states.permute(0, 2, 1, 3).reshape(-1, head_num, head_dim)
        value_states = value_states.permute(0, 2, 1, 3).reshape(-1, head_num, head_dim)
        cu_seqlens = torch.arange(0, (batch_size + 1) * new_seq_len, step=new_seq_len, device=key_states.device)

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

        self.store_paged_kv(
            key_states,
            value_states,
            self.k_cache,
            self.v_cache,
            self.block_tables[layer_idx],
            cu_seqlens,
            current_seq_lens,
        )
        self.seq_lens[layer_idx] += new_seq_len

    def get_kv_for_prefill(self, layer_idx: int):
        return None, None

    def get_kv_for_decode(self, layer_idx: int):
        max_slen = self.seq_lens[layer_idx].max().item()
        max_blocks = (max_slen + self.block_size - 1) // self.block_size
        # Return per-layer block table
        return self.k_cache, self.v_cache, self.block_tables[layer_idx, :, :max_blocks]

    def get_seq_length(self, layer_idx: int = 0):
        return self.seq_lens[layer_idx].clone()


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


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size, self.intermediate_size = config.hidden_size, config.intermediate_size
        self.gate_proj = MojoLinear(weight=nn.Parameter(torch.ones(self.intermediate_size, self.hidden_size)))
        self.up_proj = MojoLinear(weight=nn.Parameter(torch.ones(self.intermediate_size, self.hidden_size)))
        self.down_proj = MojoLinear(weight=nn.Parameter(torch.ones(self.hidden_size, self.intermediate_size)))
        self.act_fn = MojoSilu()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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

        self.q_proj = MojoLinear(weight=nn.Parameter(torch.ones(self.num_heads * self.head_dim, self.hidden_size)))
        self.k_proj = MojoLinear(
            weight=nn.Parameter(torch.ones(self.num_key_value_heads * self.head_dim, self.hidden_size))
        )
        self.v_proj = MojoLinear(
            weight=nn.Parameter(torch.ones(self.num_key_value_heads * self.head_dim, self.hidden_size))
        )
        self.o_proj = MojoLinear(weight=nn.Parameter(torch.ones(self.hidden_size, self.num_heads * self.head_dim)))

        self.q_norm = MojoRMSNorm(
            norm_size=self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = MojoRMSNorm(
            norm_size=self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.rope = MojoRoPE()
        self.attn_prefill = MojoPagedPrefillGQA()
        self.attn_decode = MojoPagedDecodeGQA()

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
        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        if past_key_values is None:
            raise ValueError("Paged Attention requires a PagedDummyCache instance.")

        attn_output, _ = self.paged_attention_forward(
            query_states,
            key_states,
            value_states,
            past_key_values=past_key_values,
            context_lens=context_lens,
        )

        # Corrected reshape logic (from hf_qwen3_dense_demo.py patch)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size).contiguous()
        return self.o_proj(attn_output), None

    def paged_attention_forward(
        self,
        query_states: torch.Tensor,  # [BNSD]
        key_states: torch.Tensor,  # [BNSD]
        value_states: torch.Tensor,  # [BNSD]
        past_key_values: PagedDummyCache,
        context_lens: torch.Tensor,
    ):
        bsz, num_q_heads, q_len, head_dim = query_states.shape
        device = query_states.device

        past_key_values.update(key_states, value_states, self.layer_idx)

        if q_len > 1:
            q_lens = torch.full((bsz,), q_len, dtype=torch.int32, device=device)
            cu_seqlens_q = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), q_lens.cumsum(0)])
            total_tokens = cu_seqlens_q[-1].item()

            q = query_states.permute(0, 2, 1, 3).reshape(total_tokens, num_q_heads, head_dim)

            current_seq_lens = context_lens + q_len

            k_cache = past_key_values.k_cache
            v_cache = past_key_values.v_cache

            # Use per-layer block table
            block_tables = past_key_values.block_tables[self.layer_idx]

            attn_output_tnd = self.attn_prefill(q, k_cache, v_cache, cu_seqlens_q, block_tables, self.scaling, current_seq_lens)
            attn_output = attn_output_tnd.reshape(bsz, q_len, num_q_heads, head_dim)
            attn_output = attn_output.transpose(1, 2)

        else:
            q = query_states.squeeze(2)
            k_cache, v_cache, block_tables = past_key_values.get_kv_for_decode(self.layer_idx)
            current_seq_lens = context_lens + 1

            attn_output_bhd = self.attn_decode(q, k_cache, v_cache, current_seq_lens, block_tables, self.scaling)
            attn_output = attn_output_bhd.unsqueeze(2)

        return attn_output, None


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.layer_idx = layer_idx

        self.input_layernorm = MojoRMSNorm(norm_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MojoRMSNorm(
            norm_size=config.hidden_size, eps=config.rms_norm_eps
        )

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


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])

        self.norm = MojoRMSNorm(norm_size=config.hidden_size, eps=config.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional["PagedDummyCache"] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, "PagedDummyCache"]:
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        if past_key_values is None:
            past_key_values = PagedDummyCache(self.config, batch_size=bsz, device=str(device), block_size=16)

        past_len = int(past_key_values.get_seq_length(0).max().item())
        position_ids = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary(hidden_states, position_ids)  # position_ids increment
        position_embeddings = (cos, sin)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=None,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = MojoLinear(weight=nn.Parameter(torch.ones(config.vocab_size, config.hidden_size)))

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional["PagedDummyCache"] = None,
        use_cache: bool = True,
    ):
        hidden_states, past_key_values = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache)
        logits = self.lm_head(hidden_states)
        return logits, past_key_values
