from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from mojo_opset import MojoRMSNorm
from mojo_opset import MojoSilu
from mojo_opset import MojoRoPE
from mojo_opset import MojoStorePagedKVCache
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoPagedDecodeGQA


class SeedOssConfig:
    def __init__(self):
        self.vocab_size = 155136
        self.max_position_embeddings = 8192
        self.hidden_size = 5120
        self.intermediate_size = 27648
        self.num_hidden_layers = 64
        self.num_attention_heads = 80
        self.num_key_value_heads = 8

        self.hidden_act = "silu"
        self.initializer_range = 0.02
        self.rms_norm_eps = 1e-06
        self.use_cache = True
        self.attention_bias = True
        self.attention_out_bias = False
        self.attention_dropout = 0.1
        self.residual_dropout = 0.1
        self.mlp_bias = False
        self.head_dim = 128
        self.rope_scaling = {"rope_type": "default"}
        self.rope_theta = (10000000.0,)

        self.tie_word_embeddings = False
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2


class PagedDummyCache:
    def __init__(self, config: SeedOssConfig, batch_size: int, device: str, block_size: int = 16):
        self.num_layers = config.num_hidden_layers
        self.device = device
        self.block_size = block_size
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.batch_size = batch_size
        max_blocks_per_seq = (config.max_position_embeddings + self.block_size - 1) // self.block_size
        total_blocks = self.batch_size * max_blocks_per_seq * self.num_layers
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
        self.block_tables = torch.zeros(
            (self.num_layers, self.batch_size, max_blocks_per_seq), dtype=torch.long, device=self.device
        )
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

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        batch_size, head_num, new_seq_len, head_dim = key_states.shape
        key_states = key_states.permute(0, 2, 1, 3).contiguous().view(-1, head_num, head_dim)
        value_states = value_states.permute(0, 2, 1, 3).contiguous().view(-1, head_num, head_dim)
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
        return self.k_cache, self.v_cache, self.block_tables[layer_idx, :, :max_blocks]

    def get_seq_length(self, layer_idx: int = 0):
        return self.seq_lens[layer_idx].clone()


class SeedOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = MojoSilu()

        self.residual_dropout = config.residual_dropout

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        down_proj = nn.functional.dropout(down_proj, p=self.residual_dropout, training=self.training)
        return down_proj


class SeedOssAttention(nn.Module):
    def __init__(self, config: SeedOssConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_out_bias
        )
        self.rope = MojoRoPE()
        self.attn_prefill = MojoPagedPrefillGQA()
        self.attn_decode = MojoPagedDecodeGQA()

        self.residual_dropout = config.residual_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[PagedDummyCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        if past_key_values is None:
            raise ValueError("PagedDummyCache is required for SeedOssAttention")

        context_lens = past_key_values.get_seq_length(self.layer_idx)
        attn_output = self._paged_attention_forward(
            query_states,
            key_states,
            value_states,
            past_key_values,
            context_lens,
        ).transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = nn.functional.dropout(attn_output, p=self.residual_dropout, training=self.training)

        return attn_output

    def _paged_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        past_key_values,
        context_lens: torch.Tensor,
    ):
        bsz, num_q_heads, q_len, head_dim = query_states.shape
        device = query_states.device
        past_key_values.update(key_states, value_states, self.layer_idx)

        if q_len > 1:
            q_lens = torch.full((bsz,), q_len, dtype=torch.int32, device=device)
            cu_seqlens_q = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), q_lens.cumsum(0)])
            total_tokens = int(cu_seqlens_q[-1].item())
            q = query_states.permute(0, 2, 1, 3).reshape(total_tokens, num_q_heads, head_dim)
            k_cache = past_key_values.k_cache
            v_cache = past_key_values.v_cache
            block_tables = past_key_values.block_tables[self.layer_idx]
            attn_output_tnd = self.attn_prefill(q, k_cache, v_cache, cu_seqlens_q, block_tables, self.scaling)
            attn_output = attn_output_tnd.reshape(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
        else:
            q = query_states.squeeze(2)
            k_cache, v_cache, block_tables = past_key_values.get_kv_for_decode(self.layer_idx)
            current_seq_lens = context_lens + 1
            attn_output_bhd = self.attn_decode(q, k_cache, v_cache, current_seq_lens, block_tables, self.scaling)
            attn_output = attn_output_bhd.unsqueeze(2)

        return attn_output


class SeedOssDecoderLayer(nn.Module):
    def __init__(self, config: SeedOssConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SeedOssAttention(config=config, layer_idx=layer_idx)

        self.mlp = SeedOssMLP(config)
        self.input_layernorm = MojoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MojoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SeedOssRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: SeedOssConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        dim = config.head_dim
        if hasattr(config, "rope_theta"):
            theta = config.rope_theta
        else:
            theta = 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class SeedOssModel(nn.Module):
    def __init__(self, config: SeedOssConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SeedOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MojoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SeedOssRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            bsz = inputs_embeds.shape[0]
            past_key_values = PagedDummyCache(
                self.config, batch_size=bsz, device=str(inputs_embeds.device), block_size=16
            )

        if cache_position is None:
            if past_key_values is not None:
                past_seen_tokens = past_key_values.get_seq_length()  # [B]
            else:
                past_seen_tokens = torch.zeros(inputs_embeds.size(0), dtype=torch.long, device=inputs_embeds.device)
            step = torch.arange(inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device)  # [S]
            cache_position: torch.Tensor = past_seen_tokens[:, None] + step[None, :]  # [B, S]

        if position_ids is None:
            position_ids = cache_position

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class SeedOssForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = SeedOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
