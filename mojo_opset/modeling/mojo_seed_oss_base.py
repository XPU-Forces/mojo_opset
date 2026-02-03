

from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel

from mojo_opset import MojoRMSNorm
from mojo_opset import MojoLinear
from mojo_opset import MojoSilu
from mojo_opset import MojoRoPE
from mojo_opset import MojoSdpa


class SeedOssConfig():
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
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": 10000000.0,
        }

        self.tie_word_embeddings = False
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2


class SeedOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = MojoLinear(
            weight=nn.Parameter(torch.empty(self.intermediate_size, self.hidden_size)),
            bias=(nn.Parameter(torch.zeros(self.intermediate_size)) if config.mlp_bias else None),
        )
        self.up_proj = MojoLinear(
            weight=nn.Parameter(torch.empty(self.intermediate_size, self.hidden_size)),
            bias=(nn.Parameter(torch.zeros(self.intermediate_size)) if config.mlp_bias else None),
        )
        self.down_proj = MojoLinear(
            weight=nn.Parameter(torch.empty(self.hidden_size, self.intermediate_size)),
            bias=(nn.Parameter(torch.zeros(self.hidden_size)) if config.mlp_bias else None),
        )
        self.act_fn = MojoSilu._registry.get("torch")()

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

        self.q_proj = MojoLinear(
            weight=nn.Parameter(torch.empty(self.num_attention_heads * self.head_dim, config.hidden_size)),
            bias=(nn.Parameter(torch.zeros(self.num_attention_heads * self.head_dim)) if config.attention_bias else None),
        )
        self.k_proj = MojoLinear(
            weight=nn.Parameter(torch.empty(config.num_key_value_heads * self.head_dim, config.hidden_size)),
            bias=(nn.Parameter(torch.zeros(config.num_key_value_heads * self.head_dim)) if config.attention_bias else None),
        )
        self.v_proj = MojoLinear(
            weight=nn.Parameter(torch.empty(config.num_key_value_heads * self.head_dim, config.hidden_size)),
            bias=(nn.Parameter(torch.zeros(config.num_key_value_heads * self.head_dim)) if config.attention_bias else None),
        )
        self.o_proj = MojoLinear(
            weight=nn.Parameter(torch.empty(config.hidden_size, self.num_attention_heads * self.head_dim)),
            bias=(nn.Parameter(torch.zeros(config.hidden_size)) if config.attention_out_bias else None),
        )
        self.rope = MojoRoPE._registry.get("torch")()
        self.sdpa = MojoSdpa._registry.get("torch")(scale=self.scaling)

        self.residual_dropout = config.residual_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
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

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention_mask
        attn_output = self.sdpa(
            query_states,
            key_states,
            value_states,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = nn.functional.dropout(attn_output, p=self.residual_dropout, training=self.training)

        return attn_output


class SeedOssDecoderLayer(nn.Module):
    def __init__(self, config: SeedOssConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SeedOssAttention(config=config, layer_idx=layer_idx)

        self.mlp = SeedOssMLP(config)
        self.input_layernorm = MojoRMSNorm._registry.get("torch")(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MojoRMSNorm._registry.get("torch")(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
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


class SeedOssModel(PreTrainedModel):
    def __init__(self, config: SeedOssConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SeedOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MojoRMSNorm._registry.get("torch")(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SeedOssRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
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


class SeedOssForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = SeedOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = MojoLinear(
            weight=nn.Parameter(torch.empty(config.vocab_size, config.hidden_size)),
            bias=None,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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
