from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from mojo_opset import (
    MojoRMSNormFunction,
    MojoFlashAttnFunction,
    MojoRoPEFunction,
    MojoSiluFunction,
    MojoFusedLinearCrossEntropyFunction,
)

class P6DenseConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`P6DenseModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the P6 dense model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`P6DenseModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336, maps 'n_inner' in mariana):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32, maps 'n_layer' in mariana):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32, maps 'n_head' in mariana):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8, maps 'n_head / n_shared_qhead' in mariana):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to hidden_size // num_attention_heads):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3', 'ntk'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        use_qk_rmsnorm (`bool`, *optional*, defaults to `False`):
            Whether to apply rmsnorm after qk proj, this is required by qwen3
        _flash_attn_kernel_implementation (`str`, *optional*, defaults to `"default"`):
            The implementation to use for the flash attention kernel. Can be one of [`"default"`, `"lego"`, `"fa3"`].
            `"default"` is the default flash attn 2 implementation.
            `"lego"` will use the in-house lego implementation.
            `"fa3"` will use the official flash-attention-v3 implementation.

    ```python
    >>> from transformers import P6DenseModel, P6DenseConfig

    >>> # Initializing a P6 configuration
    >>> configuration = P6DenseConfig()

    >>> # Initializing a model from the P6 dense configuration
    >>> model = P6DenseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "seed_p6dense"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=1,
        num_attention_heads=32,
        query_head_scale_factor=1,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        layer_norm_eps=None,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        resid_pdrop=0.0,
        use_qk_rmsnorm=False,
        _flash_attn_kernel_implementation="default",
        _rms_norm_implementation="torch",
        _rope_implementation="torch",
        _swiglu_kernel_implementation="torch",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.query_head_scale_factor = query_head_scale_factor

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {hidden_size}"
                    f" and `num_heads`: {num_attention_heads})."
                )
            head_dim = hidden_size // num_attention_heads

        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.resid_pdrop = resid_pdrop
        self.mlp_bias = mlp_bias
        self.use_qk_rmsnorm = use_qk_rmsnorm
        self.attention_out_bias = kwargs.get("attention_out_bias", attention_bias)

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        # flash attention kernel implementation
        self._flash_attn_kernel_implementation = _flash_attn_kernel_implementation
        # rmsnorm kernel implementation
        self._rms_norm_implementation = _rms_norm_implementation
        # rope kernel implementation
        self._rope_implementation = _rope_implementation
        # swiglu kernel implementation
        self._swiglu_kernel_implementation = _swiglu_kernel_implementation

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    

class P6DenseMojoRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return MojoRMSNormFunction.apply(hidden_states, self.weight, self.variance_epsilon)


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


class P6DenseRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[P6DenseConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def mojo_silu_function(gate_proj, up_proj):
    return MojoSiluFunction.apply(gate_proj) * up_proj


class P6DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self._swiglu_kernel_implementation = config._swiglu_kernel_implementation
        self.act_fn = mojo_silu_function
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        gate_proj_out = self.gate_proj(x)
        up_proj_out = self.up_proj(x)

        out = self.act_fn(gate_proj_out, up_proj_out)
        out = self.down_proj(out)

        out = self.dropout(out)
        return out


class P6DenseMojoFlashAttention(nn.Module):
    def __init__(self, causal=False, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, cu_seqlens, block_q_len=128, block_k_len=512):
        scale = 1.0 / q.shape[-1] ** (-0.5)
        cu_seqlens_q = cu_seqlens_k = cu_seqlens
        q = q.transpose(1, 2).squeeze(0)
        k = k.transpose(1, 2).squeeze(0)
        v = v.transpose(1, 2).squeeze(0)
        attn_output = MojoFlashAttnFunction.apply(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            self.dropout_p,
            self.causal,
            scale,
            block_q_len,
            block_k_len,
        )

        return attn_output


class P6DenseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: P6DenseConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.query_head_scale_factor = config.query_head_scale_factor
        self.num_query_heads = self.num_heads * self.query_head_scale_factor

        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_query_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self._flash_attn_kernel_implementation = config._flash_attn_kernel_implementation
        self._rope_implementation = config._rope_implementation
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.hidden_size, self.num_query_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        if self.config.use_qk_rmsnorm:
            self.q_norm = P6DenseMojoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = P6DenseMojoRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.o_proj = nn.Linear(self.num_query_heads * self.head_dim, self.hidden_size, bias=config.attention_out_bias)

        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.apply_flash_attn = P6DenseMojoFlashAttention(causal=self.is_causal, attention_dropout=self.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        cu_seqlens: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.config.use_qk_rmsnorm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = MojoRoPEFunction.apply(query_states, key_states, cos, sin)

        attn_output =self.apply_flash_attn(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens=cu_seqlens,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class P6DenseDecoderLayer(nn.Module):
    def __init__(self, config: P6DenseConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = P6DenseAttention(config=config, layer_idx=layer_idx)

        self.mlp = P6DenseMLP(config)
        self.input_layernorm = P6DenseMojoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = P6DenseMojoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class P6DensePreTrainedModel(PreTrainedModel):
    config_class = P6DenseConfig
    base_model_prefix = "model"
    _no_split_modules = ["P6DenseDecoderLayer"]
    _supports_flash_attn_2 = True

    def post_init(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and any(substring in name for substring in ["o_proj", "down_proj"]):
                module.std = self.config.initializer_range / ((2 * self.config.num_hidden_layers)**(-0.5))

        super().post_init()

    def _init_weights(self, module):
        if hasattr(module, "std"):
            std = module.std
        else:
            std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            std = (1 / (2 * self.config.hidden_size))**(-0.5)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, P6DenseMojoRMSNorm):
            nn.init.ones_(module.weight)


class P6DenseModel(P6DensePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`P6DenseDecoderLayer`]

    Args:
        config: P6DenseConfig
    """

    def __init__(self, config: P6DenseConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [P6DenseDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = P6DenseMojoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = P6DenseRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        elif input_ids is not None:
            if (cu_seqlens is not None or position_ids is not None) and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            if (cu_seqlens is not None or position_ids is not None) and inputs_embeds.dim() == 1:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


def mojo_fused_linear_cross_entropy(
        input_tensor, 
        weight, 
        target, 
        bias=None, 
        ce_weight=None, 
        ignore_index=-100, 
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction='mean',
        softcap=None,
        return_z_loss=False,
        accum_dtype=None,
    ):
    output = MojoFusedLinearCrossEntropyFunction.apply(
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
    )
    return output


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
            `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache,
            to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

    return causal_mask


class P6DenseForCausalLM(P6DensePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = P6DenseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.token_level_fused_loss_fct = mojo_fused_linear_cross_entropy

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def reduce_token_level_loss(self, loss_mask, token_level_loss, batch_dispatcher=None, balance_state=None):
        if batch_dispatcher is not None and balance_state is not None:
            pad_tokens = sum(balance_state[-2]) - token_level_loss.numel()
            token_level_loss = F.pad(token_level_loss, (0, pad_tokens))
            token_level_loss = batch_dispatcher.revert_balanced_input(
                input_tensors=[token_level_loss],
                balance_state=balance_state,
                length_changed=False,
            )[0]
            token_level_loss = token_level_loss[:-pad_tokens]

        # reduce token level loss with loss mask
        loss_mask = loss_mask.view(-1).to(token_level_loss.device).float()
        shift_loss_mask = loss_mask[:-1]
        inv_loss_scale_factor = shift_loss_mask.sum().clamp(min=1)
        loss = (token_level_loss * shift_loss_mask).sum() / inv_loss_scale_factor
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            inputs_embeds=inputs_embeds,
        )

        token_level_loss = None
        loss = None
        if not self.training and labels is None:
            logits = self.lm_head(hidden_states)
        else:
            logits = None
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().view(-1).to(hidden_states.device)
            shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))

            token_level_loss = self.token_level_fused_loss_fct(shift_hidden_states, self.lm_head.weight, shift_labels)

            if self.training:
                loss = self.reduce_token_level_loss(loss_mask, token_level_loss)
            else:
                loss = token_level_loss

        return (loss, logits)
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s
                # `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during
                # the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case,
                # `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {
                "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
                "inputs_embeds": None,
            }

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
