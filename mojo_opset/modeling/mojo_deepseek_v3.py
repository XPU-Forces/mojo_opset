"""
DeepSeek V3 Model Implementation with Paged Attention

This module implements the DeepSeek V3 architecture with support for:
- Multi-Latent Attention (MLA)
- Mixture of Experts (MoE)
- Paged KV Cache for efficient inference
- RoPE positional embeddings
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from mojo_opset import (
    MojoLinear,
    MojoRMSNorm,
    MojoMoE,
    MojoPagedDecodeMLA,
    MojoPagedPrefillMLA,
    MojoRoPE,
    MojoSilu,
    MojoStorePagedKVCache,
)


class DeepseekV3Config:
    """Configuration class for DeepSeek V3 model."""
    
    def __init__(self):
        # Basic model dimensions
        self.vocab_size = 102400
        self.hidden_size = 7168
        self.intermediate_size = 18432
        self.num_hidden_layers = 2
        self.num_attention_heads = 128
        self.num_key_value_heads = 128
        
        # MoE configuration
        self.moe_intermediate_size = 2048
        self.n_shared_experts = 1
        self.n_routed_experts = 256
        self.num_experts_per_tok = 8
        self.routed_scaling_factor = 2.5
        self.n_group = 8
        self.topk_group = 4
        self.first_k_dense_replace = 1
        self.norm_topk_prob = True
        
        # MLA (Multi-head Latent Attention) configuration
        self.kv_lora_rank = 512
        self.q_lora_rank = 1536
        self.qk_rope_head_dim = 64
        self.qk_nope_head_dim = 128
        self.v_head_dim = 128
        
        # Attention configuration
        self.attention_bias = False
        self.attention_dropout = 0.0
    
        # Normalization and activation
        self.rms_norm_eps = 1e-6
        self.hidden_act = "silu"
        
        # RoPE configuration
        self.rope_theta = 10000.0
        self.max_position_embeddings = 4096
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
        }
        
        # Derived attributes
        self.head_dim = self.hidden_size // self.num_attention_heads


class PagedDummyCache:
    """
    Paged KV cache implementation for efficient memory management.
    
    Manages key-value cache using block-based allocation, supporting
    both prefill and decode phases with per-layer cache management.
    """
    
    def __init__(
        self,
        config: DeepseekV3Config,
        batch_size: int,
        device: str,
        block_size: int = 16,
    ):
        """
        Initialize paged cache.
        
        Args:
            config: Model configuration
            batch_size: Number of sequences in batch
            device: Device to allocate tensors on
            block_size: Number of tokens per cache block
        """
        self.num_layers = config.num_hidden_layers
        self.device = device
        self.block_size = block_size
        self.num_kv_heads = config.num_key_value_heads
        self.v_head_dim = config.v_head_dim
        self.k_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.batch_size = batch_size

        # Calculate total blocks needed
        max_blocks_per_seq = (
            config.max_position_embeddings + self.block_size - 1
        ) // self.block_size
        total_blocks = self.batch_size * max_blocks_per_seq * self.num_layers

        # Allocate cache tensors for all layers
        self.k_cache = torch.zeros(
            (total_blocks, self.num_kv_heads, self.block_size, self.k_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.v_cache = torch.zeros(
            (total_blocks, self.num_kv_heads, self.block_size, self.k_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Per-layer block tables and sequence lengths
        self.block_tables = torch.zeros(
            (self.num_layers, self.batch_size, max_blocks_per_seq),
            dtype=torch.long,
            device=self.device,
        )
        self.seq_lens = torch.zeros(
            (self.num_layers, self.batch_size),
            dtype=torch.long,
            device=self.device,
        )

        # Free block management
        self.free_blocks = torch.arange(
            total_blocks, device=self.device, dtype=torch.long
        )
        self.num_free_blocks = total_blocks
        
        # KV cache storage operator
        self.store_paged_kv = MojoStorePagedKVCache()

    def _allocate_blocks(self, num_blocks: int) -> torch.Tensor:
        """
        Allocate blocks from free pool.
        
        Args:
            num_blocks: Number of blocks to allocate
            
        Returns:
            Tensor of allocated block indices
            
        Raises:
            ValueError: If insufficient free blocks available
        """
        if num_blocks > self.num_free_blocks:
            raise ValueError(
                f"PagedDummyCache: Out of memory! "
                f"Requested {num_blocks} blocks, but only "
                f"{self.num_free_blocks} available."
            )
        
        allocated = self.free_blocks[
            self.num_free_blocks - num_blocks : self.num_free_blocks
        ]
        self.num_free_blocks -= num_blocks
        return allocated

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """
        Update cache with new key-value states.
        
        Args:
            key_states: New key states [B, num_heads, seq_len, head_dim]
            value_states: New value states [B, num_heads, seq_len, head_dim]
            layer_idx: Layer index to update
        """
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

    def get_kv_for_prefill(self, layer_idx: int) -> Tuple[None, None]:
        """Get KV cache for prefill phase (not used in current implementation)."""
        return None, None

    def get_kv_for_decode(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get KV cache for decode phase.
        Args:
            layer_idx: Layer index   
        Returns:
            Tuple of (k_cache, v_cache, block_tables)
        """
        max_slen = self.seq_lens[layer_idx].max().item()
        max_blocks = (max_slen + self.block_size - 1) // self.block_size
        return (
            self.k_cache,
            self.v_cache,
            self.block_tables[layer_idx, :, :max_blocks],
        )

    def get_seq_length(self, layer_idx: int = 0) -> torch.Tensor:
        """
        Get sequence lengths for a layer.
        Args:
            layer_idx: Layer index (default: 0)    
        Returns:
            Sequence lengths tensor [batch_size]
        """
        return self.seq_lens[layer_idx].clone()


class DeepseekV3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation."""
    
    def __init__(self, config: DeepseekV3Config, device: Optional[str] = None):
        """
        Initialize RoPE.
        
        Args:
            config: Model configuration
            device: Device to place tensors on
        """
        super().__init__()
        self.config = config
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        base = config.rope_theta
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.int64)
                .to(device=device, dtype=torch.float) / dim
            )
        )
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine and sine for RoPE.
        
        Args:
            x: Input tensor (used for dtype and device)
            position_ids: Position indices [batch_size, seq_len]
            
        Returns:
            Tuple of (cos, sin) tensors
        """

        # Expand dimensions for broadcasting
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float()
            .expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Handle device type for autocast
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV3MLP(nn.Module):
    """Feed-forward MLP with SwiGLU activation."""
    
    def __init__(self, config: DeepseekV3Config, intermediate_size: Optional[int] = None):
        """
        Initialize MLP.
        
        Args:
            config: Model configuration
            intermediate_size: Override intermediate size (optional)
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        
        self.gate_proj = MojoLinear(weight=nn.Parameter(torch.ones(self.intermediate_size, self.hidden_size)))
        self.up_proj = MojoLinear(weight=nn.Parameter(torch.ones(self.intermediate_size, self.hidden_size)))
        self.down_proj = MojoLinear(weight=nn.Parameter(torch.ones(self.hidden_size, self.intermediate_size)))
        self.act_fn = MojoSilu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SiLU activation."""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV3MoE(nn.Module):
    """Mixture of Experts module with shared experts."""
    
    def __init__(self, config: DeepseekV3Config):
        """
        Initialize MoE layer.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        self.routed_experts = MojoMoE(
            hidden_size=config.hidden_size,
            ffn_intermediate_size=config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
        )
        self.shared_experts = DeepseekV3MLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        
        # MoE configuration
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Output after routing through experts and shared experts
        """
        residuals = hidden_states #[BSH]
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.routed_experts(hidden_states).view(*orig_shape)
        
        # Add shared expert output
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class DeepseekV3Attention(nn.Module):
    """Multi-Latent Attention (MLA) with paged attention support."""
    
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        """
        Initialize attention layer.
        
        Args:
            config: Model configuration
            layer_idx: Layer index in the model
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads

        # MLA configuration
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        # Query projection (with optional LoRA)
        if self.q_lora_rank is None:
            self.q_proj = MojoLinear(
                weight=nn.Parameter(
                    torch.ones((self.num_heads * self.qk_head_dim, config.hidden_size))
                )
            )
        else:
            self.q_a_proj = MojoLinear(
                weight=nn.Parameter(
                    torch.ones((config.q_lora_rank, config.hidden_size))
                )
            )
            self.q_a_layernorm = MojoRMSNorm(
                eps=config.rms_norm_eps,
                norm_size=config.q_lora_rank,
                )
            self.q_b_proj = MojoLinear(
                weight=nn.Parameter(
                    torch.ones((self.num_heads * self.qk_head_dim, config.q_lora_rank))
                )
            )

        # Key-Value projection (with LoRA)
        self.kv_a_proj_with_mqa = MojoLinear(
            weight=nn.Parameter(
                torch.ones((
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    config.hidden_size,
                ))
            )
        )
        self.kv_a_layernorm = MojoRMSNorm(
            eps=config.rms_norm_eps,
            norm_size=self.kv_lora_rank,
            )
        self.kv_b_proj = MojoLinear(
            weight=nn.Parameter(
                torch.ones((
                    self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                    self.kv_lora_rank,
                ))
            )
        )

        # Output projection
        self.o_proj = MojoLinear(
            weight=nn.Parameter(
                torch.ones((config.hidden_size, self.num_heads * self.v_head_dim))
            )
        )

        # Attention operators
        self.scaling = self.qk_head_dim ** (-0.5)
        self.rope = MojoRoPE()
        #TODO MLA算子实现
        self.attn_prefill = MojoPagedPrefillMLA()
        self.attn_decode = MojoPagedDecodeMLA()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Attention forward pass.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Attention mask (optional)
            past_key_values: KV cache
            cache_position: Cache position indices
            use_cache: Whether to use KV cache
            
        Returns:
            Tuple of (attention_output, None)
        """
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # Get context lengths from cache
        context_lens = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
        )

        # Query projection
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Key-Value projection
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        
        # Apply RoPE
        cos, sin = position_embeddings
        q_rot, k_rot = self.rope(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        # Combine rotated and non-rotated parts
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is None:
            raise ValueError("Paged Attention requires a PagedDummyCache instance.")

        # Pad value states if needed
        if self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(
                value_states, [0, self.qk_head_dim - self.v_head_dim]
            )

        # Paged attention forward
        attn_output, _ = self.paged_attention_forward(
            query_states,
            key_states,
            value_states,
            past_key_values,
            context_lens,
        )

        # Remove padding if added
        if self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        # Output projection
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None

    def paged_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        past_key_values: PagedDummyCache,
        context_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Paged attention implementation for prefill and decode.
        
        Args:
            query_states: Query tensor [batch, num_heads, seq_len, head_dim]
            key_states: Key tensor [batch, num_heads, seq_len, head_dim]
            value_states: Value tensor [batch, num_heads, seq_len, head_dim]
            past_key_values: KV cache
            context_lens: Current context lengths per sequence
            
        Returns:
            Tuple of (attention_output, None)
        """
        batch_size, num_q_heads, seq_len, qk_head_dim = query_states.shape
        device = query_states.device

        # Update KV cache
        past_key_values.update(key_states, value_states, self.layer_idx)

        if seq_len > 1:  # Prefill phase
            # Prepare cumulative sequence lengths
            q_lens = torch.full(
                (batch_size,), seq_len, dtype=torch.int32, device=device
            )
            cu_seqlens_q = torch.cat([
                torch.tensor([0], device=device, dtype=torch.int32),
                q_lens.cumsum(0),
            ])
            total_tokens = cu_seqlens_q[-1].item()

            # Reshape queries for prefill attention
            q = query_states.permute(0, 2, 1, 3).reshape(
                total_tokens, num_q_heads, qk_head_dim
            )

            # Get cache and block tables
            k_cache = past_key_values.k_cache
            v_cache = past_key_values.v_cache
            block_tables = past_key_values.block_tables[self.layer_idx]

            # Prefill attention
            attn_output_tnd = self.attn_prefill(
                q, k_cache, v_cache, cu_seqlens_q, block_tables, self.scaling
            )
            
            # Reshape output
            attn_output = attn_output_tnd.reshape(
                batch_size, seq_len, num_q_heads, qk_head_dim
            )
            attn_output = attn_output.transpose(1, 2)

        else:  # Decode phase
            # Squeeze sequence dimension
            q = query_states.squeeze(2)
            
            # Get cache and block tables for decode
            k_cache, v_cache, block_tables = past_key_values.get_kv_for_decode(
                self.layer_idx
            )
            current_seq_lens = context_lens + 1

            # Decode attention
            attn_output_bhd = self.attn_decode(
                q, k_cache, v_cache, current_seq_lens, block_tables, self.scaling
            )
            attn_output = attn_output_bhd.unsqueeze(2)

        return attn_output, None


class DeepseekV3DecoderLayer(nn.Module):
    """Transformer decoder layer with attention and MLP/MoE."""
    
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        """
        Initialize decoder layer.
        
        Args:
            config: Model configuration
            layer_idx: Layer index
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)

        # Use MoE or dense MLP based on layer index
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

        # Layer normalization
        self.input_layernorm = MojoRMSNorm(
            eps=config.rms_norm_eps,
            norm_size=config.hidden_size,
        )
        self.post_attention_layernorm = MojoRMSNorm(
            eps=config.rms_norm_eps,
            norm_size=config.hidden_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Decoder layer forward pass.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_values: KV cache
            use_cache: Whether to use cache
            cache_position: Cache positions
            position_embeddings: RoPE embeddings
            
        Returns:
            Output hidden states
        """
        # Self-attention with residual connection
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

        # MLP/MoE with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class DeepseekV3Model(nn.Module):
    """DeepSeek V3 transformer model."""
    
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]

    def __init__(self, config: DeepseekV3Config):
        """
        Initialize model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Model components
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = MojoRMSNorm(
            eps=config.rms_norm_eps,
            norm_size=config.hidden_size,
        )
        self.rotary_emb = DeepseekV3RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, PagedDummyCache]:
        """
        Model forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_values: KV cache
            inputs_embeds: Input embeddings (if not using input_ids)
            cache_position: Cache positions
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (hidden_states, past_key_values)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Initialize cache if needed
        if past_key_values is None:
            past_key_values = PagedDummyCache(
                self.config,
                batch_size=batch_size,
                device=str(device),
                block_size=16,
            )

        # Compute position IDs
        past_len = int(past_key_values.get_seq_length(0).max().item())
        position_ids = torch.arange(
            past_len,
            past_len + seq_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)

        # Get embeddings and position encodings
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # Process through decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, past_key_values


class DeepseekV3ForCausalLM(nn.Module):
    """DeepSeek V3 model with language modeling head."""
    
    def __init__(self, config: DeepseekV3Config):
        """
        Initialize causal LM.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.model = DeepseekV3Model(config)
        self.lm_head = MojoLinear(
            weight=nn.Parameter(torch.ones((config.vocab_size, config.hidden_size)))
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedDummyCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, PagedDummyCache]:
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_values: KV cache
            use_cache: Whether to use cache
            cache_position: Cache positions
            
        Returns:
            Tuple of (logits, past_key_values)
            
        Example:
            >>> from transformers import AutoTokenizer
            >>> model = DeepseekV3ForCausalLM(config)
            >>> tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v3")
            >>> prompt = "Hello, how are you?"
            >>> inputs = tokenizer(prompt, return_tensors="pt")
            >>> logits, cache = model(**inputs)
        """
        # Get hidden states from base model
        hidden_states, past_key_values = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        return logits, past_key_values