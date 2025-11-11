import torch
from torch import nn
import torch_npu
from mojo_opset import MojoRoPE, MojoNorm, MojoSwiGLU, MojoPagedPrefillGQA, MojoPagedDecodeGQA

# Example: Use Mojo ops APIs directly for modeling — simply import the required modules and plug them in as needed.
mojo_apply_rope = MojoRoPE()
MojoRMSNorm = MojoNorm
mojo_paged_attention_prefill = MojoPagedPrefillGQA()
mojo_paged_attention_decode = MojoPagedDecodeGQA()


class MojoSwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.hidden_act != "silu":
            raise ValueError(f"MojoSwiGLUMLP requires 'silu' activation, but got {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)

        silu = MojoSwiGLU()
        fused_output = silu(gate_output, up_output)

        return self.down_proj(fused_output)


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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def paged_attention_forward(
    module: "Qwen3Attention",
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
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

        attn_output_tnd = mojo_paged_attention_prefill(
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

        attn_output_bhd = mojo_paged_attention_decode(
            q, k_cache, v_cache, current_seq_lens, block_tables, module.scaling
        )
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
            MojoRMSNorm(eps=config.rms_norm_eps, gamma=self.gamma) if hasattr(config, "q_norm") else nn.Identity()
        )
        self.k_norm = (
            MojoRMSNorm(eps=config.rms_norm_eps, gamma=self.gamma) if hasattr(config, "k_norm") else nn.Identity()
        )

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, use_cache, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        context_lens = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else torch.zeros(bsz, dtype=torch.long, device=hidden_states.device)
        )

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = mojo_rope(query_states, key_states, cos, sin)

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
        self.mlp = MojoSwiGLUMLP(config)
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.input_layernorm = MojoRMSNorm(config.rms_norm_eps, gamma=self.gamma)
        self.post_attention_layernorm = MojoRMSNorm(config.rms_norm_eps, gamma=self.gamma)
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
