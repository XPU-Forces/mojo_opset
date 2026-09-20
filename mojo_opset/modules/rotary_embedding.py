import torch

from mojo_opset import functions


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self, rope_theta, rope_dim, attention_scaling=1.0, init_max_length=None, *, implementation=None, device=None
    ):
        super().__init__()
        self.rope_theta, self.attention_scaling = rope_theta, attention_scaling
        self.init_max_length, self.implementation = init_max_length, implementation
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        if init_max_length is not None:
            self._rope_init(init_max_length)

    def _rope_init(self, max_length):
        self.init_max_length = max_length
        freqs = torch.arange(max_length, device=self.inv_freq.device)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos, self.sin = emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling

    def forward(self, x, cu_q_lens=None, total_seq_lens=None, position_ids=None):
        return functions.rotary_embedding(
            x,
            self.inv_freq,
            self.cos,
            self.sin,
            cu_q_lens,
            total_seq_lens,
            position_ids,
            self.attention_scaling,
            implementation=self.implementation,
        )


class VisionRotaryEmbedding2D(torch.nn.Module):
    def __init__(self, rope_theta=10000.0, rope_dim=64, adapooling_factor=1, *, implementation=None, device=None):
        super().__init__()
        if adapooling_factor < 1 or rope_dim % 4:
            raise ValueError("adapooling_factor must be positive and rope_dim must be divisible by 4")
        self.rope_theta, self.rope_dim, self.adapooling_factor = rope_theta, rope_dim, adapooling_factor
        self.implementation = implementation
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, rope_dim // 2, 2, dtype=torch.float32, device=device) / (rope_dim // 2))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, grid_hw):
        return functions.vision_rotary_embedding2d(
            self.inv_freq, grid_hw, self.rope_dim, self.adapooling_factor, implementation=self.implementation
        )
