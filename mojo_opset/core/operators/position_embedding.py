import math
from typing import Optional
from typing import Tuple, List

import torch

from ..operator import MojoOperator


class MojoRelativeEmbedding(MojoOperator):
    def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128):
        """
        Initialize T5-style relative position embedding.

        Args:
            num_buckets (int): Number of relative position buckets.
            num_heads (int): Attention heads; also the embedding output channels.
            bidirectional (bool): If True, allocate half buckets for positive direction.
            max_dist (int, default=128): Maximum distance used in logarithmic bucketing.
        """
        super().__init__()
        if not isinstance(num_buckets, int) or num_buckets <= 0:
            raise ValueError("num_buckets must be a positive integer")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(bidirectional, bool):
            raise TypeError("bidirectional must be a bool")
        if not isinstance(max_dist, int) or max_dist <= 0:
            raise ValueError("max_dist must be a positive integer")
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = torch.nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        """
        Compute relative position bias tensor for attention.

        Args:
            lq (int): Length of query sequence (Lq).
            lk (int): Length of key/value sequence (Lk).

        Returns:
            torch.Tensor: Bias tensor of shape [1, num_heads, Lq, Lk], dtype follows embedding weights.
        """
        if not isinstance(lq, int) or not isinstance(lk, int) or lq <= 0 or lk <= 0:
            raise ValueError("lq and lk must be positive integers")
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets

    def extra_repr(self) -> str:
        return f"{self.num_buckets=}, {self.num_heads=}, {self.bidirectional=}, {self.max_dist=}".replace("self.", "")


class MojoRotaryEmbedding(MojoOperator):
    def __init__(self, rope_theta, rope_dim, attention_scaling: float = 1.0, init_max_length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device = self.tensor_factory_kwargs.get("device")) / rope_dim)
        )
        self.attention_scaling = attention_scaling
        self.init_max_length = None
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if init_max_length is not None:
            self._rope_init(init_max_length)

        def load_state_dict_post_hook(module, incompatible_keys) -> None:
            del module
            key2ignore = []
            for miss in incompatible_keys.missing_keys:
                if miss.split('.')[-1] in ("inv_freq", "cos", "sin"):
                    key2ignore.append(miss)
            for key in key2ignore:
                incompatible_keys.missing_keys.remove(key)
        self.register_load_state_dict_post_hook(load_state_dict_post_hook)


    def _rope_init(self, max_length: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.init_max_length = max_length
        position_ids = torch.arange(max_length, device = self.tensor_factory_kwargs.get("device"))
        freqs = position_ids[..., None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        cu_q_lens: Optional[torch.Tensor] = None,
        total_seq_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cos/sin for Rotary Position Embedding (RoPE).
        x is necessary for the kernel to determine the output shape.

        Scenario descriptions:
        1. Varlen prefill: input [T, H], cu_q_lens [T+1] or position_ids [T] -> cos/sin [T, D].
        2. Padded prefill: input [B, S, H], cu_q_lens None, position_ids None -> cos/sin [S, D].
        3. Decode: input [B, H], cu_q_lens None, position_ids [B] -> cos/sin [B, D].
        """
        if cu_q_lens is not None:
            assert cu_q_lens.dtype == torch.int32
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        if position_ids is not None:
            assert position_ids.dtype == torch.int32
        assert position_ids is None or cu_q_lens is None, "At most one of cu_q_lens or position_ids should be provided"

        if cu_q_lens is not None:
            assert x.dim() == 2, "x must be 2D: [T, D]"
            position_ids = torch.full((x.shape[0],), -1, device = x.device, dtype = torch.int32)
            q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
            bsz = q_lens.size(0)
            for i in range(bsz):
                q_len = q_lens[i].item()
                context_len = 0 if total_seq_lens is None else total_seq_lens[i].item() - q_len
                position_ids[cu_q_lens[i]:cu_q_lens[i+1]] = torch.arange(
                    context_len,
                    context_len + q_len, 
                    device = cu_q_lens.device,
                    dtype = torch.int32,
                )
        elif position_ids is not None:
            assert position_ids.shape == x.shape[:-1], "position_ids must have the same shape as x except the hidden dimension"
        else:
            position_ids = torch.arange(x.shape[1], device = x.device, dtype = torch.int32)

        if self.init_max_length is None:
            freqs = position_ids[..., None] * self.inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        else:
            cos = self.cos[position_ids]
            sin = self.sin[position_ids]
        
        return cos, sin


class MojoApplyRoPE(MojoOperator):

    def __init__(self, interleaved: bool = False):
        super().__init__()
        assert not interleaved, "interleaved impl is not supported yet."
        self.interleaved = interleaved

    def extra_repr(self) -> str:
        return f"{self.interleaved=}".replace("self.", "")

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_dim = cos.shape[-1]
        nope_dim = q.shape[-1] - rope_dim

        if nope_dim > 0:
            q_nope, q = torch.split(q, [nope_dim, rope_dim], dim=-1)
            k_nope, k = torch.split(k, [nope_dim, rope_dim], dim=-1)

        q_rot = (q * cos + self._rotate_half(q) * sin).to(q.dtype)
        k_rot = (k * cos + self._rotate_half(k) * sin).to(k.dtype)

        if nope_dim > 0:
            q_rot = torch.cat([q_nope, q_rot], dim=-1)
            k_rot = torch.cat([k_nope, k_rot], dim=-1)

        return q_rot, k_rot

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embedding (RoPE).

        Scenario descriptions:
        1. Varlen prefill: q/k [T, N, D] or [N, T, D], cos/sin [T, d]
        2. Padded prefill: q/k [B, S, N, D] or [B, N, S, D], cos/sin [S, d] or [B, S, d]
        3. Decode: q/k [B, N, D] or [N, B, D], cos/sin [B, d]

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine position embeddings
            sin: Sine position embeddings
            unsqueeze_dim: Unsqueeze dimension for cos and sin for multi-heads

        Returns:
            (q_rot, k_rot) with same shape as input
        """
        assert q.ndim == k.ndim, "q and k must have the same dimension"
        assert q.ndim == 3 or q.ndim == 4, "q and k must be 3D or 4D"
        assert cos.shape == sin.shape, "cos and sin must have the same shape"
        if q.ndim == 3:
            assert cos.ndim == 2, "rotary position embedding (cos/sin) must be of shape [num_tokens, rope_dim] for varlen prefill or decode"
        
        if head_first:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        else:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        return self._apply_rope(q, k, cos, sin)


class MojoRoPEStoreKV(MojoOperator):
    pass


class MojoNormRoPE(MojoOperator):
    pass


class MojoNormRoPEStoreKV(MojoOperator):
    pass


class MojoGridRoPE(MojoOperator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply 3D grid rotary position embeddings (RoPE) over (F, H, W) axes using
        precomputed per-sample frequency tensors.

        Args:
            x (torch.Tensor): [B, L, N, D]; D must be even (paired into complex components).
            grid_sizes (torch.Tensor): [B, 3] per-sample (F, H, W); seq_len = F*H*W.
            freqs_list (List[torch.Tensor]): length-B list; each item is a complex unit-phase tensor
                of shape [seq_len, 1, D/2], broadcastable to [seq_len, N, D/2].

        Returns:
            torch.Tensor: Same shape as `x`. Per sample, the first F*H*W tokens are rotated;
                remaining padding tokens are preserved. Output dtype matches input.
        """
        assert x.dim() == 4, "x must be 4D: [B, L, N, D]"
        assert x.size(-1) % 2 == 0, "D must be even for complex pairing"
        assert grid_sizes.dim() == 2 and grid_sizes.size(1) == 3, "grid_sizes must be [B, 3]"

        n = x.size(2)
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
            freqs_i = freqs_list[i]
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        y = torch.stack(output)
        return y.type_as(x)


class MojoVisionRotaryEmbedding2D(MojoOperator):
    def __init__(
        self,
        rope_theta: float = 10000.0,
        rope_dim: int = 64,
        adapooling_factor: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # The native position regrouping assumes a positive adapooling window.
        assert adapooling_factor >= 1, "adapooling_factor must be >= 1"
        assert rope_dim % 4 == 0, "vision 2D rope_dim must be divisible by 4"
        self.rope_theta = rope_theta
        self.rope_dim = rope_dim
        self.adapooling_factor = adapooling_factor
        rotary_dim = self.rope_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=self.tensor_factory_kwargs.get("device"))
                / rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def extra_repr(self) -> str:
        return (
            f"rope_theta={self.rope_theta}, rope_dim={self.rope_dim}, "
            f"adapooling_factor={self.adapooling_factor}"
        )

    def _vision_rotary_embedding(self, seqlen: int, device: torch.device) -> torch.Tensor:
        inv_freq = self.inv_freq.to(device=device)
        seq = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
        return torch.outer(seq, inv_freq)

    def _build_position_ids(self, grid_hw: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert grid_hw.ndim == 2 and grid_hw.size(-1) == 2, "grid_hw must be [B, 2]"
        assert not torch.is_floating_point(grid_hw), "grid_hw must be an integer tensor"

        pos_ids = []
        grid_hw_cpu = grid_hw.to(device="cpu", dtype=torch.int64)
        for gh, gw in grid_hw_cpu.tolist():
            assert gh > 0 and gw > 0, "grid height/width must be positive"
            assert gh % self.adapooling_factor == 0 and gw % self.adapooling_factor == 0, (
                "grid height/width must be divisible by adapooling_factor"
            )
            # Match the native rotary regrouping: positions are first regrouped by the
            # adapooling window before being flattened back to patch order.
            hpos_ids = torch.arange(gh, device=device).unsqueeze(1).expand(-1, gw)
            hpos_ids = hpos_ids.reshape(
                gh // self.adapooling_factor,
                self.adapooling_factor,
                gw // self.adapooling_factor,
                self.adapooling_factor,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(gw, device=device).unsqueeze(0).expand(gh, -1)
            wpos_ids = wpos_ids.reshape(
                gh // self.adapooling_factor,
                self.adapooling_factor,
                gw // self.adapooling_factor,
                self.adapooling_factor,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            sample_pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
            pos_ids.append(sample_pos_ids)

        return torch.cat(pos_ids, dim=0)

    def forward(self, grid_hw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rebuild the native frequency lookup in two steps:
        # 1. create one 1D rotary table up to the largest grid extent,
        # 2. gather H/W pairs using the adapooling-aware patch order.
        device = self.inv_freq.device if self.inv_freq.device.type != "cpu" or grid_hw.device.type == "cpu" else grid_hw.device
        grid_hw_cpu = grid_hw.to(device="cpu", dtype=torch.int64)
        max_grid_size = int(grid_hw_cpu.max().item())
        rotary_pos_emb_full = self._vision_rotary_embedding(max_grid_size, device=device)
        pos_ids = self._build_position_ids(grid_hw, device=device)
        freqs = rotary_pos_emb_full[pos_ids].flatten(-2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


class MojoApplyVisionRoPE2D(MojoOperator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        assert x.ndim == 3, "vision rope expects packed token-first tensors [T, H, D]"
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        output = (x * cos) + (self._rotate_half(x) * sin)
        return output.to(orig_dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply native vision 2D RoPE to packed token-first tensors with prebuilt cos/sin.

        Supported layout:
        1. Packed varlen: q/k [T, N, D], cos/sin [T, D]
        """
        assert q.ndim == k.ndim, "q and k must have the same dimension"
        assert q.ndim == 3, "q and k must be 3D packed token-first tensors"
        assert q.shape[-1] == k.shape[-1], "q and k must have the same head_dim"
        assert cos.shape == sin.shape, "cos and sin must have the same shape"
        assert cos.ndim == 2, "vision rotary embedding (cos/sin) must be of shape [num_tokens, rope_dim]"
        assert q.shape[0] == cos.shape[0], "packed token count must match cos/sin sequence length"
        assert k.shape[0] == cos.shape[0], "packed token count must match cos/sin sequence length"
        assert q.shape[-1] == cos.shape[-1], "vision rope rotates the full head_dim"
        assert k.shape[-1] == cos.shape[-1], "vision rope rotates the full head_dim"
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)
