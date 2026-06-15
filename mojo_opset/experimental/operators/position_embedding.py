import math
from typing import List
from typing import Optional

import torch

from mojo_opset.core.operator import MojoOperator


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


class MojoRotaryEmbedding(MojoOperator):
    """
    Apply RoPE to packed QKV tensors inplace.
    """

    def __init__(self, query_head_num: int, kv_head_num: int, rope_dim: int, interleaved: bool = False, **kwargs):
        """
        Args:
            query_head_num (int): Number of query heads packed at the front of
                ``qkv_input``.
            kv_head_num (int): Number of key heads and value heads. ``qkv_input``
                is expected to contain ``query_head_num + 2 * kv_head_num`` heads
                in ``[Q, K, V]`` order.
            rope_dim (int): Number of tail head channels to rotate.
            interleaved (bool): Whether to use interleaved RoPE rotation. When
                ``True``, implementations may consume ``interleave_offset`` in
                ``forward``.
            **kwargs: Tensor factory kwargs passed to ``MojoOperator``.
        """
        super().__init__(**kwargs)
        if query_head_num <= 0:
            raise ValueError(f"query_head_num must be positive, got {query_head_num}.")
        if kv_head_num <= 0:
            raise ValueError(f"kv_head_num must be positive, got {kv_head_num}.")
        if rope_dim <= 0:
            raise ValueError(f"rope_dim must be positive, got {rope_dim}.")
        self.query_head_num = int(query_head_num)
        self.kv_head_num = int(kv_head_num)
        self.rope_dim = int(rope_dim)
        self.interleaved = bool(interleaved)

    @staticmethod
    def _rotate(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        y = torch.empty_like(x)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        y[..., ::2] = -x2
        y[..., 1::2] = x1
        return y

    def _apply_rope_by_seq(
        self,
        tensor: torch.Tensor,
        sin_embeds: torch.Tensor,
        cos_embeds: torch.Tensor,
        kv_len: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        rope_dim: int,
    ) -> None:
        head_dim = tensor.size(-1)
        rope_offset = head_dim - rope_dim
        batch_size = kv_len.numel()

        for batch_idx in range(batch_size):
            start = int(cu_seq_lens[batch_idx].item())
            end = int(cu_seq_lens[batch_idx + 1].item())
            seq_len = end - start
            position = int(kv_len[batch_idx].item())
            x_rope = tensor[start:end, :, rope_offset : rope_offset + rope_dim].float()
            sin = sin_embeds[position : position + seq_len].float().unsqueeze(1)
            cos = cos_embeds[position : position + seq_len].float().unsqueeze(1)
            tensor[start:end, :, rope_offset : rope_offset + rope_dim] = (
                self._rotate(x_rope, self.interleaved) * sin + x_rope * cos
            ).to(tensor.dtype)

    def forward(
        self,
        qkv_input: torch.Tensor,
        sin_embeds: torch.Tensor,
        cos_embeds: torch.Tensor,
        kv_len: torch.Tensor,
        cu_seq_lens: Optional[torch.Tensor] = None,
        interleave_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply rotary position embedding to packed QKV.

        Args:
            qkv_input (torch.Tensor): Packed QKV tensor with shape
                ``[total_seq_len, (query_head_num + 2 * kv_head_num) * head_dim]``.
                Q heads come first, followed by K heads and then V heads. Q/K are
                rotated in the tail ``rope_dim`` channels of each head; V is
                preserved.
            sin_embeds (torch.Tensor): Sine table with shape
                ``[max_position_embeddings, rope_dim]``. Dtype should match
                ``qkv_input``.
            cos_embeds (torch.Tensor): Cosine table with the same shape and dtype
                contract as ``sin_embeds``.
            kv_len (torch.Tensor): Per-sequence starting position / existing KV
                length, shape ``[batch_size]``.
            cu_seq_lens (Optional[torch.Tensor]): Cumulative query lengths for
                packed varlen input, shape ``[batch_size + 1]``. The torch
                fallback currently requires it.
            interleave_offset (Optional[torch.Tensor]): Offset table used by the
                interleaved rotation path by implementations that need explicit
                offsets. It is not used by the torch fallback reference
                implementation.

        Returns:
            torch.Tensor: Packed QKV tensor with the same shape as ``qkv_input``.
            Implementations may return ``qkv_input`` updated in-place or a newly
            constructed tensor.
        """
        if not qkv_input.is_contiguous():
            raise NotImplementedError("Torch MojoRotaryEmbedding currently supports qkv_input is contiguous only.")
        if qkv_input.dim() != 2:
            raise NotImplementedError("Torch MojoRotaryEmbedding currently supports packed 2D qkv_input only.")
        if cu_seq_lens is None:
            raise NotImplementedError("Torch MojoRotaryEmbedding currently requires cu_seq_lens.")
        if sin_embeds.shape != cos_embeds.shape:
            raise ValueError(
                f"sin_embeds and cos_embeds must have same shape, got {tuple(sin_embeds.shape)} and {tuple(cos_embeds.shape)}."
            )
        if sin_embeds.dim() != 2:
            raise ValueError(f"sin_embeds must be 2D [max_position_embeddings, rope_dim], got {tuple(sin_embeds.shape)}.")
        if kv_len.dim() != 1:
            raise ValueError(f"kv_len must be 1D [batch_size], got {tuple(kv_len.shape)}.")
        if cu_seq_lens.dim() != 1 or cu_seq_lens.numel() != kv_len.numel() + 1:
            raise ValueError(
                f"cu_seq_lens must be 1D with size batch_size + 1, got shape {tuple(cu_seq_lens.shape)} for kv_len {tuple(kv_len.shape)}."
            )

        if sin_embeds.size(-1) != self.rope_dim:
            raise ValueError(f"sin_embeds last dim must match rope_dim={self.rope_dim}, got {sin_embeds.size(-1)}.")
        total_head_num = self.query_head_num + 2 * self.kv_head_num
        if qkv_input.size(-1) % total_head_num != 0:
            raise ValueError(
                f"qkv hidden dim {qkv_input.size(-1)} must be divisible by total head count {total_head_num}."
            )
        head_dim = qkv_input.size(-1) // total_head_num
        if head_dim < self.rope_dim:
            raise ValueError(f"head_dim must be >= rope_dim, got head_dim={head_dim}, rope_dim={self.rope_dim}.")

        qkv = qkv_input.view(qkv_input.size(0), total_head_num, head_dim)
        q = qkv[:, : self.query_head_num, :]
        k = qkv[:, self.query_head_num : self.query_head_num + self.kv_head_num, :]

        self._apply_rope_by_seq(q, sin_embeds, cos_embeds, kv_len, cu_seq_lens, self.rope_dim)
        self._apply_rope_by_seq(k, sin_embeds, cos_embeds, kv_len, cu_seq_lens, self.rope_dim)
        return qkv_input


__all__ = [
    "MojoRelativeEmbedding",
    "MojoGridRoPE",
    "MojoRotaryEmbedding",
]
