import math
from typing import Optional, Tuple, List

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


class MojoMRoPEInplace(MojoOperator):
    """Multimodal Rotary Position Embedding (MRoPE) for Qwen2-VL.

    Applies 3D rotary position embedding over temporal (T), height (H), and width (W)
    dimensions to query and key tensors. Supports both interleaved and non-interleaved modes.

    Reference: https://qwenlm.github.io/blog/qwen2-vl/
    """

    supported_platforms_list = ["npu", "mlu", "meta_device", "ilu"]

    def __init__(self, inplace: bool = False, **kwargs):
        """
        Args:

            inplace (bool, default=False): Whether to perform MRoPE in-place on the input tensor.
        """
        super().__init__(**kwargs)
        self.inplace = inplace

    def extra_repr(self) -> str:
        return ""

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        hidden_size = hidden_states.shape[-1]
        hidden_states_half = hidden_size // 2
        left = hidden_states[..., :hidden_states_half]
        right = hidden_states[..., hidden_states_half:]
        return torch.cat((-right, left), dim=-1)

    @staticmethod
    def _apply_interleaved_mrope(
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply interleaved MRoPE pattern to cos/sin tables."""
        cos_interleaved = cos_table[0].clone()
        cos_interleaved[..., 1 : mrope_section[1] * 3 : 3] = cos_table[1, ..., 1 : mrope_section[1] * 3 : 3]
        cos_interleaved[..., 2 : mrope_section[2] * 3 : 3] = cos_table[2, ..., 2 : mrope_section[2] * 3 : 3]

        sin_interleaved = sin_table[0].clone()
        sin_interleaved[..., 1 : mrope_section[1] * 3 : 3] = sin_table[1, ..., 1 : mrope_section[1] * 3 : 3]
        sin_interleaved[..., 2 : mrope_section[2] * 3 : 3] = sin_table[2, ..., 2 : mrope_section[2] * 3 : 3]

        return cos_interleaved, sin_interleaved

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool = False,
        head_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_query = query
        orig_key = key

        num_tokens, n_qh_head_dim = query.shape
        num_tokens_k, n_kh_head_dim = key.shape

        rope_dim = sum(mrope_section) * 2
        half_rope_dim = rope_dim // 2

        if head_dim is None:
            head_dim = rope_dim

        n_qh = n_qh_head_dim // head_dim
        n_kh = n_kh_head_dim // head_dim

        query = query.view(num_tokens, n_qh, head_dim)
        key = key.view(num_tokens_k, n_kh, head_dim)

        query_rot, query_pass = query.split([rope_dim, head_dim - rope_dim], dim=-1)
        key_rot, key_pass = key.split([rope_dim, head_dim - rope_dim], dim=-1)

        if cos_table.dim() == 3:
            if is_interleaved:
                cos_table, sin_table = self._apply_interleaved_mrope(cos_table, sin_table, mrope_section)
            else:
                cos_table = torch.cat([m[i] for i, m in enumerate(cos_table.split(mrope_section, dim=-1))], dim=-1)
                sin_table = torch.cat([m[i] for i, m in enumerate(sin_table.split(mrope_section, dim=-1))], dim=-1)

        cos_table = cos_table.view(num_tokens, half_rope_dim)
        sin_table = sin_table.view(num_tokens, half_rope_dim)

        query_rot_half1 = query_rot[..., :half_rope_dim]
        query_rot_half2 = query_rot[..., half_rope_dim:]
        key_rot_half1 = key_rot[..., :half_rope_dim]
        key_rot_half2 = key_rot[..., half_rope_dim:]

        cos_expanded = cos_table.unsqueeze(1)
        sin_expanded = sin_table.unsqueeze(1)

        query_rot_new_half1 = query_rot_half1 * cos_expanded - query_rot_half2 * sin_expanded
        query_rot_new_half2 = query_rot_half2 * cos_expanded + query_rot_half1 * sin_expanded
        key_rot_new_half1 = key_rot_half1 * cos_expanded - key_rot_half2 * sin_expanded
        key_rot_new_half2 = key_rot_half2 * cos_expanded + key_rot_half1 * sin_expanded

        query_rot = torch.cat([query_rot_new_half1, query_rot_new_half2], dim=-1)
        key_rot = torch.cat([key_rot_new_half1, key_rot_new_half2], dim=-1)

        query = torch.cat([query_rot, query_pass], dim=-1).view(num_tokens, -1)
        key = torch.cat([key_rot, key_pass], dim=-1).view(num_tokens_k, -1)

        # 根据 self.inplace 决定是否原地修改并返回原始张量
        if self.inplace:
            orig_query.copy_(query)   # 将计算结果复制回原始输入
            orig_key.copy_(key)
            return orig_query, orig_key
        else:
            return query, key

__all__ = [
    "MojoRelativeEmbedding",
    "MojoGridRoPE",
    "MojoMRoPEInplace",
]
