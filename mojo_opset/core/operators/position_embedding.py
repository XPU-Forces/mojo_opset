from typing import Optional
from typing import Tuple, List

import torch

from ..operator import MojoOperator


def generate_pos_embs(
    sin: torch.Tensor,
    cos: torch.Tensor,
    kv_lens: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract required position embeddings from full sin/cos tensors.

    Args:
        sin: Full sine embeddings [1, max_seq, d] or [max_seq, d]
        cos: Full cosine embeddings, same shape as sin
        kv_lens: KV cache lengths [bs]
        cu_seqlens: Cumulative sequence lengths for varlen scenario [bs+1]

    Returns:
        varlen: (cos_embs, sin_embs) shape [T, d]
        decode: (cos_embs, sin_embs) shape [B, d]
    """
    sin = sin.squeeze(0)
    cos = cos.squeeze(0)

    if cu_seqlens is not None:
        num_seqs = cu_seqlens.size(0) - 1
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        cos_embs, sin_embs = [], []
        for i in range(num_seqs):
            qlen = seq_lens[i].item()
            shift = kv_lens[i].item()
            cos_embs.append(cos[shift : shift + qlen])
            sin_embs.append(sin[shift : shift + qlen])
        return torch.cat(cos_embs, dim=0), torch.cat(sin_embs, dim=0)

    return cos[kv_lens], sin[kv_lens]


class MojoRoPE(MojoOperator):
    """Rotary Position Embedding (RoPE) operator.

    Supports three scenarios:
    1. Varlen prefill: input [T, N, D], cos/sin [max_seq, d].
    2. Padded prefill: input [B, S, N, D] or [B, N, S, D], cos/sin [S, d](already split).
    3. Decode: input [B, N, D], cos/sin [max_seq, d].
    """

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
        rope_percentage: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_dim = int(q.shape[-1] * rope_percentage)
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
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
        head_first: bool = True,
        rope_percentage: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embedding (RoPE).

        Scenario descriptions:
        1. Varlen prefill: q/k [T, N, D], requires cu_seqlens, sin/cos are full
        2. Padded prefill: q/k [B, S, N, D] or [B, N, S, D], sin/cos pre-sliced [B, S, d]
        3. Decode: q/k [B, N, D], requires kv_lens, sin/cos are full

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine position embeddings
            sin: Sine position embeddings
            cu_seqlens: Cumulative sequence lengths for varlen scenario
            kv_lens: Historical KV cache lengths(NOT include the tokens from the current decode step).
            head_first: True for padded input [B, N, S, D], False for [B, S, N, D], only used for padded input(as 'Scenario descriptions' above)
            rope_percentage: Percentage of head dim to apply RoPE (default: 1.0)

        Returns:
            (q_rot, k_rot) with same shape as input
        """
        # Varlen prefill: [T, N, D]
        if cu_seqlens is not None:
            num_seqs = cu_seqlens.size(0) - 1
            if kv_lens is None:
                kv_lens = torch.zeros(num_seqs, device=q.device, dtype=torch.long)
            cos, sin = generate_pos_embs(sin, cos, kv_lens, cu_seqlens=cu_seqlens)
            return self._apply_rope(q, k, cos.unsqueeze(1), sin.unsqueeze(1), rope_percentage)

        # Decode: [B, N, D]
        if q.dim() == 3:
            bsz = q.shape[0]
            if kv_lens is None:
                kv_lens = torch.zeros(bsz, device=q.device, dtype=torch.long)
            cos, sin = generate_pos_embs(sin, cos, kv_lens)
            return self._apply_rope(q, k, cos.unsqueeze(1), sin.unsqueeze(1), rope_percentage)

        # Padded prefill: [B, S, N, D] or [B, N, S, D]
        if head_first:
            return self._apply_rope(q, k, cos.unsqueeze(1), sin.unsqueeze(1), rope_percentage)
        return self._apply_rope(q, k, cos.unsqueeze(2), sin.unsqueeze(2), rope_percentage)


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


class MojoMRoPE(MojoOperator):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        assert rotary_dim % 2 == 0, "rotary_dim must be even for RoPE"
        if mrope_section is not None:
            assert len(mrope_section) == 3, "mrope_section must be [t, h, w]"
            assert sum(mrope_section) == rotary_dim, "sum(mrope_section) must equal rotary_dim"

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # 预计算cos/sin缓存
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, rotary_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)       # [max_pos, rotary_dim]
        cos = emb.cos()
        sin = emb.sin()
        self.register_buffer(
            "cos_sin_cache",
            torch.cat([cos, sin], dim=-1),
            persistent=False,
        )

    def _get_cos_sin(
        self,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (positions < self.max_position_embeddings).all(), "positions exceed max_position_embeddings"
        cos_sin = self.cos_sin_cache[positions]  # 1D: [num_tokens, 2*rotary_dim]; 2D: [3, num_tokens, 2*rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)      # 拆分cos/sin: 1D->[num_tokens, rotary_dim]; 2D->[3, num_tokens, rotary_dim]
        return cos, sin
    
    def apply_interleaved_mrope(self, x: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
        t_dim, h_dim, w_dim = mrope_section
        num_tokens = x.shape[0]
        
        x_t = x[..., :t_dim]    # [num_tokens, 3, t_dim] -> 仅保留t维度分配的长度
        x_h = x[..., t_dim:t_dim+h_dim]  # [num_tokens, 3, h_dim]
        x_w = x[..., t_dim+h_dim:]       # [num_tokens, 3, w_dim]
        
        x_t = x_t[:, 0, :]  # [num_tokens, t_dim] - 时间/序列维度
        x_h = x_h[:, 1, :]  # [num_tokens, h_dim] - 高度维度
        x_w = x_w[:, 2, :]  # [num_tokens, w_dim] - 宽度维度
        
        interleaved = torch.stack([
            x_t.reshape(num_tokens, -1, 1),
            x_h.reshape(num_tokens, -1, 1),
            x_w.reshape(num_tokens, -1, 1)
        ], dim=-1).flatten(-2)  # [num_tokens, rotary_dim]
        
        return interleaved

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """RoPE核心旋转操作：旋转最后一维的前半部分和后半部分"""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_mrope_cos_sin(
        self,
        cos: torch.Tensor,    # 输入形状: [num_tokens, 3, rotary_dim]
        sin: torch.Tensor,    # 输入形状: [num_tokens, 3, rotary_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用3D MRoPE的cos/sin维度重排"""
        t, h, w = self.mrope_section

        if self.mrope_interleaved:
            cos = self.apply_interleaved_mrope(cos, self.mrope_section)
            sin = self.apply_interleaved_mrope(sin, self.mrope_section)
        else:
            cos_t = cos[:, 0, :t]    # [num_tokens, t]
            cos_h = cos[:, 1, t:t+h] # [num_tokens, h]
            cos_w = cos[:, 2, t+h:]  # [num_tokens, w]
            cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)  # [num_tokens, rotary_dim]
            
            sin_t = sin[:, 0, :t]
            sin_h = sin[:, 1, t:t+h]
            sin_w = sin[:, 2, t+h:]
            sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)

        return cos, sin

    def forward(
        self,
        positions: torch.Tensor,        # 1D: [num_tokens]; 2D: [3, num_tokens]
        query: torch.Tensor,            # [batch_size, num_heads, num_tokens, head_size]
        key: torch.Tensor,              # [batch_size, num_heads, num_tokens, head_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        assert positions.ndim in (1, 2), "positions must be 1D (num_tokens) or 2D (3, num_tokens)"
        assert query.shape == key.shape, "query and key must have the same shape"
        assert query.shape[-1] == self.head_size, f"query head size must be {self.head_size}"
        assert self.rotary_dim <= self.head_size, "rotary_dim cannot exceed head_size"

        cos, sin = self._get_cos_sin(positions.to(self.cos_sin_cache.device, non_blocking=True))

        if positions.ndim == 2:
            assert self.mrope_section is not None, "mrope_section must be set for 2D positions (3D MRoPE)"
            cos = cos.transpose(0, 1)
            sin = sin.transpose(0, 1)
            cos, sin = self._apply_mrope_cos_sin(cos, sin)  # 输出: [num_tokens, rotary_dim]

        batch_size, num_heads, num_tokens, head_size = query.shape
        cos = cos.unsqueeze(0).unsqueeze(0)  # 扩展batch和head维度
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot, q_pass = query[..., :self.rotary_dim], query[..., self.rotary_dim:]
        k_rot, k_pass = key[..., :self.rotary_dim], key[..., self.rotary_dim:]

        q_rot = q_rot * cos + self._rotate_half(q_rot) * sin
        k_rot = k_rot * cos + self._rotate_half(k_rot) * sin

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        return q, k