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
