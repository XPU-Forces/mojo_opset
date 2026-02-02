from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


def generate_pos_embs(
    sin: torch.Tensor,
    cos: torch.Tensor,
    kv_lens: torch.Tensor,
    seq_lens: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate position embeddings for different modes.

    Args:
        sin: Full sin tensor [1, max_seq_len, d]
        cos: Full cos tensor [1, max_seq_len, d]
        kv_lens: KV cache lengths [bs]
        seq_lens: Sequence lengths for prefill [bs], None for decode
        cu_seqlens: Cumulative sequence lengths for varlen

    Returns:
        (cos_embs, sin_embs): Position embeddings with appropriate shape
    """
    sin = sin.squeeze(0)
    cos = cos.squeeze(0)

    cos_embs = []
    sin_embs = []

    if cu_seqlens is not None:
        num_seqs = cu_seqlens.size(0) - 1
        seq_lens_from_cu = cu_seqlens[1:] - cu_seqlens[:-1]

        for i in range(num_seqs):
            qlen = seq_lens_from_cu[i].item()
            shift = kv_lens[i].item()
            cos_emb = cos[shift : shift + qlen]
            sin_emb = sin[shift : shift + qlen]
            cos_embs.append(cos_emb)
            sin_embs.append(sin_emb)

        cos_embs = torch.cat(cos_embs, dim=0)
        sin_embs = torch.cat(sin_embs, dim=0)

    elif seq_lens is not None:
        bsz = seq_lens.size(0)

        for i in range(bsz):
            qlen = seq_lens[i].item()
            shift = kv_lens[i].item()
            cos_emb = cos[shift : shift + qlen]
            sin_emb = sin[shift : shift + qlen]
            cos_embs.append(cos_emb)
            sin_embs.append(sin_emb)

        cos_embs = torch.stack(cos_embs, dim=0)
        sin_embs = torch.stack(sin_embs, dim=0)

    else:
        bsz = kv_lens.size(0)

        for i in range(bsz):
            shift = kv_lens[i].item()
            cos_emb = cos[shift : shift + 1]
            sin_emb = sin[shift : shift + 1]
            cos_embs.append(cos_emb)
            sin_embs.append(sin_emb)

        cos_embs = torch.stack(cos_embs, dim=0)
        sin_embs = torch.stack(sin_embs, dim=0)

    return cos_embs, sin_embs


class MojoRoPE(MojoOperator):
    def __init__(
        self,
        interleaved: bool = False,
    ):
        """
        Args:
            interleaved (bool, default=False): If True, use interleaved head layout when applying rotary.

        """
        super().__init__()

        assert interleaved == False, "interleaved impl is not supported yet."
        self.interleaved = interleaved

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings (RoPE) to queries and keys.

        Args:
            q (torch.Tensor): Query tensor [bs, n, seq, d] or [1, n, total_seq, d] for varlen
            k (torch.Tensor): Key tensor; same shape as `q`.
            cos (torch.Tensor): Precomputed cosine tensor [bs, seq, d] (train) or [1, max_seq, d] (others)
            sin (torch.Tensor): Precomputed sine tensor [bs, seq, d] (train) or [1, max_seq, d] (others)
            cu_seqlens (Optional[torch.Tensor], default=None): Cumulative sequence lengths for varlen
            kv_lens (Optional[torch.Tensor], default=None): KV cache lengths [bs]
            head_first (bool, default=True): If True, expect `q` and `k` in [bs, n, seq, d] layout.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(q_rot, k_rot)` with the same shape/dtype as inputs.
        """
        if cu_seqlens is not None:
            assert not head_first, "input must shaped [total_seq, n, d] for varlen."
        else:
            if not head_first:
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        if kv_lens is not None:
            seq_len = q.shape[2]

            if cu_seqlens is not None:
                cos, sin = generate_pos_embs(sin, cos, kv_lens, cu_seqlens=cu_seqlens)
            elif seq_len > 1:
                seq_lens = torch.full((q.shape[0],), seq_len, device=q.device, dtype=torch.long)
                cos, sin = generate_pos_embs(sin, cos, kv_lens, seq_lens=seq_lens)
            else:
                cos, sin = generate_pos_embs(sin, cos, kv_lens)

            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_rot = (q * cos + rotate_half(q) * sin).to(q.dtype)
        k_rot = (k * cos + rotate_half(k) * sin).to(k.dtype)

        return (
            q_rot,
            k_rot if head_first or cu_seqlens is not None else q_rot.transpose(1, 2).contiguous(),
            k_rot.transpose(1, 2).contiguous(),
        )


class MojoRoPEStoreKV(MojoOperator):
    pass


class MojoNormRoPE(MojoOperator):
    pass


class MojoNormRoPEStoreKV(MojoOperator):
    pass
