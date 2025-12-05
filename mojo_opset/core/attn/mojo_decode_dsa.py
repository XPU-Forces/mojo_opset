from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoDecodeDSA(MojoOperator):
    def __init__(self, softmax_scale: float = None):
        self.softmax_scale = softmax_scale

    def forward_std(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_ref(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        bsz, n_heads, _, head_dim = q.shape
        max_seq_len = k_cache.shape[2]
        end_pos = start_pos + 1

        q = q.permute(0, 2, 1, 3)  # [bs, head_num, seq, dim]
        k_cache = k_cache.permute(0, 2, 1, 3)  # [bs, head_num, seq, dim]
        v_cache = v_cache.permute(0, 2, 1, 3)  # [bs, head_num, seq, dim]

        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / (head_dim**0.5)

        scores = torch.full((bsz, n_heads, 1, max_seq_len), float("-inf"), device=q.device, dtype=torch.float32)
        k_causal = k_cache[:, :, :end_pos, :]
        q_float = q.float()
        scores_causal = torch.matmul(q_float, k_causal.float().transpose(-1, -2)) * self.softmax_scale
        scores[:, :, :, :end_pos] = scores_causal
        squeezed_indices = topk_indices.squeeze(1)
        for i in range(bsz):
            for h in range(n_heads):
                k_sparse = k_cache[i, h, squeezed_indices[i], :]
                q_vec = q[i, h, :, :].float()
                scores_sparse = torch.matmul(q_vec, k_sparse.float().transpose(0, 1)) * self.softmax_scale
                scores[i, h, 0, squeezed_indices[i]] = torch.max(
                    scores[i, h, 0, squeezed_indices[i]], scores_sparse.squeeze(0)
                )
        attn_weights = torch.softmax(scores, dim=-1).to(q.dtype)
        output = torch.matmul(attn_weights, v_cache)
        return output.permute(0, 2, 1, 3)

    def forward_analysis(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        start_pos: int,
    ) -> Tuple[int, int, int]:
        pass


class MojoPagedDecodeDSA(MojoOperator):
    def __init__(self, is_causal: bool = True, softmax_scale: float = None):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale
