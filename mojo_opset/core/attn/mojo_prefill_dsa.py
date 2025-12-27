from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoPrefillDSA(MojoOperator):
    def __init__(self, is_causal: bool = True, softmax_scale: float = None):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale

    def forward_std(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_ref(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, n_heads, qk_head_dim = q.shape
        q_fp32 = q.to(torch.float32).transpose(1, 2).contiguous()  # [bs, head_num, seq, dim]
        k_fp32 = k.to(torch.float32).transpose(1, 2).contiguous()  # [bs, head_num, seq, dim]
        v_fp32 = v.to(torch.float32).transpose(1, 2).contiguous()  # [bs, head_num, seq, dim]

        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / (qk_head_dim**0.5)

        scores = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * self.softmax_scale

        causal_mask = torch.tril(torch.ones((seqlen, seqlen), device=q.device, dtype=torch.bool))
        sparse_mask = torch.zeros((bsz, seqlen, seqlen), device=q.device, dtype=torch.bool)
        sparse_mask.scatter_(2, topk_indices, True)
        combined_mask = causal_mask.unsqueeze(0) & sparse_mask
        combined_mask = combined_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)

        scores.masked_fill_(~combined_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1).to(v_fp32.dtype)
        output = torch.matmul(attn_weights, v_fp32)
        output = output.to(q.dtype).permute(0, 2, 1, 3)  # [bs, seq, head_num, dim]
        output = torch.where(output != output, 0.0, output)

        return output

    def forward_analysis(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> Tuple[int, int, int]:
        pass


class MojoPagedPrefillDSA(MojoOperator):
    def __init__(self, is_causal: bool = True, softmax_scale: float = None):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale
