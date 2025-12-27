from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoDecodeDSA(MojoOperator):
    def __init__(self, softmax_scale: float = None):
        self.softmax_scale = softmax_scale

    def forward_std(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        wkv_b: torch.Tensor,  # [kv_lora_rank, n_head * (qk_nope_head_dim + v_head_dim)] e.g. --> [512, 16 * (128 + 128)]
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_ref(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        wkv_b: torch.Tensor,  # [kv_lora_rank, n_head * (qk_nope_head_dim + v_head_dim)] e.g. --> [512, 16 * (128 + 128)]
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        bsz, _, n_heads, qk_nope_head_dim = q_nope.shape
        qk_rope_head_dim = q_pe.shape[-1]
        kv_lora_rank = kv_cache.shape[-1]
        v_head_dim = wkv_b.shape[0] // n_heads - qk_nope_head_dim
        end_pos = start_pos + 1

        q_nope = q_nope.to(torch.float32)
        q_pe = q_pe.to(torch.float32)
        kv_cache = kv_cache.to(torch.float32)
        pe_cache = pe_cache.to(torch.float32)
        wkv_b = wkv_b.to(torch.float32)

        if self.softmax_scale is None:
            self.softmax_scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

        wkv_b = wkv_b.view(n_heads, -1, kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])
        scores = (
            torch.einsum("bshc,btc->bsht", q_nope, kv_cache[:bsz, :end_pos])
            + torch.einsum("bshr,btr->bsht", q_pe, pe_cache[:bsz, :end_pos])
        ) * self.softmax_scale

        index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=scores.device).scatter_(-1, topk_indices, 0)
        scores += index_mask.unsqueeze(2)

        scores = scores.softmax(dim=-1)
        o = torch.einsum("bsht,btc->bshc", scores, kv_cache[:bsz, :end_pos])
        o = torch.einsum("bshc,hdc->bshd", o, wkv_b[:, -v_head_dim:])

        return o

    def forward_analysis(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        wkv_b: torch.Tensor,  # [kv_lora_rank, n_head * (qk_nope_head_dim + v_head_dim)] e.g. --> [512, 16 * (128 + 128)]
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        start_pos: int,
    ) -> Tuple[int, int, int]:
        pass


class MojoPagedDecodeDSA(MojoOperator):
    def __init__(self, is_causal: bool = True, softmax_scale: float = None):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale
