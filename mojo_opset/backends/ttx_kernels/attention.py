import torch
import math
from typing import Optional, Tuple, Any

from mojo_opset.backends.ttx_kernels.src.ascend.paged_attention import (
    ttx_paged_attention_prefill,
    ttx_paged_attention_decode,
)

from mojo_opset.backends.ttx_kernels.src.ascend.flash_attention import (
    ttx_flash_attention_fwd,
    ttx_flash_attention_bwd,
)

from mojo_opset.core import MojoPagedDecodeGQA, MojoPagedPrefillGQA, MojoFlashAttnFunction


class TTXPagedPrefillGQA(MojoPagedPrefillGQA, default_priority=2):
    def forward_std(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ):
        assert self.window_size == -1, (
            f"[TTXPagedPrefillGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        )
        assert self.gqa_layout == "ABAB", (
            f"[TTXPagedPrefillGQA] TTX only support ABAB layout, but got gqa_layout={self.gqa_layout}"
        )
        assert self.is_causal, (
            f"[TTXPagedPrefillGQA] TTX only support causal attention, but got is_causal={self.is_causal}"
        )

        output = ttx_paged_attention_prefill(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            block_tables=block_tables,
            softmax_scale=softmax_scale,
        )

        return output


class TTXPagedDecodeGQA(MojoPagedDecodeGQA, default_priority=2):
    def forward_std(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ):
        assert self.window_size == -1, (
            f"[TTXPagedPrefillGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        )
        assert self.gqa_layout == "ABAB", (
            f"[TTXPagedPrefillGQA] TTX only support ABAB layout, but got gqa_layout={self.gqa_layout}"
        )
        assert self.is_causal, (
            f"[TTXPagedPrefillGQA] TTX only support causal attention, but got is_causal={self.is_causal}"
        )

        output = ttx_paged_attention_decode(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            seqlens=seqlens,
            block_tables=block_tables,
            softmax_scale=softmax_scale,
        )

        return output


class TTXFlashAttnFunction(MojoFlashAttnFunction):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=0.0,
        causal=False,
        softmax_scale=None,
    ):
        gqa_interleave = False  # FIXME: temporary hard code.
        assert dropout_p == 0.0, f"[TTXFlashAttnFunction] TTX does not support dropout, but got dropout_p={dropout_p}"

        if softmax_scale is None:
            softmax_scale = 1 / math.sqrt(q.shape[-1])

        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max()
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max()

        o, lse = ttx_flash_attention_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            softmax_scale,
            gqa_interleave,
        )
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.gqa_interleave = gqa_interleave
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        softmax_scale = ctx.softmax_scale
        causal = ctx.causal
        gqa_interleave = ctx.gqa_interleave
        dq, dk, dv = ttx_flash_attention_bwd(
            o,
            grad_o,
            lse,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            softmax_scale,
            gqa_interleave,
        )
        return dq, dk, dv, None, None, None, None, None, None, None
