from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from mojo_opset.core import MojoGatedDeltaRule
from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA


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

        output = torch.ops.ttx.paged_attention_prefill(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            block_tables=block_tables,
            sm_scale=softmax_scale,
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

        output = torch.ops.ttx.paged_attention_decode(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            seqlens=seqlens,
            block_tables=block_tables,
            sm_scale=softmax_scale,
        )

        return output
class TTXGatedDeltaRule(MojoGatedDeltaRule, default_priority=2):
    def forward_std(self, q, k, v, g, beta, cu_seqlens=None, scale=None):
        o_var, ht_var = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
        )

        return o_var
