from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels.ascend.linear_attn.modules.l2norm import l2norm_bwd
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.modules.l2norm import l2norm_fwd
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd
from mojo_opset.backends.ttx.kernels.ascend.linear_attn.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from mojo_opset.backends.ttx.kernels.ascend.utils import input_guard
from mojo_opset.core import MojoGatedDeltaRule
from mojo_opset.core import MojoGatedDeltaRuleFunction
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


class TTXGatedDeltaRuleFunction(MojoGatedDeltaRuleFunction):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        # FIXME: tempory hard code.
        initial_state = None
        output_final_state = False
        chunk_indices = None
        chunk_offsets = None
        chunk_size = 16
        enable_gqa = True

        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, w, u, h, v_new, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, beta, A, w, u, h, v_new, initial_state, cu_seqlens, chunk_indices, chunk_offsets
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.enable_gqa = enable_gqa
        ctx.chunk_size = chunk_size
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor = None):
        (
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta,
            A,
            w,
            u,
            h,
            v_new,
            initial_state,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
        ) = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            w=w,
            u=u,
            h=h,
            v_new=v_new,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            enable_gqa=ctx.enable_gqa,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            None,
            None,
        )
