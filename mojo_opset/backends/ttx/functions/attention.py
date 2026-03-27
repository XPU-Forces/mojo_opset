import torch
from typing import Optional, Tuple

from mojo_opset.backends.ttx.kernels import diffusion_attention_bwd
from mojo_opset.backends.ttx.kernels import diffusion_attention_fwd
from mojo_opset.backends.ttx.kernels import swa_fwd
from mojo_opset.backends.ttx.kernels import swa_bwd
from mojo_opset.experimental import MojoDiffusionAttentionFunction
from mojo_opset.core import MojoSWAFunction


class TTXDiffusionAttentionFunction(MojoDiffusionAttentionFunction):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        scale: float = 1.0,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa
        output, output_fp32, lse = diffusion_attention_fwd(
            query,
            key,
            value,
            mask,
            scale,
            enable_gqa,
        )
        ctx.save_for_backward(query, key, value, mask, output_fp32, lse)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value, mask, output_fp32, lse = ctx.saved_tensors
        dq, dk, dv = diffusion_attention_bwd(
            output_fp32,
            grad_output,
            query,
            key,
            value,
            lse,
            mask,
            ctx.scale,
            ctx.enable_gqa,
        )
        return dq, dk, dv, None, None, None


class TTXSWAFunction(MojoSWAFunction):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        cu_seqlens_kv: torch.Tensor,  # [bsz + 1]
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        gqa_interleave: bool = False,
        output_f32: bool = False,
    ) -> torch.Tensor:

        fwd_results = swa_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            is_causal,
            local_window_size,
            global_window_size,
            softmax_scale,
            gqa_interleave,
            output_f32,
        )
        if output_f32:
            o, softmax_lse, o_f32 = fwd_results
            ctx.save_for_backward(o_f32, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv)
        else:
            o, softmax_lse = fwd_results
            ctx.save_for_backward(o, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv)
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.local_window_size = local_window_size
        ctx.global_window_size = global_window_size
        ctx.gqa_interleave = gqa_interleave
        ctx.output_f32 = output_f32
        return o

    @staticmethod
    def backward(
        ctx, do: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None, None, None]:
        o, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        is_causal = ctx.is_causal
        local_window_size = ctx.local_window_size
        global_window_size = ctx.global_window_size
        gqa_interleave = ctx.gqa_interleave

        dq, dk, dv = swa_bwd(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            is_causal,
            local_window_size,
            global_window_size,
            softmax_scale,
            gqa_interleave,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None

