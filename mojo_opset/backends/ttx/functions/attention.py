import torch

from mojo_opset.backends.ttx.kernels import diffusion_attention_bwd
from mojo_opset.backends.ttx.kernels import diffusion_attention_fwd
from mojo_opset.backends.ttx.kernels import diffusion_attention_up_bwd
from mojo_opset.backends.ttx.kernels import diffusion_attention_up_fwd
from mojo_opset.experimental import MojoDiffusionAttentionFunction
from mojo_opset.experimental import MojoDiffusionAttentionUpFunction


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


class TTXDiffusionAttentionUpFunction(MojoDiffusionAttentionUpFunction):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlen: torch.Tensor,
        scale: float = 1.0,
        BLOCK_SIZE: int = 8,
    ) -> torch.Tensor:
        output, output_fp32, lse = diffusion_attention_up_fwd(
            query,
            key,
            value,
            cu_seqlen,
            scale,
            BLOCK_SIZE,
        )
        ctx.save_for_backward(query, key, value, output_fp32, lse, cu_seqlen, scale, BLOCK_SIZE)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value, output_fp32, lse, cu_seqlen, scale, BLOCK_SIZE = ctx.saved_tensors
        dq, dk, dv = diffusion_attention_up_bwd(
            output_fp32,
            grad_output,
            query,
            key,
            value,
            lse,
            cu_seqlen,
            scale,
            BLOCK_SIZE,
        )
        return dq, dk, dv, None, None, None
