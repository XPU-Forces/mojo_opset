"""NPU Fusion Attention Implementation for PyTorch."""

import torch
import torch_npu
from mojo_opset.core import MojoFusionAttentionFunction


class TorchNpuFusionAttentionFunction(MojoFusionAttentionFunction):
    """NPU-optimized Fusion Attention supporting TND (varlen) and BNSD layouts."""

    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        actual_seq_qlen,
        actual_seq_kvlen,
        head_num,
        scale=1.0,
        dropout_p=0.0,
        atten_mask=None,
        is_varlen=True,
        is_causal=True,
        **kwargs,
    ):
        input_layout = "TND" if is_varlen else "BNSD"

        # Prepare causal mask if needed
        if is_causal and atten_mask is None:
            if is_varlen:
                if actual_seq_qlen is None or actual_seq_kvlen is None:
                    raise ValueError("actual_seq_qlen/kvlen required for varlen mode")
                seq_q = actual_seq_qlen if isinstance(actual_seq_qlen, torch.Tensor) else torch.tensor(actual_seq_qlen)
                seq_kv = actual_seq_kvlen if isinstance(actual_seq_kvlen, torch.Tensor) else torch.tensor(actual_seq_kvlen)
                max_q = torch.diff(seq_q, prepend=torch.tensor([0])).max().item()
                max_kv = torch.diff(seq_kv, prepend=torch.tensor([0])).max().item()
            else:
                max_q, max_kv = query.shape[-2], key.shape[-2]
            atten_mask = torch.triu(
                torch.ones(max_q, max_kv, dtype=torch.bool, device=query.device),
                diagonal=1,
            )

        # Convert to list for NPU API
        if isinstance(actual_seq_qlen, torch.Tensor):
            actual_seq_qlen = actual_seq_qlen.tolist()
        if isinstance(actual_seq_kvlen, torch.Tensor):
            actual_seq_kvlen = actual_seq_kvlen.tolist()

        # Forward
        attn_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = \
            torch_npu.npu_fusion_attention(
                query, key, value, head_num,
                input_layout=input_layout,
                atten_mask=atten_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                scale=scale,
                keep_prob=1.0 - dropout_p,
                **kwargs,
            )

        # Save for backward
        ctx.save_for_backward(
            query, key, value, atten_mask,
            softmax_max, softmax_sum, softmax_out,
            seed, offset, numels,
        )
        ctx.input_layout = input_layout
        ctx.head_num = head_num
        ctx.scale = scale
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.kwargs = kwargs

        return attn_out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, atten_mask, softmax_max, softmax_sum, softmax_out, *_ = ctx.saved_tensors

        # Recompute attention output
        with torch.enable_grad():
            query = query.detach().requires_grad_(True)
            key = key.detach().requires_grad_(True)
            value = value.detach().requires_grad_(True)

            attention_in, *_ = torch_npu.npu_fusion_attention(
                query, key, value, ctx.head_num,
                input_layout=ctx.input_layout,
                atten_mask=atten_mask,
                actual_seq_qlen=ctx.actual_seq_qlen,
                actual_seq_kvlen=ctx.actual_seq_kvlen,
                scale=ctx.scale,
                keep_prob=1.0,
                **ctx.kwargs,
            )

        # Backward
        grad_q, grad_k, grad_v, _ = torch_npu.npu_fusion_attention_grad(
            query, key, value, grad_output, ctx.head_num, ctx.input_layout,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attention_in,
            scale_value=ctx.scale,
            actual_seq_qlen=ctx.actual_seq_qlen,
            actual_seq_kvlen=ctx.actual_seq_kvlen,
            **ctx.kwargs,
        )

        # Gradients: query, key, value, actual_seq_qlen, actual_seq_kvlen,
        #            head_num, scale, dropout_p, atten_mask, is_varlen, is_causal
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None