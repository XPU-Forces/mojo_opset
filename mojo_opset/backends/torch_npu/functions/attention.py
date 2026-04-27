import math
from typing import Optional
from typing import Tuple

import torch
import torch_npu

from mojo_opset.core import MojoSWAFunction


def _generate_window_mask(
    q_seq_len: int,
    kv_seq_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
) -> torch.Tensor:
    kv_computed_len = kv_seq_len - q_seq_len
    causal_mask = (torch.arange(0, q_seq_len)[:, None] + kv_computed_len) >= torch.arange(0, kv_seq_len)[None, :]
    if local_window_size is not None or global_window_size is not None:
        local_window_mask = (
            (
                torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
            )
            if local_window_size is not None
            else False
        )
        global_window_mask = (
            (torch.arange(0, kv_seq_len) < global_window_size)[None, :] if global_window_size is not None else False
        )
        mask = causal_mask & (local_window_mask | global_window_mask)
    else:
        mask = causal_mask

    return mask

class TorchNpuSWAFunction(MojoSWAFunction):
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
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        total_q_tokens, n_q_heads, head_dim = q.shape
        n_kv_heads = k.shape[1]

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise NotImplementedError("TorchNpuSWAFunction only supports 3D TND tensors on NPU.")
        if q.device.type != "npu" or k.device.type != "npu" or v.device.type != "npu":
            raise NotImplementedError("TorchNpuSWAFunction only supports tensors on NPU.")
        if not is_causal:
            raise NotImplementedError("TorchNpuSWAFunction only supports is_causal=True.")
        if local_window_size is None:
            raise NotImplementedError("TorchNpuSWAFunction requires local_window_size for SWA.")

        actual_seq_qlen = cu_seqlens_q[1:].to(dtype=torch.int64).tolist()
        actual_seq_kvlen = cu_seqlens_kv[1:].to(dtype=torch.int64).tolist()
        max_q_seq_len = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        max_kv_seq_len = int((cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item())
        sparse_mode = 0 if global_window_size is not None else 4
        next_tockens = 0
        if sparse_mode == 0:
            pre_tockens = max_kv_seq_len
            atten_mask = (
                ~_generate_window_mask(max_q_seq_len, max_kv_seq_len, local_window_size, global_window_size)
            ).to(q.device)
        else:
            mask_dim = max(max_q_seq_len, max_kv_seq_len)
            pre_tockens = local_window_size
            atten_mask = torch.triu(torch.ones((mask_dim, mask_dim), dtype=torch.bool, device=q.device), diagonal=1)

        need_head_reorder = gqa_interleave and n_q_heads != n_kv_heads
        if need_head_reorder:
            if n_q_heads % n_kv_heads != 0:
                raise ValueError(f"n_q_heads must be divisible by n_kv_heads, got {n_q_heads} and {n_kv_heads}")
        group_size = n_q_heads // n_kv_heads if need_head_reorder else 1
        need_backward = any(t.requires_grad for t in (q, k, v))

        if need_backward:
            with torch.enable_grad():
                q_tensor = q.detach().requires_grad_(True)
                k_tensor = k.detach().requires_grad_(True)
                v_tensor = v.detach().requires_grad_(True)
                if need_head_reorder:
                    q_fa = q_tensor.reshape(total_q_tokens, group_size, n_kv_heads, head_dim).transpose(1, 2).reshape(
                        total_q_tokens, n_q_heads, head_dim
                    )
                else:
                    q_fa = q_tensor
                o_internal = torch_npu.npu_fusion_attention(
                    q_fa,
                    k_tensor,
                    v_tensor,
                    n_q_heads,
                    "TND",
                    scale=softmax_scale,
                    keep_prob=1.0,
                    pre_tockens=pre_tockens,
                    next_tockens=next_tockens,
                    atten_mask=atten_mask,
                    actual_seq_qlen=actual_seq_qlen,
                    actual_seq_kvlen=actual_seq_kvlen,
                    sparse_mode=sparse_mode,
                )[0]
                if need_head_reorder:
                    o = o_internal.reshape(total_q_tokens, n_kv_heads, group_size, head_dim).transpose(1, 2).reshape(
                        total_q_tokens, n_q_heads, head_dim
                    )
                else:
                    o = o_internal
                o = o.detach()
            ctx.q_graph = q_tensor
            ctx.k_graph = k_tensor
            ctx.v_graph = v_tensor
            ctx.o_internal = o_internal
            ctx.input_requires_grad = (q.requires_grad, k.requires_grad, v.requires_grad)
        else:
            if need_head_reorder:
                q_fa = q.reshape(total_q_tokens, group_size, n_kv_heads, head_dim).transpose(1, 2).reshape(
                    total_q_tokens, n_q_heads, head_dim
                )
            else:
                q_fa = q
            o_internal = torch_npu.npu_fusion_attention(
                q_fa,
                k,
                v,
                n_q_heads,
                "TND",
                scale=softmax_scale,
                keep_prob=1.0,
                pre_tockens=pre_tockens,
                next_tockens=next_tockens,
                atten_mask=atten_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                sparse_mode=sparse_mode,
            )[0]
            if need_head_reorder:
                o = o_internal.reshape(total_q_tokens, n_kv_heads, group_size, head_dim).transpose(1, 2).reshape(
                    total_q_tokens, n_q_heads, head_dim
                )
            else:
                o = o_internal

        ctx.need_head_reorder = need_head_reorder
        ctx.group_size = group_size
        ctx.is_causal = is_causal
        ctx.local_window_size = local_window_size
        ctx.global_window_size = global_window_size
        ctx.softmax_scale = softmax_scale
        ctx.gqa_interleave = gqa_interleave
        ctx.n_kv_heads = k.shape[1]
        ctx.output_f32 = output_f32
        return o

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None, None, None]:
        if ctx.need_head_reorder:
            total_q_tokens, n_q_heads, head_dim = do.shape
            do_internal = do.reshape(total_q_tokens, ctx.group_size, ctx.n_kv_heads, head_dim).transpose(1, 2).reshape(
                total_q_tokens, n_q_heads, head_dim
            )
        else:
            do_internal = do
        dq, dk, dv = torch.autograd.grad(
            outputs=ctx.o_internal,
            inputs=(ctx.q_graph, ctx.k_graph, ctx.v_graph),
            grad_outputs=do_internal,
            allow_unused=False,
        )
        q_requires_grad, k_requires_grad, v_requires_grad = ctx.input_requires_grad
        if not q_requires_grad:
            dq = None
        if not k_requires_grad:
            dk = None
        if not v_requires_grad:
            dv = None
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None
