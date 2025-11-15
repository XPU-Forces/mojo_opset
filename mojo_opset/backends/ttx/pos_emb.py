import torch
import triton
from mojo_opset.backends.ttx.kernels.ascend.rope import ttx_rope, _rope_forward_kernel, _rope_backward_kernel

from mojo_opset.core import MojoRoPE, MojoRoPEFunction


class TTXRoPE(MojoRoPE, default_priority=0):
    def forward_std(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return ttx_rope(q, k, cos, sin)


class TTXRoPEFunction(MojoRoPEFunction):
    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()

        batch_size, seq_len, n_q_head, head_dim = q_t.shape
        n_kv_head = k_t.shape[2]

        num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

        grid = (num_programs,)

        cos_batch_size = cos.shape[0]
        cos = cos.contiguous()
        sin = sin.contiguous()

        _rope_forward_kernel[grid](
            q_t,
            q_t.stride(0),
            q_t.stride(1),
            k_t,
            k_t.stride(0),
            k_t.stride(1),
            cos,
            cos.stride(-2),
            sin,
            sin.stride(-2),
            seq_len,
            batch_size,
            cos_batch_size,
            n_q_head,
            n_kv_head,
            head_dim,
            head_dim // 2,
        )

        ctx.save_for_backward(cos, sin)
        ctx.seq_len = seq_len
        return q_t.transpose(1, 2), k_t.transpose(1, 2)

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        seq_len = ctx.seq_len

        dq_t = dq.transpose(1, 2).contiguous()
        dk_t = dk.transpose(1, 2).contiguous()

        batch_size, _, n_q_head, head_dim = dq_t.shape
        n_kv_head = dk_t.shape[2]
        cos_batch_size = cos.shape[0]

        num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        grid = (num_programs,)

        _rope_backward_kernel[grid](
            dq_t,
            dq_t.stride(0),
            dq_t.stride(1),
            dk_t,
            dk_t.stride(0),
            dk_t.stride(1),
            cos,
            cos.stride(-2),
            sin,
            sin.stride(-2),
            seq_len,
            batch_size,
            cos_batch_size,
            n_q_head,
            n_kv_head,
            head_dim,
            head_dim // 2,
        )

        return dq_t.transpose(1, 2), dk_t.transpose(1, 2), None, None, None, None
