"""Deterministic packed A2 FA with an FP32 backward cache."""

import torch

from ._extension import launcher

_CAUSAL_MASKS = {}


def _mask(causal, q):
    if not causal:
        return q.new_empty((0,), dtype=torch.bool)
    # Use the stream ID: reading the raw handle can drain the NPU task queue.
    key = (q.device, torch.npu.current_stream(q.device).stream_id)
    if key not in _CAUSAL_MASKS:
        _CAUSAL_MASKS[key] = torch.ones(2048, 2048, dtype=torch.bool, device=q.device).triu_(1)
    return _CAUSAL_MASKS[key]


def _sequence_ends(cu):
    return cu[1:].to(dtype=torch.int32).contiguous()


@torch.library.custom_op("mojo_npu_native_a2::flash_attention_fwd", mutates_args=())
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    max_q_len: int,
    max_k_len: int,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output, softmax_max, softmax_sum = launcher("flash_attention_forward")(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        _mask(causal, q),
        _sequence_ends(cu_q_lens),
        _sequence_ends(cu_k_lens),
        max_q_len,
        max_k_len,
        softmax_scale,
        causal,
    )
    return output.to(q.dtype), output, softmax_max, softmax_sum


@_forward.register_fake
def _forward_fake(q, k, v, cu_q_lens, cu_k_lens, max_q_len, max_k_len, softmax_scale, causal):
    stats = (q.shape[0], q.shape[1], 8)
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(q, dtype=torch.float32, memory_format=torch.contiguous_format),
        q.new_empty(stats, dtype=torch.float32),
        q.new_empty(stats, dtype=torch.float32),
    )


@torch.library.custom_op("mojo_npu_native_a2::flash_attention_bwd", mutates_args=())
def _backward(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return launcher("flash_attention_backward")(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        grad.contiguous(),
        _mask(causal, q),
        output,
        softmax_max,
        softmax_sum,
        _sequence_ends(cu_q_lens),
        _sequence_ends(cu_k_lens),
        softmax_scale,
        causal,
    )


@_backward.register_fake
def _backward_fake(grad, q, k, v, output, softmax_max, softmax_sum, cu_q_lens, cu_k_lens, softmax_scale, causal):
    return tuple(torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v))


def flash_attention_fwd(
    q, k, v, attention_mask, cu_q_lens, cu_k_lens, max_q_len, max_k_len, dropout_p, softmax_scale, causal, deterministic
):
    if dropout_p != 0.0 or attention_mask is not None:
        raise NotImplementedError("native flash_attention requires dropout_p=0 and attention_mask=None")
    # Only deterministic backward is compiled, including for legacy callers
    # passing deterministic=False. No process-wide setting is changed.
    return _forward(q, k, v, cu_q_lens, cu_k_lens, max_q_len, max_k_len, softmax_scale, causal)


def flash_attention_bwd(
    grad,
    q,
    k,
    v,
    attention_mask,
    output,
    softmax_max,
    softmax_sum,
    cu_q_lens,
    cu_k_lens,
    max_q_len,
    max_k_len,
    dropout_p,
    softmax_scale,
    causal,
    deterministic,
):
    return _backward(grad, q, k, v, output, softmax_max, softmax_sum, cu_q_lens, cu_k_lens, softmax_scale, causal)
