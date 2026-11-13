"""Packed training FA using torch_npu, without model scheduling or offload."""

from typing import Optional

import torch
import torch_npu

_CAUSAL_MASKS = {}


def _mask(attention_mask, causal, device):
    if attention_mask is not None:
        return attention_mask, 1
    if not causal:
        return None, 0
    if device not in _CAUSAL_MASKS:
        _CAUSAL_MASKS[device] = torch.ones(2048, 2048, dtype=torch.bool, device=device).triu_(1)
    return _CAUSAL_MASKS[device], 3


@torch.library.custom_op("mojo_npu_torch_npu::flash_attention_fwd", mutates_args=())
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask, sparse_mode = _mask(attention_mask, causal, q.device)
    result = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        q.shape[1],
        input_layout="TND",
        pse=None,
        atten_mask=mask,
        scale=softmax_scale,
        keep_prob=1.0 - dropout_p,
        actual_seq_qlen=tuple(cu_q_lens[1:].tolist()),
        actual_seq_kvlen=tuple(cu_k_lens[1:].tolist()),
        sparse_mode=sparse_mode,
    )
    # CANN regenerates the forward mask in backward from these host scalars.
    # Keep them on CPU so backward does not introduce a device synchronization.
    rng_state = torch.tensor(result[4:7], dtype=torch.int64, device="cpu")
    return result[0], result[1], result[2], rng_state


@_forward.register_fake
def _forward_fake(q, k, v, attention_mask, cu_q_lens, cu_k_lens, softmax_scale, causal, dropout_p):
    stats = (q.shape[0], q.shape[1], 8)
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        q.new_empty(stats, dtype=torch.float32),
        q.new_empty(stats, dtype=torch.float32),
        torch.empty(3, dtype=torch.int64, device="cpu"),
    )


@torch.library.custom_op("mojo_npu_torch_npu::flash_attention_bwd", mutates_args=())
def _backward(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    rng_state: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask, sparse_mode = _mask(attention_mask, causal, q.device)
    seed, offset, numels = rng_state.tolist()
    result = torch_npu.npu_fusion_attention_grad(
        q,
        k,
        v,
        grad,
        q.shape[1],
        input_layout="TND",
        pse=None,
        atten_mask=mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=output,
        scale_value=softmax_scale,
        keep_prob=1.0 - dropout_p,
        seed=seed,
        offset=offset,
        numels=numels,
        actual_seq_qlen=tuple(cu_q_lens[1:].tolist()),
        actual_seq_kvlen=tuple(cu_k_lens[1:].tolist()),
        sparse_mode=sparse_mode,
    )
    return result[0], result[1], result[2]


@_backward.register_fake
def _backward_fake(
    grad,
    q,
    k,
    v,
    attention_mask,
    output,
    softmax_max,
    softmax_sum,
    rng_state,
    cu_q_lens,
    cu_k_lens,
    softmax_scale,
    causal,
    dropout_p,
):
    return tuple(torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v))


def flash_attention_fwd(
    q, k, v, attention_mask, cu_q_lens, cu_k_lens, max_q_len, max_k_len, dropout_p, softmax_scale, causal, deterministic
):
    if attention_mask is not None:
        raise NotImplementedError("flash_attention does not yet support an explicit attention mask")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    # torch_npu follows the caller's global determinism policy. The legacy
    # deterministic argument is retained for signature compatibility only.
    output, softmax_max, softmax_sum, rng_state = _forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        attention_mask,
        cu_q_lens,
        cu_k_lens,
        softmax_scale,
        causal,
        dropout_p,
    )
    return output, output, softmax_max, softmax_sum, rng_state


def flash_attention_bwd(
    grad,
    q,
    k,
    v,
    attention_mask,
    output,
    softmax_max,
    softmax_sum,
    rng_state,
    cu_q_lens,
    cu_k_lens,
    max_q_len,
    max_k_len,
    dropout_p,
    softmax_scale,
    causal,
    deterministic,
):
    if attention_mask is not None:
        raise NotImplementedError("flash_attention does not yet support an explicit attention mask")
    return _backward(
        grad.contiguous(),
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        attention_mask,
        output,
        softmax_max,
        softmax_sum,
        rng_state,
        cu_q_lens,
        cu_k_lens,
        softmax_scale,
        causal,
        dropout_p,
    )
