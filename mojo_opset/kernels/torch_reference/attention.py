from typing import Optional

import torch


def _expand_kv_heads(tensor: torch.Tensor, num_q_heads: int, interleaved: bool) -> torch.Tensor:
    num_kv_heads = tensor.shape[1]
    if num_q_heads == num_kv_heads:
        return tensor
    repeats = num_q_heads // num_kv_heads
    tensor = tensor.permute(1, 0, 2)
    if interleaved:
        tensor = tensor.repeat((repeats, 1, 1))
    else:
        tensor = tensor.repeat_interleave(repeats, dim=0)
    return tensor.permute(1, 0, 2)


def _window_mask(
    q_length: int,
    kv_length: int,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    cached_length = kv_length - q_length
    query_positions = torch.arange(q_length, device=device) + cached_length
    key_positions = torch.arange(kv_length, device=device)
    mask = query_positions[:, None] >= key_positions[None, :]
    if local_window_size is not None or global_window_size is not None:
        local = (
            query_positions[:, None] <= key_positions[None, :] + local_window_size
            if local_window_size is not None
            else False
        )
        global_tokens = key_positions[None, :] < global_window_size if global_window_size is not None else False
        mask = mask & (local | global_tokens)
    return mask


def _attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    is_causal: bool,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
    softmax_scale: float,
    gqa_interleave: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_offsets = cu_q_lens.detach().cpu().tolist()
    k_offsets = cu_k_lens.detach().cpu().tolist()
    outputs = []
    outputs_f32 = []
    lse_parts = []
    for q_start, q_end, k_start, k_end in zip(q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]):
        q_part = q[q_start:q_end].permute(1, 0, 2)
        k_part = _expand_kv_heads(k[k_start:k_end], q.shape[1], gqa_interleave).permute(1, 0, 2)
        v_part = _expand_kv_heads(v[k_start:k_end], q.shape[1], gqa_interleave).permute(1, 0, 2)
        scores = torch.bmm(q_part.float(), k_part.float().transpose(1, 2)) * softmax_scale
        if is_causal:
            mask = _window_mask(
                q_end - q_start,
                k_end - k_start,
                local_window_size,
                global_window_size,
                q.device,
            )
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        # Keep the semantic reference in FP32 through PV. Rounding normalized
        # probabilities to BF16 changes both the result and autograd's gradients;
        # optimized online-softmax kernels round different intermediate values.
        output_f32 = torch.bmm(probabilities, v_part.float()).permute(1, 0, 2)
        outputs_f32.append(output_f32)
        outputs.append(output_f32.to(q.dtype))
        lse_parts.append(torch.logsumexp(scores, dim=-1))
    return torch.cat(outputs), torch.cat(lse_parts, dim=1), torch.cat(outputs_f32)


def swa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    is_causal: bool,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
    softmax_scale: float,
    gqa_interleave: bool,
    output_f32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    output, softmax_lse, output_cache = _attention_forward(
        q,
        k,
        v,
        cu_q_lens,
        cu_k_lens,
        is_causal,
        local_window_size,
        global_window_size,
        softmax_scale,
        gqa_interleave,
    )
    return output, softmax_lse, output_cache if output_f32 else None


def native_swa_infer_fwd(
    q, k, v, cu_q_lens, cu_k_lens, is_causal, local_window_size, global_window_size, softmax_scale, gqa_interleave
):
    return swa_fwd(
        q, k, v, cu_q_lens, cu_k_lens, is_causal, local_window_size, global_window_size, softmax_scale, gqa_interleave
    )[0]


def swa_infer_fwd(
    q, k, v, cu_q_lens, cu_k_lens, is_causal, local_window_size, global_window_size, softmax_scale, gqa_interleave
):
    # Preserve MojoSWA's QK and unnormalized-PV rounding, independently of training.
    output = torch.empty_like(q)
    q_offsets = cu_q_lens.detach().cpu().tolist()
    k_offsets = cu_k_lens.detach().cpu().tolist()
    for qs, qe, ks, ke in zip(q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]):
        query = q[qs:qe].permute(1, 0, 2)
        key = _expand_kv_heads(k[ks:ke], q.shape[1], gqa_interleave).permute(1, 2, 0)
        value = _expand_kv_heads(v[ks:ke], q.shape[1], gqa_interleave).permute(1, 0, 2)
        scores = torch.bmm(query, key).float() * softmax_scale
        if is_causal:
            mask = _window_mask(qe - qs, ke - ks, local_window_size, global_window_size, q.device)
            scores = scores.masked_fill(~mask, float("-inf"))
        probabilities = (scores - scores.amax(dim=-1, keepdim=True)).exp()
        denominator = probabilities.sum(dim=-1, keepdim=True)
        value_sum = torch.bmm(probabilities.to(v.dtype), value).float()
        output[qs:qe] = (value_sum / denominator).permute(1, 0, 2).to(q.dtype)
    return output


def swa_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_f32: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    is_causal: bool,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
    softmax_scale: float,
    gqa_interleave: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del output_f32, softmax_lse
    inputs = tuple(tensor.detach().requires_grad_(True) for tensor in (q, k, v))
    with torch.enable_grad():
        output, _, _ = _attention_forward(
            *inputs,
            cu_q_lens,
            cu_k_lens,
            is_causal,
            local_window_size,
            global_window_size,
            softmax_scale,
            gqa_interleave,
        )
        gradients = torch.autograd.grad(output, inputs, grad_output)
    return tuple(gradient.to(tensor.dtype) for gradient, tensor in zip(gradients, (q, k, v)))


def flash_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    max_q_len: int,
    max_k_len: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del max_q_len, max_k_len, deterministic
    if attention_mask is not None:
        raise NotImplementedError("flash_attention does not yet support an explicit attention mask")
    if dropout_p != 0.0:
        raise NotImplementedError("flash_attention currently supports dropout_p=0 only")
    return _attention_forward(q, k, v, cu_q_lens, cu_k_lens, causal, None, None, softmax_scale, False)


def flash_attention_infer_fwd(
    q, k, v, attention_mask, cu_q_lens, cu_k_lens, max_q_len, max_k_len, dropout_p, softmax_scale, causal, deterministic
):
    return flash_attention_fwd(
        q,
        k,
        v,
        attention_mask,
        cu_q_lens,
        cu_k_lens,
        max_q_len,
        max_k_len,
        dropout_p,
        softmax_scale,
        causal,
        deterministic,
    )[0]


def flash_attention_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    output_f32: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    max_q_len: int,
    max_k_len: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del max_q_len, max_k_len, deterministic
    if attention_mask is not None:
        raise NotImplementedError("flash_attention does not yet support an explicit attention mask")
    if dropout_p != 0.0:
        raise NotImplementedError("flash_attention currently supports dropout_p=0 only")
    return swa_bwd(
        grad_output,
        q,
        k,
        v,
        output_f32,
        softmax_lse,
        cu_q_lens,
        cu_k_lens,
        causal,
        None,
        None,
        softmax_scale,
        False,
    )
