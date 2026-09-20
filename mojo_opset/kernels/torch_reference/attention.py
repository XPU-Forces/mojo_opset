"""Original Mojo SWA arithmetic with bounded-memory query chunks.

QK and unnormalized PV matmuls retain the input dtype. Softmax statistics,
normalization and backward pointwise arithmetic use FP32, as in Mojo core.
"""

from typing import Optional

import torch


def _generate_window_mask_chunk(
    q_start: int,
    q_end: int,
    kv_seq_len: int,
    kv_computed_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    q_arange = torch.arange(q_start, q_end, device=device)
    kv_arange = torch.arange(0, kv_seq_len, device=device)
    causal_mask = (q_arange[:, None] + kv_computed_len) >= kv_arange[None, :]
    if local_window_size is not None or global_window_size is not None:
        local_window_mask = (
            (q_arange[:, None] + kv_computed_len <= kv_arange[None, :] + local_window_size)
            if local_window_size is not None
            else False
        )
        global_window_mask = (kv_arange < global_window_size)[None, :] if global_window_size is not None else False
        mask = causal_mask & (local_window_mask | global_window_mask)
    else:
        mask = causal_mask
    return mask


def _swa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    q_chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_q_len, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o_f32 = torch.empty_like(q, dtype=torch.float32)
    softmax_lse = torch.empty((n_q_heads, total_q_len), dtype=torch.float32, device=q.device)
    bsz = cu_q_lens.shape[0] - 1

    for i in range(bsz):
        q_batch_start = cu_q_lens[i].item()
        q_batch_end = cu_q_lens[i + 1].item()
        kv_batch_start = cu_k_lens[i].item()
        kv_batch_end = cu_k_lens[i + 1].item()
        q_seq_len = q_batch_end - q_batch_start
        kv_seq_len = kv_batch_end - kv_batch_start
        kv_computed_len = kv_seq_len - q_seq_len

        k_i = k[kv_batch_start:kv_batch_end]
        v_i = v[kv_batch_start:kv_batch_end]

        k_i_T = k_i.permute(1, 2, 0)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                k_i_T = k_i_T.repeat_interleave(n_q_heads // n_kv_heads, dim=0)

        v_i_perm = v_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i_perm = v_i_perm.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                v_i_perm = v_i_perm.repeat_interleave(n_q_heads // n_kv_heads, dim=0)

        for qc_start in range(0, q_seq_len, q_chunk_size):
            qc_end = min(qc_start + q_chunk_size, q_seq_len)

            q_chunk = q[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)

            s_chunk = torch.bmm(q_chunk, k_i_T).float() * softmax_scale

            if is_causal:
                s_mask = _generate_window_mask_chunk(
                    qc_start,
                    qc_end,
                    kv_seq_len,
                    kv_computed_len,
                    local_window_size,
                    global_window_size,
                    device=s_chunk.device,
                )
                s_chunk = torch.where(s_mask, s_chunk, float("-inf"))

            m_chunk = torch.max(s_chunk, dim=-1, keepdim=True).values
            s_chunk = s_chunk - m_chunk
            p_chunk = torch.exp(s_chunk)
            l_chunk = torch.sum(p_chunk, dim=-1, keepdim=True)
            p_chunk = p_chunk.to(v.dtype)

            o_chunk = torch.bmm(p_chunk, v_i_perm).float()
            o_chunk = o_chunk / l_chunk
            o_chunk = o_chunk.permute(1, 0, 2)
            o_f32[q_batch_start + qc_start : q_batch_start + qc_end] = o_chunk

            lse_chunk = m_chunk + torch.log(l_chunk)
            softmax_lse[:, q_batch_start + qc_start : q_batch_start + qc_end] = lse_chunk.squeeze(-1)

    o = o_f32.to(q.dtype)
    return o, softmax_lse, o_f32


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
    output, softmax_lse, output_cache = _swa_forward(
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
    return _dense_attention_forward(
        q, k, v, cu_q_lens, cu_k_lens, is_causal, local_window_size,
        global_window_size, softmax_scale, gqa_interleave,
    )[0]



def swa_infer_fwd(
    q, k, v, cu_q_lens, cu_k_lens, is_causal, local_window_size, global_window_size, softmax_scale, gqa_interleave
):
    return _swa_forward(
        q, k, v, cu_q_lens, cu_k_lens, is_causal, local_window_size,
        global_window_size, softmax_scale, gqa_interleave,
    )[0]


def swa_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_f32: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    q_chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = torch.sum(output_f32.float() * grad_output.float(), dim=-1)
    bsz = cu_q_lens.shape[0] - 1
    group_size = n_q_heads // n_kv_heads if n_q_heads != n_kv_heads else 1

    for i in range(bsz):
        q_batch_start = cu_q_lens[i].item()
        q_batch_end = cu_q_lens[i + 1].item()
        kv_batch_start = cu_k_lens[i].item()
        kv_batch_end = cu_k_lens[i + 1].item()
        q_seq_len = q_batch_end - q_batch_start
        kv_seq_len = kv_batch_end - kv_batch_start
        kv_computed_len = kv_seq_len - q_seq_len

        k_i = k[kv_batch_start:kv_batch_end]
        v_i = v[kv_batch_start:kv_batch_end]

        k_i_perm = k_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_expanded = k_i_perm.repeat((group_size, 1, 1))
            else:
                k_i_expanded = k_i_perm.repeat_interleave(group_size, dim=0)
        else:
            k_i_expanded = k_i_perm

        v_i_perm = v_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i_expanded = v_i_perm.repeat((group_size, 1, 1))
            else:
                v_i_expanded = v_i_perm.repeat_interleave(group_size, dim=0)
        else:
            v_i_expanded = v_i_perm

        dk_dtype = torch.float32 if k.dtype in (torch.float16, torch.bfloat16) else k.dtype
        dv_dtype = torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype
        dk_i = torch.zeros((kv_seq_len, n_kv_heads, head_dim), dtype=dk_dtype, device=k.device)
        dv_i = torch.zeros((kv_seq_len, n_kv_heads, head_dim), dtype=dv_dtype, device=v.device)

        for qc_start in range(0, q_seq_len, q_chunk_size):
            qc_end = min(qc_start + q_chunk_size, q_seq_len)

            q_chunk = q[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)
            do_chunk = grad_output[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)

            s_chunk = torch.bmm(q_chunk, k_i_expanded.mT).float() * softmax_scale

            if is_causal:
                s_mask = _generate_window_mask_chunk(
                    qc_start,
                    qc_end,
                    kv_seq_len,
                    kv_computed_len,
                    local_window_size,
                    global_window_size,
                    device=s_chunk.device,
                )
                s_chunk = torch.where(s_mask, s_chunk, float("-inf"))

            lse_chunk = softmax_lse[:, q_batch_start + qc_start : q_batch_start + qc_end]
            p_chunk = torch.exp(s_chunk - lse_chunk.unsqueeze(-1))

            dp_chunk = torch.bmm(do_chunk, v_i_expanded.mT).float()
            delta_chunk = delta[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0).unsqueeze(-1)
            ds_chunk = p_chunk * (dp_chunk - delta_chunk)
            ds_chunk = ds_chunk * softmax_scale
            ds_chunk = ds_chunk.to(do_chunk.dtype)
            p_chunk = p_chunk.to(do_chunk.dtype)

            dq_chunk = torch.bmm(ds_chunk, k_i_expanded)
            dq[q_batch_start + qc_start : q_batch_start + qc_end] = dq_chunk.permute(1, 0, 2)

            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    ds_reduced = ds_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    q_reduced = q_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    p_reduced = p_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    do_reduced = do_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                else:
                    ds_reduced = ds_chunk.unflatten(0, (n_kv_heads, group_size))
                    q_reduced = q_chunk.unflatten(0, (n_kv_heads, group_size))
                    p_reduced = p_chunk.unflatten(0, (n_kv_heads, group_size))
                    do_reduced = do_chunk.unflatten(0, (n_kv_heads, group_size))

                ds_reduced = ds_reduced.flatten(1, 2)
                q_reduced = q_reduced.flatten(1, 2)
                p_reduced = p_reduced.flatten(1, 2)
                do_reduced = do_reduced.flatten(1, 2)
            else:
                ds_reduced = ds_chunk
                q_reduced = q_chunk
                p_reduced = p_chunk
                do_reduced = do_chunk

            # Keep the original low-precision dS/P operands, but defer rounding
            # the reduction over queries until every chunk has contributed.
            # Casting a low-precision bmm result afterwards would already have
            # lost small contributions within each chunk.
            with torch.autocast(device_type=q.device.type, enabled=False):
                dk_i += torch.bmm(ds_reduced.to(dk_dtype).mT, q_reduced.to(dk_dtype)).permute(1, 0, 2)
                dv_i += torch.bmm(p_reduced.to(dv_dtype).mT, do_reduced.to(dv_dtype)).permute(1, 0, 2)

        dk[kv_batch_start:kv_batch_end] = dk_i
        dv[kv_batch_start:kv_batch_end] = dv_i

    return dq, dk, dv


# FlashAttention and the temporary native interface retain their existing FP32 reference.
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


def _dense_attention_forward(
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
    return _dense_attention_forward(q, k, v, cu_q_lens, cu_k_lens, causal, None, None, softmax_scale, False)


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
    del output_f32, softmax_lse
    inputs = tuple(tensor.detach().requires_grad_(True) for tensor in (q, k, v))
    with torch.enable_grad():
        output, _, _ = _dense_attention_forward(
            *inputs, cu_q_lens, cu_k_lens, causal, None, None, softmax_scale, False,
        )
        gradients = torch.autograd.grad(output, inputs, grad_output)
    return tuple(gradient.to(tensor.dtype) for gradient, tensor in zip(gradients, (q, k, v)))
