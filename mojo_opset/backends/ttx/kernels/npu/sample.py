import torch
import triton
import triton.language as tl


@triton.jit
def _top_p_sample_kernel(
    sorted_logits_ptr,
    sorted_indices_ptr,
    rand_data_ptr,
    output_ptr,
    output_probs_ptr,
    top_p,
    filter_value,
    min_tokens_to_keep,
    strategy: tl.constexpr,
    stride_logits_b,
    stride_logits_k,
    stride_indices_b,
    stride_indices_k,
    stride_rand_b,
    stride_rand_k,
    stride_out0_b,
    stride_out0_k,
    stride_out1_b,
    stride_out1_k,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)

    row_logits_ptr = sorted_logits_ptr + pid * stride_logits_b
    offsets = tl.arange(0, TOP_K)

    logits = tl.load(row_logits_ptr + offsets * stride_logits_k)

    logits_max = tl.max(logits, 0)
    numerator = tl.exp(logits - logits_max)
    probs = numerator / tl.sum(numerator, 0)
    cum_probs = tl.cumsum(probs, 0)
    to_remove = (cum_probs - probs) > top_p
    to_remove = tl.where(offsets < min_tokens_to_keep, False, to_remove)
    filtered_logits = tl.where(to_remove, filter_value, logits)
    f_logits_max = tl.max(filtered_logits, 0)
    f_numerator = tl.exp(filtered_logits - f_logits_max)
    f_probs = f_numerator / tl.sum(f_numerator, 0)

    if strategy == 0:
        row_indices_ptr = sorted_indices_ptr + pid * stride_indices_b
        out_token_ptr = output_ptr + pid * stride_out0_b
        out_prob_ptr = out_token_ptr + 1

        threshold = tl.load(rand_data_ptr + pid * stride_rand_b)
        f_cum_probs = tl.cumsum(f_probs, 0)
        is_candidate = f_cum_probs >= threshold
        candidate_indices = tl.where(is_candidate, offsets, TOP_K)
        sampled_index_in_topk = tl.min(candidate_indices, 0)
        sampled_index_in_topk = tl.where(sampled_index_in_topk == TOP_K, 0, sampled_index_in_topk)

        is_selected_mask = offsets == sampled_index_in_topk
        selected_prob_val = tl.sum(tl.where(is_selected_mask, f_probs, 0.0), 0)
        all_topk_indices = tl.load(row_indices_ptr + offsets * stride_indices_k)
        selected_token_val = tl.sum(tl.where(is_selected_mask, all_topk_indices, 0), 0)

        tl.store(out_token_ptr, selected_token_val.to(tl.int32))
        tl.store(out_prob_ptr, selected_prob_val)

    elif strategy == 1:
        row_rand_ptr = rand_data_ptr + pid * stride_rand_b
        row_scores_ptr = output_ptr + pid * stride_out0_b
        row_probs_ptr = output_probs_ptr + pid * stride_out1_b

        noise = tl.load(row_rand_ptr + offsets * stride_rand_k)
        eps = 1e-9
        scores = f_probs / (noise + eps)

        tl.store(row_scores_ptr + offsets * stride_out0_k, scores)
        tl.store(row_probs_ptr + offsets * stride_out1_k, f_probs)


def top_p_sampling_impl(
    logits: torch.FloatTensor,
    top_p: float = 0.75,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    rand_top_k: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = logits.device
    logits = logits.to(torch.float32)
    batch_size, _ = logits.shape
    top_k = min(rand_top_k, logits.size(-1))

    sorted_logits, sorted_topk_indices = torch.topk(logits, top_k)

    probs_bytes_count = logits.element_size() * logits.shape[0] * rand_top_k

    if probs_bytes_count <= 20000:
        strategy = 0
        rand_data = torch.rand(batch_size, device=device)

        output_data = torch.empty((batch_size, 2), dtype=torch.float32, device=device)

        grid = (batch_size,)

        _top_p_sample_kernel[grid](
            sorted_logits,
            sorted_topk_indices,
            rand_data,
            output_data,
            None,
            top_p,
            filter_value,
            min_tokens_to_keep,
            strategy,
            sorted_logits.stride(0),
            sorted_logits.stride(1),
            sorted_topk_indices.stride(0),
            sorted_topk_indices.stride(1),
            rand_data.stride(0),
            1,
            output_data.stride(0),
            output_data.stride(1),
            0,
            0,
            TOP_K=top_k,
        )

        output_tokens = output_data[:, 0].long().unsqueeze(-1)
        output_probs = output_data[:, 1].unsqueeze(-1)

    else:
        strategy = 1
        rand_data = torch.rand_like(sorted_logits)

        output_scores = torch.empty_like(sorted_logits)
        output_final_probs = torch.empty_like(sorted_logits)

        grid = (batch_size,)
        _top_p_sample_kernel[grid](
            sorted_logits,
            sorted_topk_indices,
            rand_data,
            output_scores,
            output_final_probs,
            top_p,
            filter_value,
            min_tokens_to_keep,
            strategy,
            sorted_logits.stride(0),
            sorted_logits.stride(1),
            sorted_topk_indices.stride(0),
            sorted_topk_indices.stride(1),
            rand_data.stride(0),
            rand_data.stride(1),
            output_scores.stride(0),
            output_scores.stride(1),
            output_final_probs.stride(0),
            output_final_probs.stride(1),
            TOP_K=top_k,
        )

        sampled_index_in_topk = torch.argmax(output_scores, dim=-1, keepdim=True)

        output_tokens = torch.gather(sorted_topk_indices, -1, sampled_index_in_topk)

        output_probs = torch.gather(output_final_probs, -1, sampled_index_in_topk)

    return output_probs, output_tokens


@triton.jit
def _top_p_filter_kernel(
    sorted_logits_ptr,
    output_ptr,
    top_p,
    filter_value,
    min_tokens_to_keep,
    stride_logits_b,
    stride_logits_k,
    stride_out0_b,
    stride_out0_k,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)

    row_logits_ptr = sorted_logits_ptr + pid * stride_logits_b
    offsets = tl.arange(0, TOP_K)

    logits = tl.load(row_logits_ptr + offsets * stride_logits_k)

    logits_max = tl.max(logits, 0)
    numerator = tl.exp(logits - logits_max)
    probs = numerator / tl.sum(numerator, 0)
    cum_probs = tl.cumsum(probs, 0)
    to_remove = (cum_probs - probs) > top_p
    to_remove = tl.where(offsets < min_tokens_to_keep, False, to_remove)
    filtered_logits = tl.where(to_remove, filter_value, logits)
    f_logits_max = tl.max(filtered_logits, 0)
    f_numerator = tl.exp(filtered_logits - f_logits_max)
    f_probs = f_numerator / tl.sum(f_numerator, 0)

    row_out_ptr = output_ptr + pid * stride_out0_b
    tl.store(row_out_ptr + offsets * stride_out0_k, f_probs)


def top_p_filter_impl(
    logits: torch.FloatTensor,
    top_p: float = 0.75,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    rand_top_k: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = logits.dtype
    device = logits.device
    logits = logits.to(torch.float32)
    batch_size, _ = logits.shape
    top_k = min(rand_top_k, logits.size(-1))

    sorted_logits, sorted_topk_indices = torch.topk(logits, top_k)

    output_probs = torch.empty((batch_size, top_k), dtype=dtype, device=device)

    grid = (batch_size,)

    _top_p_filter_kernel[grid](
        sorted_logits,
        output_probs,
        top_p,
        filter_value,
        min_tokens_to_keep,
        sorted_logits.stride(0),
        sorted_logits.stride(1),
        output_probs.stride(0),
        output_probs.stride(1),
        TOP_K=top_k,
    )

    return output_probs, sorted_topk_indices

@triton.jit()
def _reject_sampler_kernel(
    output_token_ids_ptr,   # [batch, spec_step + 1]
    output_accept_lens_ptr, # [batch] 
    draft_token_ids_ptr,    # [batch, spec_step]
    draft_probs_ptr,        # [batch, spec_step]
    target_probs_ptr,       # [batch, spec_step + 1, vocab_size]
    max_spec_len: tl.constexpr,
    vocab_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # draft
    batch_draft_token_ids_ptr = draft_token_ids_ptr + batch_idx * max_spec_len
    batch_draft_probs_ptr     = draft_probs_ptr     + batch_idx * max_spec_len
    batch_target_probs_ptr    = target_probs_ptr    + batch_idx * (max_spec_len + 1) * vocab_size

    batch_output_token_ids_ptr   = output_token_ids_ptr   + batch_idx * (max_spec_len + 1)
    batch_output_accept_lens_ptr = output_accept_lens_ptr + batch_idx

    # reject sampler
    accept_len = 0
    rejected = False
    for pos in range(0, max_spec_len):
        if not rejected:
            draft_token_id = tl.load(batch_draft_token_ids_ptr + pos)
            draft_prob     = tl.load(batch_draft_probs_ptr     + pos)
            target_prob    = tl.load(batch_target_probs_ptr    + pos * vocab_size + draft_token_id)

            if draft_prob > 0 and target_prob / draft_prob >= tl.rand(seed=0, offset=0):
                # accept
                accept_len += 1
                tl.store(batch_output_token_ids_ptr + pos, draft_token_id)
            else:
                rejected = True

    tl.store(batch_output_accept_lens_ptr, accept_len)

def reject_sampling_impl(
    target_logits: torch.Tensor, # [batch, spec_step + 1, vocab_size]
    draft_tokens: torch.Tensor,  # [batch, spec_step]
    draft_probs: torch.Tensor,   # [batch, spec_step]
    spec_step: int,
    top_p: float,
    rand_top_k: int,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1, 
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = target_logits.dtype
    device = target_logits.device
    logits = target_logits.to(torch.float32)
    batch_size, _, vocab_size = target_logits.shape
    top_k = min(rand_top_k, logits.size(-1))
    
    sorted_logits, sorted_topk_indices = torch.topk(logits, top_k)

    output_probs = torch.empty((batch_size * (spec_step + 1), top_k), dtype=dtype, device=device)
    output_token_ids = torch.empty((batch_size, spec_step + 1), device=device, dtype=torch.int32)
    output_accept_lens = torch.empty((batch_size), device=device, dtype=torch.int32)

    grid = (batch_size,)

    _top_p_filter_kernel[grid](
        sorted_logits.view(-1, sorted_logits.shape[-1]),
        output_probs,
        top_p,
        filter_value,
        min_tokens_to_keep,
        sorted_logits.stride(0),
        sorted_logits.stride(1),
        output_probs.stride(0),
        output_probs.stride(1),
        TOP_K=top_k,
    )

    target_probs = output_probs.view(batch_size, -1, output_probs.shape[-1])
    target_topk_indices = sorted_topk_indices.view(batch_size, -1, sorted_topk_indices.shape[-1])

    filter_target_probs = torch.zeros_like(target_logits)
    filter_target_probs.scatter_(-1, target_topk_indices, target_probs)

    _reject_sampler_kernel[grid](
        output_token_ids_ptr=output_token_ids,
        output_accept_lens_ptr=output_accept_lens,
        draft_token_ids_ptr=draft_tokens,
        draft_probs_ptr=draft_probs,
        target_probs_ptr=filter_target_probs,
        max_spec_len=spec_step,
        vocab_size=vocab_size,
    )

    last_token_probs = filter_target_probs[range(batch_size), output_accept_lens]
    last_token = torch.multinomial(last_token_probs, num_samples=1)
    output_token_ids.scatter_(-1, output_accept_lens.unsqueeze(1), last_token.int())

    return output_token_ids, output_accept_lens

@triton.jit
def _magic_reject_sampler_kernel(
    output_token_ids_ptr,   # [batch, max_spec_len + 1]
    output_accept_lens_ptr, # [batch] 
    draft_token_ids_ptr,    # [batch, max_spec_len]
    draft_probs_ptr,        # [batch, max_spec_len]
    target_probs_ptr,       # [batch, max_spec_len + 1, vocab_size]
    cum_probs_ptr,          # [batch, max_spec_len]
    cum_rand_ptr,           # [batch, max_spec_len]
    max_spec_len: tl.constexpr,
    vocab_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    # draft
    batch_draft_token_ids_ptr = draft_token_ids_ptr + batch_idx * max_spec_len
    batch_draft_probs_ptr     = draft_probs_ptr     + batch_idx * max_spec_len
    batch_target_probs_ptr    = target_probs_ptr    + batch_idx * (max_spec_len + 1) * vocab_size

    # cum probs
    batch_cum_probs_ptr = cum_probs_ptr + batch_idx * max_spec_len
    batch_cum_rand_ptr  = cum_rand_ptr  + batch_idx * max_spec_len
    
    # output ptr
    batch_output_token_ids_ptr   = output_token_ids_ptr   + batch_idx * (max_spec_len + 1)
    batch_output_accept_lens_ptr = output_accept_lens_ptr + batch_idx

    spec_offset = tl.arange(0, max_spec_len)

    draft_token_ids = tl.load(batch_draft_token_ids_ptr + spec_offset)
    draft_probs     = tl.load(batch_draft_probs_ptr     + spec_offset)
    target_probs    = tl.load(batch_target_probs_ptr    + spec_offset * vocab_size + draft_token_ids)

    ratio = target_probs / draft_probs

    uniform_rand = tl.rand(seed=42, offset=spec_offset)
    cum_probs    = tl.cumprod(ratio, axis=0)
    cum_rands    = tl.cumprod(uniform_rand, axis=0)

    tl.store(batch_cum_probs_ptr + spec_offset, cum_probs)
    tl.store(batch_cum_rand_ptr  + spec_offset, cum_rands)

    accept_len = 0
    for pos in range(0, max_spec_len):
        index = max_spec_len - pos + 1
        cum_prob = tl.load(batch_cum_probs_ptr + index)
        cum_rand = tl.load(batch_cum_rand_ptr + index)

        if cum_prob >= cum_rand:
            accept_len = (index + 1)

    write_mask = spec_offset < accept_len

    tl.store(batch_output_token_ids_ptr+spec_offset, draft_token_ids, mask=write_mask)    
    tl.store(batch_output_accept_lens_ptr, accept_len)        

def magic_reject_sampling_impl(
    target_logits: torch.Tensor, # [batch, spec_step + 1, vocab_size]
    draft_tokens: torch.Tensor,  # [batch, spec_step]
    draft_probs: torch.Tensor,   # [batch, spec_step]
    spec_step: int,
    top_p: float,
    rand_top_k: int,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1, 
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = target_logits.dtype
    device = target_logits.device
    logits = target_logits.to(torch.float32)
    batch_size, _, vocab_size = target_logits.shape
    top_k = min(rand_top_k, logits.size(-1))
    
    sorted_logits, sorted_topk_indices = torch.topk(logits, top_k)

    output_probs = torch.empty((batch_size * (spec_step + 1), top_k), dtype=dtype, device=device)
    output_token_ids = torch.empty((batch_size, spec_step + 1), device=device, dtype=torch.int32)
    output_accept_lens = torch.empty((batch_size), device=device, dtype=torch.int32)

    grid = (batch_size,)

    _top_p_filter_kernel[grid](
        sorted_logits.view(-1, sorted_logits.shape[-1]),
        output_probs,
        top_p,
        filter_value,
        min_tokens_to_keep,
        sorted_logits.stride(0),
        sorted_logits.stride(1),
        output_probs.stride(0),
        output_probs.stride(1),
        TOP_K=top_k,
    )

    target_probs = output_probs.view(batch_size, -1, output_probs.shape[-1])
    target_topk_indices = sorted_topk_indices.view(batch_size, -1, sorted_topk_indices.shape[-1])

    filter_target_probs = torch.zeros_like(target_logits) # [b, step + 1, v]
    filter_target_probs.scatter_(-1, target_topk_indices, target_probs)

    cumsum_prob_buffer =  torch.empty((batch_size, spec_step), device=device, dtype=torch.float32)
    cumsum_rand_buffer =  torch.empty((batch_size, spec_step), device=device, dtype=torch.float32)

    _magic_reject_sampler_kernel[grid](
        output_token_ids_ptr=output_token_ids,
        output_accept_lens_ptr=output_accept_lens,
        draft_token_ids_ptr=draft_tokens,
        draft_probs_ptr=draft_probs,
        target_probs_ptr=filter_target_probs,
        cum_probs_ptr = cumsum_prob_buffer,
        cum_rand_ptr = cumsum_rand_buffer,
        max_spec_len=spec_step,
        vocab_size=vocab_size,
    )

    last_token_probs = filter_target_probs[range(batch_size), output_accept_lens]
    last_token = torch.multinomial(last_token_probs, num_samples=1)
    output_token_ids.scatter_(-1, output_accept_lens.unsqueeze(1), last_token.int())

    return output_token_ids, output_accept_lens

@triton.jit
def _fused_penalty_temp_kernel(
    Logits_ptr,
    Freqs_ptr,
    Is_present_ptr,
    Freq_pen_ptr,
    Pres_pen_ptr,
    Rep_pen_ptr,
    Temp_ptr,
    stride_logits_b,
    stride_logits_v,
    stride_freqs_b,
    stride_freqs_v,
    stride_is_present,
    stride_freq_pen,
    stride_pres_pen,
    stride_rep_pen,
    stride_temp,
    n_batch,
    n_vocab,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_vocab_blocks = tl.cdiv(n_vocab, BLOCK_SIZE)

    total_tasks = n_batch * num_vocab_blocks

    for task_id in range(pid, total_tasks, grid_size):
        pid_b = task_id // num_vocab_blocks
        pid_v = task_id % num_vocab_blocks

        is_present_float = tl.load(Is_present_ptr + pid_b * stride_is_present)

        freq_pen = tl.load(Freq_pen_ptr + pid_b * stride_freq_pen)
        pres_pen = tl.load(Pres_pen_ptr + pid_b * stride_pres_pen)
        rep_pen = tl.load(Rep_pen_ptr + pid_b * stride_rep_pen)

        temperature = 1.0
        if Temp_ptr is not None:
            temperature = tl.load(Temp_ptr + pid_b * stride_temp)

        offs_v = pid_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs_v < n_vocab

        logit_ptrs = Logits_ptr + (pid_b * stride_logits_b) + (offs_v * stride_logits_v)
        freq_ptrs = Freqs_ptr + (pid_b * stride_freqs_b) + (offs_v * stride_freqs_v)

        logits = tl.load(logit_ptrs, mask=mask, other=0.0).to(tl.float32)

        token_freqs = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if is_present_float != 0.0:
            token_freqs = tl.load(freq_ptrs, mask=mask, other=0.0).to(tl.float32)

            if freq_pen != 0.0:
                logits = logits - (freq_pen * token_freqs)

            if pres_pen != 0.0:
                is_present = token_freqs > 0
                logits = logits - (pres_pen * is_present.to(tl.float32))

            if rep_pen != 1.0:
                has_freq = token_freqs > 0

                logits = tl.where(has_freq & (logits > 0), logits / rep_pen, logits)

                logits = tl.where(has_freq & (logits < 0), logits * rep_pen, logits)

        if Temp_ptr is not None:
            logits = logits / temperature

        tl.store(logit_ptrs, logits, mask=mask)


def fused_penalties_temp_impl(
    logits: torch.Tensor,
    token_freqs,
    frequency_penalties: torch.Tensor,
    presence_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    temperatures: torch.Tensor = None,
):
    assert logits.dim() == 2, "Logits must be [Batch, Vocab]"

    batch_size, n_vocab = logits.shape

    def prepare_scalar_tensor(t, name):
        if isinstance(t, list):
            t = torch.tensor(t, device=logits.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        if t.dim() > 1:
            t = t.view(-1)
        assert t.size(0) == batch_size, f"{name} batch size mismatch"
        return t.contiguous()

    f_pen = prepare_scalar_tensor(frequency_penalties, "freq_pen")
    p_pen = prepare_scalar_tensor(presence_penalties, "pres_pen")
    r_pen = prepare_scalar_tensor(repetition_penalties, "rep_pen")

    t_ptr = None
    stride_temp = 0
    if temperatures is not None:
        t_vals = prepare_scalar_tensor(temperatures, "temperature")
        t_ptr = t_vals
        stride_temp = t_vals.stride(0)

    is_present_list = [1.0 if t is not None else 0.0 for t in token_freqs]

    is_present_mask = torch.tensor(is_present_list, device=logits.device, dtype=torch.float32).contiguous()
    stride_is_present = is_present_mask.stride(0)

    first_non_none = next((t for t in token_freqs if t is not None), None)
    freq_dtype = first_non_none.dtype if first_non_none is not None else torch.int64

    dense_token_freqs = torch.zeros((batch_size, n_vocab), dtype=freq_dtype, device=logits.device)

    for i, freq_tensor in enumerate(token_freqs):
        if freq_tensor is not None:
            dense_token_freqs[i, :] = freq_tensor.to(dense_token_freqs.device, non_blocking=True).view(-1)

    logits = logits.contiguous()
    dense_token_freqs = dense_token_freqs.contiguous()

    BLOCK_SIZE = 1024

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _fused_penalty_temp_kernel[grid](
        logits,
        dense_token_freqs,
        is_present_mask,
        f_pen,
        p_pen,
        r_pen,
        t_ptr,
        logits.stride(0),
        logits.stride(1),
        dense_token_freqs.stride(0),
        dense_token_freqs.stride(1),
        stride_is_present,
        f_pen.stride(0),
        p_pen.stride(0),
        r_pen.stride(0),
        stride_temp,
        batch_size,
        n_vocab,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return logits
