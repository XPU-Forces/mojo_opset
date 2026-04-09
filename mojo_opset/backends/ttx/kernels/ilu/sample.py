import torch
import triton
import triton.language as tl

from .utils import libentry


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
    n = batch_size * triton.cdiv(n_vocab, BLOCK_SIZE)
    grid = (n,)

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


# --- Top-p sampling / filter, reject sampling (aligned with NPU Triton; ILU uses @libentry) ---


@libentry()
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
    top_k_valid,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_logits_ptr = sorted_logits_ptr + pid * stride_logits_b
    offsets = tl.arange(0, TOP_K)
    mask_k = offsets < top_k_valid
    logits = tl.load(row_logits_ptr + offsets * stride_logits_k, mask=mask_k, other=-float("inf"))
    logits_max = tl.max(logits, 0)
    numerator = tl.exp(logits - logits_max)
    probs = numerator / tl.sum(numerator, 0)
    cum_probs = tl.cumsum(probs, 0)
    to_remove = (cum_probs - probs) > top_p
    to_remove = tl.where(offsets < min_tokens_to_keep, False, to_remove)
    to_remove = tl.where(mask_k, to_remove, True)
    filtered_logits = tl.where(to_remove, filter_value, logits)
    f_logits_max = tl.max(filtered_logits, 0)
    f_numerator = tl.exp(filtered_logits - f_logits_max)
    f_probs = f_numerator / tl.sum(f_numerator, 0)
    row_out_ptr = output_ptr + pid * stride_out0_b
    tl.store(row_out_ptr + offsets * stride_out0_k, f_probs, mask=mask_k)


def top_p_sampling_impl(
    logits: torch.Tensor,
    top_p: float = 0.75,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    rand_top_k: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match :class:`MojoTopPSampling` (torch reference)."""
    logits = logits.to(torch.float32)
    top_k = min(rand_top_k, logits.size(-1))
    sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)
    cumulative_probs = sorted_topk_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    filtered_logits = sorted_topk_logits.masked_fill(sorted_indices_to_remove, filter_value)
    final_probs_dist = torch.nn.functional.softmax(filtered_logits, dim=-1)
    select_index = torch.multinomial(final_probs_dist, num_samples=1)
    next_tokens = torch.gather(sorted_topk_indices, dim=-1, index=select_index)
    next_probs = torch.gather(final_probs_dist, dim=-1, index=select_index)
    return next_probs, next_tokens


def top_p_filter_impl(
    logits: torch.Tensor,
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
    top_k_pad = triton.next_power_of_2(top_k)
    if top_k_pad != top_k:
        sorted_logits = torch.nn.functional.pad(sorted_logits, (0, top_k_pad - top_k), value=-float("inf"))
        sorted_topk_indices = torch.nn.functional.pad(sorted_topk_indices, (0, top_k_pad - top_k), value=0)
    output_probs = torch.empty((batch_size, top_k_pad), dtype=torch.float32, device=device)
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
        top_k_valid=top_k,
        TOP_K=top_k_pad,
    )
    return output_probs[:, :top_k].to(dtype), sorted_topk_indices[:, :top_k]


@libentry()
@triton.jit
def _reject_sampler_kernel(
    output_token_ids_ptr,
    output_accept_lens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    uniform_random_ptr,
    max_spec_len: tl.constexpr,
    vocab_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    batch_draft_token_ids_ptr = draft_token_ids_ptr + batch_idx * max_spec_len
    batch_draft_probs_ptr = draft_probs_ptr + batch_idx * max_spec_len
    batch_target_probs_ptr = target_probs_ptr + batch_idx * (max_spec_len + 1) * vocab_size
    batch_output_token_ids_ptr = output_token_ids_ptr + batch_idx * (max_spec_len + 1)
    batch_output_accept_lens_ptr = output_accept_lens_ptr + batch_idx
    batch_uniform_random = tl.load(uniform_random_ptr + batch_idx)
    accept_len = 0
    rejected = False
    for pos in range(0, max_spec_len):
        if not rejected:
            draft_token_id = tl.load(batch_draft_token_ids_ptr + pos)
            draft_prob = tl.load(batch_draft_probs_ptr + pos)
            target_prob = tl.load(batch_target_probs_ptr + pos * vocab_size + draft_token_id)
            if draft_prob > 0 and target_prob / draft_prob >= batch_uniform_random:
                accept_len += 1
                tl.store(batch_output_token_ids_ptr + pos, draft_token_id)
            else:
                rejected = True
    tl.store(batch_output_accept_lens_ptr, accept_len)


def reject_sampling_impl(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    random_seed,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = target_probs.device
    batch_size, _, vocab_size = target_probs.shape
    spec_step = draft_probs.shape[1]
    output_token_ids = torch.empty((batch_size, spec_step + 1), device=device, dtype=torch.int32)
    output_accept_lens = torch.empty((batch_size), device=device, dtype=torch.int32)
    if random_seed is not None:
        torch.manual_seed(random_seed)
    rand_vals = torch.rand(batch_size, 1, device=target_probs.device)
    grid = (batch_size,)
    _reject_sampler_kernel[grid](
        output_token_ids,
        output_accept_lens,
        draft_tokens,
        draft_probs,
        target_probs,
        rand_vals,
        max_spec_len=spec_step,
        vocab_size=vocab_size,
    )
    next_tokens = torch.cat(
        [draft_tokens.to(torch.long), torch.zeros((batch_size, 1), dtype=torch.long, device=device)], dim=-1
    )
    return next_tokens, output_accept_lens


def join_prob_reject_sampling_impl(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    random_seed,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match MojoJoinProbRejectSampling.forward numerics (device-side PyTorch; prefix logic is non-trivial to bit-match in Triton)."""
    batch_size, _, _ = target_probs.shape
    spec_step = draft_probs.shape[1]
    device = target_probs.device
    target_token_probs = torch.gather(target_probs[:, :spec_step, :], -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    ratios = torch.minimum(torch.ones_like(target_token_probs), target_token_probs / draft_probs)
    pi = torch.cumprod(ratios, dim=1)
    if random_seed is not None:
        torch.manual_seed(random_seed)
    rand_u = torch.rand(batch_size, spec_step, device=device)
    rand_cum = torch.cumprod(rand_u, dim=1)
    reject_matrix = pi < rand_cum
    reject_matrix = torch.cat([torch.zeros((batch_size, 1), device=device), reject_matrix.int()], dim=1)
    accepted_len = spec_step - reject_matrix.flip(dims=[1]).argmin(dim=1).int()
    next_tokens = torch.cat(
        [draft_tokens.to(torch.long), torch.zeros((batch_size, 1), dtype=torch.long, device=device)], dim=-1
    )
    return next_tokens, accepted_len.int()
