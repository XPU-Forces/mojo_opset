import torch
import triton
import triton.language as tl


@triton.jit
def _sample_kernel(
    sorted_logits_ptr,
    sorted_indices_ptr,
    thresholds_ptr,
    output_tokens_ptr,
    output_probs_ptr,
    top_p,
    filter_value,
    min_tokens_to_keep,
    stride_logits_b,
    stride_logits_k,
    stride_indices_b,
    stride_indices_k,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)

    row_logits_ptr = sorted_logits_ptr + pid * stride_logits_b
    row_indices_ptr = sorted_indices_ptr + pid * stride_indices_b

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

    threshold = tl.load(thresholds_ptr + pid)
    f_cum_probs = tl.cumsum(f_probs, 0)

    is_candidate = f_cum_probs >= threshold

    candidate_indices = tl.where(is_candidate, offsets, TOP_K)
    sampled_index_in_topk = tl.min(candidate_indices, 0)

    sampled_index_in_topk = tl.where(sampled_index_in_topk == TOP_K, 0, sampled_index_in_topk)

    is_selected_mask = offsets == sampled_index_in_topk
    selected_prob_val = tl.sum(tl.where(is_selected_mask, f_probs, 0.0), 0)

    all_topk_indices = tl.load(row_indices_ptr + offsets * stride_indices_k)
    selected_token_val = tl.sum(tl.where(is_selected_mask, all_topk_indices, 0), 0)

    tl.store(output_tokens_ptr + pid, selected_token_val)
    tl.store(output_probs_ptr + pid, selected_prob_val)


def ttx_topp_sampling(
    logits: torch.FloatTensor,
    thresholds: torch.FloatTensor,
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

    output_tokens = torch.empty((batch_size, 1), dtype=torch.long, device=device)
    output_probs = torch.empty((batch_size, 1), dtype=torch.float32, device=device)

    grid = (batch_size,)

    _sample_kernel[grid](
        sorted_logits,
        sorted_topk_indices,
        thresholds,
        output_tokens,
        output_probs,
        top_p,
        filter_value,
        min_tokens_to_keep,
        sorted_logits.stride(0),
        sorted_logits.stride(1),
        sorted_topk_indices.stride(0),
        sorted_topk_indices.stride(1),
        TOP_K=top_k,
    )

    return output_probs, output_tokens
