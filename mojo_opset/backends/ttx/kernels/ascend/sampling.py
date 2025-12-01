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


def ttx_top_p_sampling(
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
