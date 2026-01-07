import os
import torch
import math


def test_paged_prefill_quest():
    qkv = torch.randn(24, 1280).bfloat16()
    key_cache = torch.randn(1024, 1, 32, 128).bfloat16()
    value_cache = torch.randn(1024, 1, 32, 128).bfloat16()
    causal_mask = torch.ones(1024, 1024, dtype=torch.bool).tril(diagonal=0)
    q_head_num = 8
    kv_head_num = 1
    kv_idx = [torch.tensor([0, 1, 2, 3])]
    kv_len = [32 * 3]
    q_len_list = [17]
    sparse_limit = 64
    original_out = original_session_cache_pa_flash_attention_quest128(
        qkv,
        key_cache,
        value_cache,
        causal_mask,
        q_head_num,
        kv_head_num,
        kv_idx,
        kv_len,
        q_len_list,
        0,
        0,
        sparse_limit,
    )
    mojo_output = mojo_quest(
        qkv,
        key_cache,
        value_cache,
        causal_mask,
        q_head_num,
        kv_head_num,
        kv_idx,
        kv_len,
        q_len_list,
        0,
        0,
        sparse_limit,
    )
    torch.testing.assert_close(mojo_output, original_out)


def original_session_cache_pa_flash_attention_quest128(
    qkv,  # [24,1280]
    key_cache,  # [1024,1,256,128]
    value_cache,  # [1024,1,256,128]
    causal_mask,  # [2048,2048]
    q_head_num,  # 8
    kv_head_num,  # 1
    kv_idx,  # [0,]
    kv_len,  # [0,]
    q_len_list,  # [17,0,..]
    global_rank,
    block_idx,
    sparse_limit,
):

    prefill_sparse = True
    # sparse_limit = 1024
    # page_size = 64
    page_size = key_cache.shape[2] // kv_head_num
    # q_seg_size = 512
    q_seg_size = int(os.environ.get("Q_SEG_SIZE", 1024))
    topk_ratio = float(os.environ.get("TOPK_RATIO", 0.25))
    page_rep = os.environ.get("PAGE_REP", "default_value")

    # cache_size = key_cache.shape[0] # 1024
    q_seq_length = qkv.shape[0]
    # kv_seq_length = key_cache.shape[2]
    head_size = key_cache.shape[-1]
    kv_cache_indices = kv_idx
    kv_seq_lengths = kv_len
    q_chunk_sizes = q_len_list

    query = (
        qkv.reshape(1, q_seq_length, (q_head_num + 2 * kv_head_num), head_size)
        .permute(0, 2, 1, 3)
        .contiguous()[:, :q_head_num, :, :]
        .contiguous()
    )  # [1,8,24,128]

    # [total_pages, h_qo, page_size, d]
    key_cache = key_cache.repeat_interleave(q_head_num // kv_head_num, dim=1)
    # [total_pages, h_qo, page_size, d]
    value_cache = value_cache.repeat_interleave(q_head_num // kv_head_num, dim=1)

    key_cache_list = []
    value_cache_list = []
    for sublist in kv_cache_indices:
        valid_mask = sublist != -1
        valid_indices = sublist[valid_mask]
        if valid_indices.numel() > 0:
            key_sub = key_cache.index_select(0, valid_indices)
            value_sub = value_cache.index_select(0, valid_indices)
            key_cache_list.append(key_sub)
            value_cache_list.append(value_sub)

    key_cache_new = []
    value_cache_new = []
    for k_sublist, v_sublist in zip(key_cache_list, value_cache_list):
        k_sublist_t = list(k_sublist)
        key_cache_new.append(torch.cat(k_sublist_t, dim=1))
        v_sublist_t = list(v_sublist)
        value_cache_new.append(torch.cat(v_sublist_t, dim=1))

    query_start = 0
    expects = []
    for i in range(len(kv_cache_indices)):

        valid_kv_seq_length = kv_seq_lengths[i] + q_chunk_sizes[i]  # 19

        top_k = int(valid_kv_seq_length * topk_ratio)
        top_k_page = top_k // page_size

        key_cache_i = key_cache_new[i]
        key = key_cache_i[:, :valid_kv_seq_length, :]  # [8,128,19]

        value_cache_i = value_cache_new[i]
        value = value_cache_i[:, :valid_kv_seq_length, :].contiguous()  # [8,19,128]

        whole_causal = torch.tril(
            torch.ones((q_chunk_sizes[i], valid_kv_seq_length), dtype=torch.bool, device=key.device),
            diagonal=valid_kv_seq_length - q_chunk_sizes[i],
        )
        if prefill_sparse and kv_len[i] > 0 and valid_kv_seq_length > sparse_limit:
            num_pages = valid_kv_seq_length // page_size
            pad_len = valid_kv_seq_length - num_pages * page_size

            # [num_heads, num_pages, chunk_size, head_dim]
            page_k = key[:, : num_pages * page_size].reshape(q_head_num, num_pages, page_size, head_size)
            # [num_heads, num_pages, head_dim]
            mins = page_k.min(dim=2).values
            maxs = page_k.max(dim=2).values

            num_q_seg = (q_chunk_sizes[i] + q_seg_size - 1) // q_seg_size
            for q_seg_id in range(num_q_seg):
                q_seg_start = query_start + q_seg_id * q_seg_size
                q_seg_end = min(q_seg_start + q_seg_size, query_start + q_chunk_sizes[i])
                curr_seg_size = q_seg_end - q_seg_start
                curr_query_seg = query.reshape(-1, q_seq_length, head_size)[:, q_seg_start:q_seg_end, :]

                # ====================quest========================
                # [num_heads, q_len, 1, head_size] * [num_heads, 1, num_pages, head_dim]
                q_min_k = curr_query_seg.unsqueeze(-2) * mins.unsqueeze(-3)
                q_max_k = curr_query_seg.unsqueeze(-2) * maxs.unsqueeze(-3)

                # [num_heads, q_len, num_pages]
                page_score = torch.maximum(q_min_k, q_max_k).sum(dim=-1)
                # [nh, ql, top_k_page]
                _, topk_page_indices = page_score.topk(top_k_page, dim=-1)

                # ====================使用相同page========================
                seq_len_t = page_score.shape[1]
                topk_page_indices_0 = topk_page_indices[:, 0]
                topk_page_indices = topk_page_indices_0.unsqueeze(1).repeat(1, seq_len_t, 1)
                # ====================使用相同page========================

                # [nh, ql, topk_page, page_size]
                topk_token_indices = (topk_page_indices * page_size).unsqueeze(-1).repeat(
                    1, 1, 1, page_size
                ) + torch.arange(page_size, device=topk_page_indices.device)
                topk_token_indices = topk_token_indices.reshape(q_head_num, curr_seg_size, top_k_page * page_size)

                pad_indices = num_pages * page_size + torch.arange(pad_len, device=topk_token_indices.device)
                # [nh, ql, pad_len]
                pad_indices = pad_indices.expand(q_head_num, curr_seg_size, -1)
                # [nh, ql, topk_page * page_size + pad_len]
                topk_token_indices = torch.cat([topk_token_indices, pad_indices], dim=-1)

                # ====================quest========================

                # [nh, q_seg_size, kv_seq_length]
                curr_seg_score = torch.bmm(curr_query_seg, key.transpose(-2, -1))
                curr_seg_score = curr_seg_score / math.sqrt(head_size)

                # [nh, q_seg_size, kv_seq_length]
                curr_seg_mask = torch.zeros_like(curr_seg_score, dtype=torch.bool)
                curr_seg_mask.scatter_(dim=-1, index=topk_token_indices[:, q_seg_start:q_seg_end, :], value=True)
                # curr_seg_causal = torch.tril(torch.ones((q_head_num, curr_seg_size, curr_seg_size), dtype= torch.bool, device=curr_seg_mask.device))
                curr_seg_causal = whole_causal[
                    q_seg_id * q_seg_size : q_seg_id * q_seg_size + curr_seg_size, -q_chunk_sizes[i] :
                ]
                # print(f"{curr_seg_mask.shape=} {q_seg_start=} {q_seg_end=}", flush=True)
                curr_seg_mask[:, :, -q_chunk_sizes[i] :] = curr_seg_causal

                curr_seg_score = curr_seg_score.masked_fill(~curr_seg_mask, torch.finfo(curr_seg_score.dtype).min)
                curr_seg_score = torch.softmax(curr_seg_score, -1, dtype=torch.float32).to(dtype=torch.bfloat16)
                # [nh, q_seg_size, head_size]
                curr_seg_output = (
                    torch.bmm(curr_seg_score, value)
                    .permute(1, 0, 2)
                    .reshape(curr_seg_size, q_head_num * head_size)
                    .to(dtype=torch.bfloat16)
                )
                if q_seg_id == 0:
                    output_all = curr_seg_output
                else:
                    output_all = torch.cat([output_all, curr_seg_output], axis=0)

            expects.append(output_all)
        else:
            assert False, "should not happen"
            num_q_seg = (q_chunk_sizes[i] + q_seg_size - 1) // q_seg_size
            for q_seg_id in range(num_q_seg):
                q_seg_start = query_start + q_seg_id * q_seg_size
                q_seg_end = min(q_seg_start + q_seg_size, query_start + q_chunk_sizes[i])
                curr_seg_size = q_seg_end - q_seg_start
                # [nh, q_seg_size, hd]
                # curr_query_seg = curr_query[:, q_seg_start:q_seg_end, :]
                curr_query_seg = query.reshape(-1, q_seq_length, head_size)[:, q_seg_start:q_seg_end, :]
                # [nh, q_seg_size, kv_seq_length]
                curr_seg_score = torch.bmm(curr_query_seg, key.transpose(-2, -1)) / math.sqrt(head_size)
                # curr_seg_causal = torch.tril(torch.ones((q_head_num, curr_seg_size, valid_kv_seq_length), dtype= torch.bool, device=curr_seg_score.device), diagonal=valid_kv_seq_length-curr_seg_size)
                curr_seg_causal = whole_causal[q_seg_id * q_seg_size : q_seg_id * q_seg_size + curr_seg_size, :]
                # curr_seg_score = curr_seg_score.masked_fill(~curr_seg_causal, torch.finfo(curr_seg_score.dtype).min)
                reverse_mask = (~curr_seg_causal).float() * torch.finfo(curr_seg_score.dtype).min
                curr_seg_score += reverse_mask

                curr_seg_score = torch.softmax(curr_seg_score, -1).to(dtype=torch.bfloat16)
                # [q_seg_size, nh*head_size]
                curr_seg_output = (
                    torch.bmm(curr_seg_score, value)
                    .permute(1, 0, 2)
                    .reshape(curr_seg_size, q_head_num * head_size)
                    .to(dtype=torch.bfloat16)
                )

                if q_seg_id == 0:
                    output_all = curr_seg_output
                else:
                    output_all = torch.cat([output_all, curr_seg_output], axis=0)

            expects.append(output_all)

        query_start += q_chunk_sizes[i]

    tmp = (
        torch.zeros((q_seq_length - sum(q_chunk_sizes), q_head_num * head_size))
        .to(dtype=expects[0].dtype)
        .to(expects[0].device)
    )
    expects.append(tmp)

    expect = torch.cat(expects, axis=0)
    return expect


def mojo_quest(
    qkv,  # [24,1280]
    key_cache,  # [1024,1,256,128]
    value_cache,  # [1024,1,256,128]
    causal_mask,  # [2048,2048]
    q_head_num,  # 8
    kv_head_num,  # 1
    kv_idx,  # [0,]
    kv_len,  # [0,]
    q_len_list,  # [17,0,..]
    global_rank,
    block_idx,
    sparse_limit,
):
    prefill_sparse = True
    # sparse_limit = 1024
    # page_size = 64
    page_size = key_cache.shape[2] // kv_head_num
    # q_seg_size = 512
    q_seg_size = int(os.environ.get("Q_SEG_SIZE", 1024))
    topk_ratio = float(os.environ.get("TOPK_RATIO", 0.25))
    page_rep = os.environ.get("PAGE_REP", "default_value")

    # cache_size = key_cache.shape[0] # 1024
    q_seq_length = qkv.shape[0]
    # kv_seq_length = key_cache.shape[2]
    head_size = key_cache.shape[-1]
    kv_cache_indices = kv_idx
    kv_seq_lengths = kv_len
    q_chunk_sizes = q_len_list

    query = (
        qkv.reshape(1, q_seq_length, (q_head_num + 2 * kv_head_num), head_size)
        .permute(0, 2, 1, 3)
        .contiguous()[:, :q_head_num, :, :]
        .contiguous()
    )  # [1,8,24,128]

    from mojo_opset import MojoPagedPrefillQuest

    mojo_quest_op = MojoPagedPrefillQuest(None, page_size, q_seg_size, topk_ratio)
    return mojo_quest_op(query, key_cache, value_cache, torch.tensor(q_len_list), kv_cache_indices, kv_seq_lengths)


if __name__ == "__main__":
    test_paged_prefill_quest()
