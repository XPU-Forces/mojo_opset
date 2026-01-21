import os
import torch
import math


def test_paged_prefill_quest():
    MAX_QLEN = 16384
    HEAD_DIM = 128
    Q_HEAD_NUM = 8
    KV_HEAD_NUM = 1
    PAGE_SIZE = 128
    MAX_PAGE = 256
    Q_SEG_SIZE = int(os.environ.get("Q_SEG_SIZE", 1024))
    TOPK_RATIO = float(os.environ.get("TOPK_RATIO", 0.25))
    RECENT_WINDOW = int(os.environ.get("RECENT_WINDOW", 0))
    assert RECENT_WINDOW >= 0
    page_rep = os.environ.get("PAGE_REP", "default_value")
    sparse_limit = 64

    qkv = torch.randn(MAX_QLEN, HEAD_DIM * (Q_HEAD_NUM + KV_HEAD_NUM * 2)).npu()  # .bfloat16()
    key_cache = torch.randn(MAX_PAGE, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM).npu()  # .bfloat16()
    value_cache = torch.randn(MAX_PAGE, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM).npu()  # .bfloat16()

    BSZ = 3
    kv_len0 = 2122
    q_len0 = 999
    kv_idx0 = torch.tensor(list(range((kv_len0 + q_len0 + PAGE_SIZE - 1) // PAGE_SIZE)), device=key_cache.device)
    kv_len = [kv_len0 for _ in range(BSZ)]
    q_len_list = [q_len0 for _ in range(BSZ)]
    kv_idx = [kv_idx0 * BSZ + i for i in range(BSZ)]

    original_out = original_session_cache_pa_flash_attention_quest128(
        qkv,
        key_cache,
        value_cache,
        Q_HEAD_NUM,
        KV_HEAD_NUM,
        kv_idx,
        kv_len,
        q_len_list,
        0,
        0,
        Q_SEG_SIZE,
        TOPK_RATIO,
        RECENT_WINDOW,
        sparse_limit,
    )
    # mojo_output = mojo_quest(
    #     qkv,
    #     key_cache,
    #     value_cache,
    #     Q_HEAD_NUM,
    #     KV_HEAD_NUM,
    #     kv_idx,
    #     kv_len,
    #     q_len_list,
    #     0,
    #     0,
    #     Q_SEG_SIZE,
    #     TOPK_RATIO,
    #     sparse_limit,
    # )
    # torch.testing.assert_close(mojo_output, original_out)
    mojo_output_v2 = mojo_block_quest(
        qkv,
        key_cache,
        value_cache,
        Q_HEAD_NUM,
        KV_HEAD_NUM,
        kv_idx,
        kv_len,
        q_len_list,
        0,
        0,
        Q_SEG_SIZE,
        TOPK_RATIO,
        RECENT_WINDOW,
        sparse_limit,
    )
    torch.testing.assert_close(mojo_output_v2, original_out)
    print("PASS!!")


def original_session_cache_pa_flash_attention_quest128(
    qkv,  # [24,1280]
    key_cache,  # [1024,1,256,128]
    value_cache,  # [1024,1,256,128]
    q_head_num,  # 8
    kv_head_num,  # 1
    kv_idx,  # [0,]
    kv_len,  # [0,]
    q_len_list,  # [17,0,..]
    global_rank,
    block_idx,
    q_seg_size,
    topk_ratio,
    RECENT_WINDOW,
    sparse_limit,
):
    topk_page_indices_debug = None
    prefill_sparse = True
    # sparse_limit = 1024
    # page_size = 64
    page_size = key_cache.shape[2]
    # q_seg_size = 512

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

    query_start = 0
    expects = []
    for i in range(len(kv_cache_indices)):
        sublist = kv_cache_indices[i]
        valid_mask = sublist != -1
        valid_indices = sublist[valid_mask]

        key_sub = key_cache.index_select(0, valid_indices)
        value_sub = value_cache.index_select(0, valid_indices)

        valid_kv_seq_length = kv_seq_lengths[i] + q_chunk_sizes[i]  # 19

        num_pages = max(0, kv_seq_lengths[i] - RECENT_WINDOW) // page_size
        pad_len = kv_seq_lengths[i] - num_pages * page_size

        top_k = int(valid_kv_seq_length * topk_ratio)
        top_k_page = min(top_k // page_size, num_pages)

        key_cache_i = key_sub.permute(1, 0, 2, 3).reshape(kv_head_num, -1, head_size)
        key = (
            key_cache_i[:, :valid_kv_seq_length, :].repeat_interleave(q_head_num // kv_head_num, dim=0).contiguous()
        )  # [8,19, 128]

        value_cache_i = value_sub.permute(1, 0, 2, 3).reshape(kv_head_num, -1, head_size)
        value = (
            value_cache_i[:, :valid_kv_seq_length, :].repeat_interleave(q_head_num // kv_head_num, dim=0).contiguous()
        )  # [8,19,128]

        whole_causal = torch.tril(
            torch.ones((q_chunk_sizes[i], valid_kv_seq_length), dtype=torch.bool, device=key.device),
            diagonal=valid_kv_seq_length - q_chunk_sizes[i],
        )
        if prefill_sparse and kv_len[i] > 0 and valid_kv_seq_length > sparse_limit:

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
                q_min_k = curr_query_seg.float().unsqueeze(-2) * mins.float().unsqueeze(-3)
                q_max_k = curr_query_seg.float().unsqueeze(-2) * maxs.float().unsqueeze(-3)

                # [num_heads, q_len, num_pages]
                page_score = torch.maximum(q_min_k, q_max_k).sum(dim=-1)
                # [nh, ql, top_k_page]
                _, topk_page_indices = page_score.topk(top_k_page, dim=-1)

                # ====================使用相同page========================
                seq_len_t = page_score.shape[1]
                topk_page_indices_0 = topk_page_indices[:, 0]
                if topk_page_indices_debug is None:
                    topk_page_indices_debug = topk_page_indices_0
                else:
                    topk_page_indices_debug = torch.cat([topk_page_indices_debug, topk_page_indices_0], dim=1)
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
                curr_seg_score = torch.bmm(curr_query_seg.float(), key.float().transpose(-2, -1))
                curr_seg_score = curr_seg_score / math.sqrt(head_size)

                # [nh, q_seg_size, kv_seq_length]
                curr_seg_mask = torch.zeros_like(curr_seg_score, dtype=torch.bool)
                curr_seg_mask.scatter_(dim=-1, index=topk_token_indices, value=True)
                # curr_seg_causal = torch.tril(torch.ones((q_head_num, curr_seg_size, curr_seg_size), dtype= torch.bool, device=curr_seg_mask.device))
                curr_seg_causal = whole_causal[
                    q_seg_id * q_seg_size : q_seg_id * q_seg_size + curr_seg_size, -q_chunk_sizes[i] :
                ]
                # print(f"{curr_seg_mask.shape=} {q_seg_start=} {q_seg_end=}", flush=True)
                curr_seg_mask[:, :, -q_chunk_sizes[i] :] = curr_seg_causal

                curr_seg_score = curr_seg_score.masked_fill(~curr_seg_mask, torch.finfo(curr_seg_score.dtype).min)
                curr_seg_score = torch.softmax(curr_seg_score, -1, dtype=torch.float32)  # .to(dtype=torch.bfloat16)
                # [nh, q_seg_size, head_size]
                curr_seg_output = (
                    torch.bmm(curr_seg_score, value.float())
                    .permute(1, 0, 2)
                    .reshape(curr_seg_size, q_head_num * head_size)
                    .to(dtype=qkv.dtype)
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
    return topk_page_indices_debug, expect


def mojo_quest(
    qkv,  # [24,1280]
    key_cache,  # [1024,1,256,128]
    value_cache,  # [1024,1,256,128]
    q_head_num,  # 8
    kv_head_num,  # 1
    kv_idx,  # [0,]
    kv_len,  # [0,]
    q_len_list,  # [17,0,..]
    global_rank,
    block_idx,
    q_seg_size,  # 1024
    topk_ratio,
    RECENT_WINDOW,
    sparse_limit,
):
    topk_page_indices_debug = None
    prefill_sparse = True
    # sparse_limit = 1024
    # page_size = 64
    page_size = key_cache.shape[2]
    # q_seg_size = 512

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

    # mojo_quest_op(query, key_cache, value_cache, torch.tensor(q_len_list), kv_cache_indices, kv_seq_lengths)
    query_lengths = torch.tensor(q_len_list, device=query.device)
    bsz, q_head_num, q_seq_length, head_size = query.shape
    assert bsz == 1
    query = query.squeeze(0)
    kv_head_num = key_cache.shape[1]
    expects = []
    q_chunk_sizes = query_lengths.tolist()
    cu_seqlen_q = torch.cumsum(query_lengths, dim=0)
    cu_seqlen_q = torch.nn.functional.pad(cu_seqlen_q, (1, 0), value=0)
    cu_seqlen_q = cu_seqlen_q.tolist()
    for i in range(len(kv_cache_indices)):

        valid_kv_seq_length = kv_seq_lengths[i] + q_chunk_sizes[i]  # 19
        top_k = int(valid_kv_seq_length * topk_ratio)
        top_k_page = top_k // page_size

        query_start = cu_seqlen_q[i]

        sublist = kv_cache_indices[i]
        valid_mask = sublist != -1
        valid_indices = sublist[valid_mask]

        key_sub = key_cache.index_select(0, valid_indices)
        value_sub = value_cache.index_select(0, valid_indices)

        key_cache_i = key_sub.permute(1, 0, 2, 3).reshape(kv_head_num, -1, head_size)
        key = (
            key_cache_i[:, :valid_kv_seq_length, :].repeat_interleave(q_head_num // kv_head_num, dim=0).contiguous()
        )  # [8,19, 128]

        value_cache_i = value_sub.permute(1, 0, 2, 3).reshape(kv_head_num, -1, head_size)
        value = (
            value_cache_i[:, :valid_kv_seq_length, :].repeat_interleave(q_head_num // kv_head_num, dim=0).contiguous()
        )  # [8,19,128]

        whole_causal = torch.tril(
            torch.ones((q_chunk_sizes[i], valid_kv_seq_length), dtype=torch.bool, device=key.device),
            diagonal=valid_kv_seq_length - q_chunk_sizes[i],
        )

        # if prefill_sparse and kv_len[i] > 0 and valid_kv_seq_length > sparse_limit:

        num_pages = valid_kv_seq_length // page_size

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
            curr_query_seg = query[:, q_seg_start:q_seg_end, :]

            # ====================block-wise quest========================

            from mojo_opset import MojoQuest

            mojo_quest_op = MojoQuest()
            topk_page_indices = mojo_quest_op(
                curr_query_seg[:, :1],
                mins,
                maxs,
                top_k_page,
            )
            topk_page_indices_0 = topk_page_indices[:, 0]
            if topk_page_indices_debug is None:
                topk_page_indices_debug = topk_page_indices_0
            else:
                topk_page_indices_debug = torch.cat([topk_page_indices_debug, topk_page_indices_0], dim=1)

            # ====================block-wise quest========================
            from mojo_opset import MojoBlockSparseAttention

            block_sparse_attention = MojoBlockSparseAttention(
                whole_causal, page_size, q_seg_size, topk_ratio, head_size, q_head_num, kv_head_num
            )

            curr_seg_output = block_sparse_attention(
                curr_query_seg,
                key,
                value,
                whole_causal,
                topk_page_indices_0,
                q_seg_id,
                q_chunk_sizes[i],
            )
            if q_seg_id == 0:
                output_all = curr_seg_output
            else:
                output_all = torch.cat([output_all, curr_seg_output], axis=1)  # [h_qo, q_seg_size, d]

        expects.append(output_all)

    tmp = (
        torch.zeros((q_head_num, q_seq_length - cu_seqlen_q[-1], head_size))
        .to(dtype=expects[0].dtype)
        .to(expects[0].device)
    )
    expects.append(tmp)

    expect = torch.cat(expects, axis=1)
    return topk_page_indices_debug, expect.permute(1, 0, 2).reshape(q_seq_length, q_head_num * head_size)  # .bfloat16()


def mojo_block_quest(
    qkv,  # [24,1280]
    key_cache,  # [1024,1,256,128]
    value_cache,  # [1024,1,256,128]
    q_head_num,  # 8
    kv_head_num,  # 1
    kv_idx,  # [0,]
    kv_len,  # [0,]
    q_len_list,  # [17,0,..]
    global_rank,
    block_idx,
    q_seg_size,  # 1024
    topk_ratio,
    RECENT_WINDOW,
    sparse_limit,
):
    topk_page_indices_debug = None
    prefill_sparse = True
    # sparse_limit = 1024
    # page_size = 64
    page_size = key_cache.shape[2]
    # q_seg_size = 512

    # cache_size = key_cache.shape[0] # 1024
    q_seq_length = qkv.shape[0]
    # kv_seq_length = key_cache.shape[2]
    head_size = key_cache.shape[-1]
    max_kv_seq_length = max(q_len_list + kv_len)
    kv_cache_indices = []
    for kv_idx_i in kv_idx:
        kv_cache_indices.append(torch.nn.functional.pad(kv_idx_i, (0, max_kv_seq_length - kv_idx_i.shape[0]), value=-1))
    kv_cache_indices = torch.stack(kv_cache_indices)
    kv_seq_lengths = kv_len
    q_chunk_sizes = q_len_list

    query = (
        qkv.reshape(1, q_seq_length, (q_head_num + 2 * kv_head_num), head_size)
        .permute(0, 2, 1, 3)
        .contiguous()[:, :q_head_num, :, :]
        .contiguous()
    )  # [1,8,24,128]

    max_seq_lengths = 16384
    whole_causal = torch.tril(
        torch.ones((max_seq_lengths, max_seq_lengths), dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # mojo_quest_op(query, key_cache, value_cache, torch.tensor(q_len_list), kv_cache_indices, kv_seq_lengths)
    query_lengths = torch.tensor(q_len_list, device=query.device)
    q_chunk_sizes = query_lengths.tolist()
    cu_seqlen_q = torch.cumsum(query_lengths, dim=0)
    cu_seqlen_q = torch.nn.functional.pad(cu_seqlen_q, (1, 0), value=0)

    kv_cache_lengths = torch.tensor(kv_seq_lengths, device=query_lengths.device, dtype=query_lengths.dtype)
    kv_lengths = query_lengths + kv_cache_lengths
    cu_seqlen_kv = torch.cumsum(kv_lengths, dim=0)
    cu_seqlen_kv = torch.nn.functional.pad(cu_seqlen_kv, (1, 0), value=0)

    bsz, q_head_num, q_seq_length, head_size = query.shape
    assert bsz == 1
    query = query.squeeze(0)
    kv_head_num = key_cache.shape[1]
    # cu_seqlen_q = cu_seqlen_q.tolist()

    # [num_pages, num_kv_heads, head_dim]
    page_k_mins = key_cache.min(dim=2).values
    page_k_maxs = key_cache.max(dim=2).values

    valid_kv_seq_lengths = (
        torch.tensor(kv_seq_lengths, device=query_lengths.device, dtype=query_lengths.dtype) + query_lengths
    )
    num_topk_tokens = (valid_kv_seq_lengths * topk_ratio).int()
    num_topk_pages = num_topk_tokens // page_size

    from mojo_opset.core import MojoPagedPrefillBlockQuest

    block_quest = MojoPagedPrefillBlockQuest(q_seg_size, page_size)
    # [num_q_heads, ~= q_len / seg_size * ~= topk_ratio * num_pages]
    topk_page_idxs, q_chunk_idx, num_sparse_pages, cu_num_topk_pages_per_seg = block_quest(
        query,
        cu_seqlen_q,
        page_k_mins,
        page_k_maxs,
        kv_cache_indices,
        cu_seqlen_kv,
        num_topk_pages,
        RECENT_WINDOW,
    )
    topk_page_indices_debug = topk_page_idxs

    from mojo_opset.core import MojoPagedPrefillBlockSparseAttention

    block_sparse_attention = MojoPagedPrefillBlockSparseAttention(
        whole_causal, page_size, q_seg_size, topk_ratio, head_size, q_head_num, kv_head_num
    )
    expects = block_sparse_attention(
        query,
        key_cache,
        value_cache,
        cu_seqlen_q,
        cu_seqlen_kv,
        None,
        kv_cache_indices,
        q_chunk_idx,
        num_sparse_pages,
        topk_page_idxs,
        cu_num_topk_pages_per_seg,
    )

    return topk_page_indices_debug, expects.permute(1, 0, 2).reshape(
        q_seq_length, q_head_num * head_size
    )  # .bfloat16()


if __name__ == "__main__":
    test_paged_prefill_quest()
