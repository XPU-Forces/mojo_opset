import math

from typing import Optional

import torch

from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoBlockQuest, MojoPagedPrefillBlockSparseAttention
from mojo_opset.core import MojoSdpa


class RefPagedPrefillGQA(MojoPagedPrefillGQA):
    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        total_q_tokens, num_q_heads, head_dim = query.shape
        num_total_blocks, num_kv_heads, block_size, _ = k_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        total_kv_tokens = total_q_tokens

        k_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)
        v_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)

        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        batch_size = len(q_lens)

        for i in range(batch_size):
            seq_len = q_lens[i].item()
            start_loc = cu_seqlens_q[i].item()
            end_loc = cu_seqlens_q[i + 1].item()

            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()

                start_pos_in_seq = j * block_size
                tokens_in_block = min(block_size, seq_len - start_pos_in_seq)

                start_loc_in_batch = start_loc + start_pos_in_seq
                end_loc_in_batch = start_loc_in_batch + tokens_in_block

                k_slice = k_cache[physical_block_id, :, :tokens_in_block, :]

                k_unpadded[start_loc_in_batch:end_loc_in_batch, :, :] = k_slice.permute(1, 0, 2)

                v_slice = v_cache[physical_block_id, :, :tokens_in_block, :]
                v_unpadded[start_loc_in_batch:end_loc_in_batch, :, :] = v_slice.permute(1, 0, 2)

        if num_q_heads != num_kv_heads:
            k_expanded = k_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
            v_expanded = v_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        else:
            k_expanded = k_unpadded
            v_expanded = v_unpadded

        attn_mask = torch.ones(total_q_tokens, total_q_tokens, device=query.device, dtype=torch.bool).tril(diagonal=0)

        tok_to_seq = torch.repeat_interleave(torch.arange(batch_size, device=query.device), q_lens)

        seq_mask = tok_to_seq[:, None] == tok_to_seq[None, :]
        final_mask = attn_mask & seq_mask

        attn_scores = torch.einsum("thd,khd->thk", query, k_expanded) * softmax_scale
        attn_scores.masked_fill_(~final_mask.unsqueeze(1), -torch.inf)

        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

        output = torch.einsum("thk,khd->thd", attn_probs, v_expanded)
        return output


class RefPagedDecodeGQA(MojoPagedDecodeGQA):
    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ):
        batch_size, num_q_heads, head_dim = q.shape
        num_kv_heads, block_size, head_dim = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
        max_len_in_batch = seqlens.max().item()

        k_ref = torch.zeros(batch_size, max_len_in_batch, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
        v_ref = torch.zeros(batch_size, max_len_in_batch, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)

        for i in range(batch_size):
            seq_len = seqlens[i].item()
            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()

                start_pos = j * block_size
                tokens_in_block = min(block_size, seq_len - start_pos)

                k_slice = k_cache[physical_block_id, :, :tokens_in_block, :]
                v_slice = v_cache[physical_block_id, :, :tokens_in_block, :]

                k_ref[i, start_pos : start_pos + tokens_in_block, :, :] = k_slice.permute(1, 0, 2)
                v_ref[i, start_pos : start_pos + tokens_in_block, :, :] = v_slice.permute(1, 0, 2)

        _, k_len, num_k_heads, _ = k_ref.shape
        num_share_q_heads = num_q_heads // num_k_heads
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if num_share_q_heads > 1:
            k_ref = k_ref.repeat_interleave(num_share_q_heads, dim=2)
            v_ref = v_ref.repeat_interleave(num_share_q_heads, dim=2)

        attn = torch.einsum("bhd,bkhd->bhk", q, k_ref) * softmax_scale

        mask = torch.arange(k_len, device=q.device)[None, :] >= seqlens[:, None]
        attn.masked_fill_(mask[:, None, :], -torch.inf)

        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.einsum("bhk,bkhd->bhd", attn, v_ref)
        return out


class RefBlockQuest(MojoBlockQuest):

    def forward(self, curr_query_seg, mins, maxs, top_k_page):
        assert curr_query_seg.shape[1] <= self.block_q
        curr_query_seg = curr_query_seg[:, 0]

        # [num_heads, 1, head_size] * [num_heads, num_pages, head_size]

        q_min_k = curr_query_seg.unsqueeze(-2) * mins
        q_max_k = curr_query_seg.unsqueeze(-2) * maxs

        # [num_heads, num_pages]
        page_score = torch.maximum(q_min_k, q_max_k).sum(dim=-1)
        # [nh, ql, top_k_page]
        _, topk_page_indices = page_score.topk(top_k_page, dim=-1)

        return topk_page_indices


class RefPagedPrefillBlockSparseAttention(MojoPagedPrefillBlockSparseAttention):

    def forward(
        self,
        curr_query_seg,
        key,
        value,
        whole_causal_mask,
        topk_page_indices,
        q_seg_id,
        q_chunk_size,
    ):
        num_pages = value.shape[1] // self.page_size
        pad_len = value.shape[1] - num_pages * self.page_size
        q_head_num, curr_seg_size, head_size = curr_query_seg.shape
        page_size = self.page_size
        top_k_page = topk_page_indices.shape[-1]

        # [nh, ql, topk_page, page_size]

        topk_token_indices = (topk_page_indices * page_size).unsqueeze(1).unsqueeze(-1).repeat(
            1, curr_seg_size, 1, page_size
        ) + torch.arange(page_size, device=topk_page_indices.device)
        topk_token_indices = topk_token_indices.reshape(q_head_num, curr_seg_size, top_k_page * page_size)

        pad_indices = num_pages * page_size + torch.arange(pad_len, device=topk_token_indices.device)
        # [nh, ql, pad_len]
        pad_indices = pad_indices.expand(q_head_num, curr_seg_size, -1)
        # [nh, ql, topk_page * page_size + pad_len]
        topk_token_indices = torch.cat([topk_token_indices, pad_indices], dim=-1)
        q_head_num, curr_seg_size, head_size = curr_query_seg.shape
        # [nh, q_seg_size, kv_seq_length]
        curr_seg_score = torch.bmm(curr_query_seg, key.transpose(-2, -1))
        curr_seg_score = curr_seg_score / math.sqrt(head_size)

        # [nh, q_seg_size, kv_seq_length]
        curr_seg_mask = torch.zeros_like(curr_seg_score, dtype=torch.bool)
        curr_seg_mask.scatter_(dim=-1, index=topk_token_indices, value=True)
        # curr_seg_causal = torch.tril(torch.ones((q_head_num, curr_seg_size, curr_seg_size), dtype= torch.bool, device=curr_seg_mask.device))
        # print(f"{curr_seg_mask.shape=} {q_seg_start=} {q_seg_end=}", flush=True)
        q_seg_start = q_seg_id * self.q_seg_size
        # Note!!!: the session_cache_part is always attentioned
        curr_seg_mask[:, :, -q_chunk_size:] = whole_causal_mask[
            q_seg_start : q_seg_start + curr_seg_size, -q_chunk_size:
        ]

        curr_seg_score = curr_seg_score.masked_fill(~curr_seg_mask, torch.finfo(curr_seg_score.dtype).min)
        curr_seg_score = torch.softmax(curr_seg_score, -1, dtype=torch.float32).to(dtype=torch.bfloat16)
        # [nh, q_seg_size, head_size]
        curr_seg_output = (
            torch.bmm(curr_seg_score, value)
            # .permute(1, 0, 2)
            # .reshape(curr_seg_size, q_head_num * head_size)
            .to(dtype=torch.bfloat16)
        )
        return curr_seg_output


class RefSdpa(MojoSdpa):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=self.mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
            enable_gqa=self.enable_gqa,
        )
        return output
