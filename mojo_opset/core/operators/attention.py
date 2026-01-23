import math

from typing import Any
from typing import Optional
from typing import Tuple

import math
import torch

from ..operator import MojoOperator


class MojoDecodeGQA(MojoOperator):
    pass


class MojoPagedDecodeGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
    ):
        """
        Initialize the Paged Decode GQA attention operator.

        Args:
            is_causal (bool, default=True): Enable causal masking (lower-triangular) if True.
            gqa_layout (str, default="ABAB"): GQA head grouping layout; one of {"ABAB", "AABB"}.
            window_size (int, default=-1): Attention window length. Use -1 for full context,
                or a positive integer (>= 1) to enable a sliding window of that length.

        Raises:
            ValueError: If `gqa_layout` is not in {"ABAB", "AABB"} or if `window_size` is neither
                -1 nor a positive integer (>= 1).

        Notes:
            This initializer stores configuration only. Actual causal masking and window enforcement
            are applied in the forward path according to these settings.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        if not isinstance(window_size, int) or (window_size != -1 and window_size < 1):
            raise ValueError(f"window_size must be -1 or >= 1, got {window_size}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.window_size = window_size

    def forward(
        self,
        query: torch.Tensor,
        key_query: torch.Tensor,
        value_query: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
    ):
        """
        Paged decode attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query of shape (B, Hq, D).
            key_query (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_query (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            cu_seqlens_q (torch.Tensor): Cumulative query lengths (unused here; see Notes).
            block_tables (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Causal mask uses per-batch sequence lengths `seqlens`.
            - Softmax is computed in float32 and cast back to the input dtype.
            - This implementation references variables `query` and `seqlens`; ensure they
              correspond to `query` and the sequence-lengths tensor in the caller.
        """
        assert not cu_seq_lens, "varlen is not supported"

        batch_size, num_q_heads, head_dim = query.shape
        num_kv_heads, block_size, head_dim = key_query.shape[1], key_query.shape[2], key_query.shape[3]
        max_len_in_batch = seqlens.max().item()

        k_ref = torch.zeros(
            batch_size, max_len_in_batch, num_kv_heads, head_dim, device=query.device, dtype=query.dtype
        )
        v_ref = torch.zeros(
            batch_size, max_len_in_batch, num_kv_heads, head_dim, device=query.device, dtype=query.dtype
        )

        for i in range(batch_size):
            seq_len = seqlens[i].item()
            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()

                start_pos = j * block_size
                tokens_in_block = min(block_size, seq_len - start_pos)

                k_slice = key_query[physical_block_id, :, :tokens_in_block, :]
                v_slice = value_query[physical_block_id, :, :tokens_in_block, :]

                k_ref[i, start_pos : start_pos + tokens_in_block, :, :] = k_slice.permute(1, 0, 2)
                v_ref[i, start_pos : start_pos + tokens_in_block, :, :] = v_slice.permute(1, 0, 2)

        _, k_len, num_k_heads, _ = k_ref.shape
        num_share_q_heads = num_q_heads // num_k_heads
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if num_share_q_heads > 1:
            k_ref = k_ref.repeat_interleave(num_share_q_heads, dim=2)
            v_ref = v_ref.repeat_interleave(num_share_q_heads, dim=2)

        attn = torch.einsum("bhd,bkhd->bhk", query, k_ref) * softmax_scale

        mask = torch.arange(k_len, device=query.device)[None, :] >= seqlens[:, None]
        attn.masked_fill_(mask[:, None, :], -torch.inf)

        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.einsum("bhk,bkhd->bhd", attn, v_ref)
        return out


class MojoPrefillGQA(MojoOperator):
    pass


class MojoPagedPrefillGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
    ):
        """
        Initialize the Paged Prefill GQA attention operator with common parameters.
        Parameter descriptions:
        - q_scale_factor (int): Multiplier for query heads (integer, default 1), no scaling applied to query.
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - window_size (int): Attention window length; -1 means full window, or >=1 means sliding window length, default -1.
        """
        super().__init__()

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        if not isinstance(window_size, int) or (window_size != -1 and window_size < 1):
            raise ValueError(f"window_size must be -1 or >= 1, got {window_size}")

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.window_size = window_size

    def forward(
        self,
        query: torch.Tensor,
        key_query: torch.Tensor,
        value_query: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[Any]:
        """
        Paged prefill attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query tokens of shape (T, Hq, D).
            key_query (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_query (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            cu_seqlens_q (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_seqlens_q[i]` is the start offset for batch i; `cu_seqlens_q[-1] == T`.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (T, Hq, D).

        Notes:
            - If Hq != Hkv, expands K/V heads to match Hq via repeat_interleave.
            - Applies a causal lower-triangular mask and restricts attention within each sequence.
            - Softmax is computed in float32 and cast back to the input dtype.
            - Despite the type annotation Tuple[Any], this implementation returns a single tensor.
        """
        total_q_tokens, num_q_heads, head_dim = query.shape
        num_total_blocks, num_kv_heads, block_size, _ = key_query.shape
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

                k_slice = key_query[physical_block_id, :, :tokens_in_block, :]

                k_unpadded[start_loc_in_batch:end_loc_in_batch, :, :] = k_slice.permute(1, 0, 2)

                v_slice = value_query[physical_block_id, :, :tokens_in_block, :]
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


class MojoBlockSparseAttention(MojoOperator):
    """
    Block Sparse Attention operator for LLM Prefill.
    """

    def __init__(
        self,
        causal_mask: Optional[torch.Tensor],
        page_size: int,
        q_seg_size: int,
        topk_ratio: float,
        head_size: int,
        q_head_num: int,
        kv_head_num: int,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.page_size = page_size
        self.q_seg_size = q_seg_size
        self.topk_ratio = topk_ratio
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.head_size = head_size
        mask_block_size = max(self.page_size, max(self.q_seg_size, 128))
        full_mask = torch.ones(mask_block_size, mask_block_size, device=causal_mask.device, dtype=torch.bool)
        empty_mask = torch.zeros(mask_block_size, mask_block_size, device=causal_mask.device, dtype=torch.bool)
        session_mask = torch.ones(
            mask_block_size, mask_block_size * 3, device=causal_mask.device, dtype=torch.bool
        ).tril(diagonal=mask_block_size)
        self.mask = torch.cat([full_mask, empty_mask, session_mask], dim=1)
        self.scale = 1 / math.sqrt(head_size)

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


class MojoPagedPrefillBlockSparseAttention(MojoOperator):
    """
    Block Sparse Attention operator for LLM Prefill with Paged KVCache.
    """

    def __init__(
        self,
        causal_mask: Optional[torch.Tensor],
        page_size: int,
        q_seg_size: int,
        topk_ratio: float,
        head_size: int,
        q_head_num: int,
        kv_head_num: int,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.page_size = page_size
        self.q_seg_size = q_seg_size
        self.topk_ratio = topk_ratio
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.head_size = head_size
        self.whole_mask = causal_mask
        mask_block_size = max(self.page_size, max(self.q_seg_size, 128))
        full_mask = torch.ones(mask_block_size, mask_block_size, device=causal_mask.device, dtype=torch.bool)
        empty_mask = torch.zeros(mask_block_size, mask_block_size, device=causal_mask.device, dtype=torch.bool)
        session_mask = torch.ones(
            mask_block_size, mask_block_size * 3, device=causal_mask.device, dtype=torch.bool
        ).tril(diagonal=mask_block_size)
        self.mask = torch.cat([full_mask, empty_mask, session_mask], dim=1)
        self.scale = 1 / math.sqrt(head_size)

    def forward(
        self,
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        whole_causal_mask,
        kv_cache_indices,
        q_chunk_idx,
        num_sparse_pages,
        topk_page_indices,
        cu_num_topk_pages_per_seg,
    ):
        expects = torch.zeros_like(query)

        curr_seq_id = -1

        for i in range(q_chunk_idx.shape[0]):
            q_idx = q_chunk_idx[i][0].item()
            seg_id = q_chunk_idx[i][1].item()
            if q_idx != curr_seq_id:
                curr_seq_id = q_idx

                q_chunk_size = cu_seqlens_q[q_idx + 1].item() - cu_seqlens_q[q_idx].item()
                kv_len = cu_seqlens_k[q_idx + 1].item() - cu_seqlens_k[q_idx].item()

                sublist = kv_cache_indices[q_idx]
                valid_mask = sublist != -1
                valid_indices = sublist[valid_mask]
                curr_query = query[:, cu_seqlens_q[q_idx] : cu_seqlens_q[q_idx + 1]]

                key_cache_i = (
                    key[valid_indices]
                    .reshape(-1, self.kv_head_num, self.head_size)
                    .permute(1, 0, 2)[:, :kv_len, :]
                    .repeat_interleave(self.q_head_num // self.kv_head_num, dim=0)
                    .contiguous()
                )
                value_cache_i = (
                    value[valid_indices]
                    .reshape(-1, self.kv_head_num, self.head_size)
                    .permute(1, 0, 2)[:, :kv_len, :]
                    .repeat_interleave(self.q_head_num // self.kv_head_num, dim=0)
                    .contiguous()
                )

            valid_num_pages = num_sparse_pages[i].item()
            pad_len = kv_len - valid_num_pages * self.page_size
            q_seg_start = seg_id * self.q_seg_size
            curr_seg_size = min(self.q_seg_size, q_chunk_size - q_seg_start)
            curr_query_seg = curr_query[:, q_seg_start : q_seg_start + curr_seg_size]
            topk_page_indices_seg = topk_page_indices[
                :, cu_num_topk_pages_per_seg[i] : cu_num_topk_pages_per_seg[i + 1]
            ]

            topk_token_indices_seg = (topk_page_indices_seg * self.page_size).unsqueeze(1).unsqueeze(-1).repeat(
                1, curr_seg_size, 1, self.page_size
            ) + torch.arange(self.page_size, device=topk_page_indices.device)

            topk_token_indices_seg = topk_token_indices_seg.reshape(self.q_head_num, curr_seg_size, -1)

            pad_indices = valid_num_pages * self.page_size + torch.arange(pad_len, device=topk_token_indices_seg.device)
            # [nh, ql, pad_len]
            pad_indices = pad_indices.expand(self.q_head_num, curr_seg_size, -1)
            # [nh, ql, topk_page * page_size + pad_len]
            topk_token_indices_seg = torch.cat([topk_token_indices_seg, pad_indices], dim=-1)

            curr_seg_score = torch.bmm(curr_query_seg.float(), key_cache_i.float().transpose(-2, -1))
            curr_seg_score = curr_seg_score * self.scale

            curr_seg_mask = torch.zeros_like(curr_seg_score, dtype=torch.bool)
            curr_seg_mask.scatter_(dim=-1, index=topk_token_indices_seg, value=True)
            curr_seg_mask[:, :, -q_chunk_size:] = self.whole_mask[
                q_seg_start : q_seg_start + curr_seg_size, :q_chunk_size
            ]

            curr_seg_score = curr_seg_score.masked_fill(~curr_seg_mask, float("-inf"))
            curr_seg_score = torch.softmax(curr_seg_score, dim=-1)
            curr_seg_output = torch.bmm(curr_seg_score, value_cache_i.float()).to(query.dtype)
            expects[:, cu_seqlens_q[q_idx] + q_seg_start : cu_seqlens_q[q_idx] + q_seg_start + curr_seg_size] = (
                curr_seg_output
            )

        return expects


class MojoDecodeMLA(MojoOperator):
    pass


class MojoPagedDecodeMLA(MojoOperator):
    pass


class MojoDecodeNSA(MojoOperator):
    pass


class MojoPagedDecodeNSA(MojoOperator):
    pass


class MojoPrefillMLA(MojoOperator):
    pass


class MojoPagedPrefillMLA(MojoOperator):
    pass


class MojoPrefillNSA(MojoOperator):
    pass


class MojoPagedPrefillNSA(MojoOperator):
    pass


class MojoSdpa(MojoOperator):
    def __init__(
        self,
        mask: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        enable_gqa: bool = False,
    ):
        super().__init__()
        self.mask = mask
        self.scale = scale
        self.enable_gqa = enable_gqa

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Scaled Dot-Product Attention (SDPA) operator.

        Args:
            query (torch.Tensor): Query tensor; shape must be compatible with SDPA.
            key (torch.Tensor): Key tensor; same embedding dimension as query.
            value (torch.Tensor): Value tensor; same embedding dimension as key.

        Returns:
            torch.Tensor: Attention output with the same batch/head layout as `query`.

        Notes:
            - Uses `attn_mask=self.mask` (provided externally) and disables dropout.
            - `scale=self.scale` sets custom scaling; if None, SDPA uses default scaling.
            - `enable_gqa=self.enable_gqa` allows grouped query attention when supported.
        """
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
