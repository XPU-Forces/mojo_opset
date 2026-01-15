import math

from typing import Any
from typing import Optional
from typing import Tuple

import math
import torch

from .. import VALID_KV_LAYOUTS
from ..operator import MojoOperator


class MojoDecodeGQA(MojoOperator):
    """
    Paged GQA attention operator.
    Args:
        is_causal (bool): Whether to apply causal masking.
        is_prefill (bool): Whether running in prefill mode.
        page_size (int): Page size for attention computation.
        softmax_scale (float): Scaling factor for the softmax operation.
        gqa_layout (str): Layout for GQA attention.
        window_size (int): Window size for attention computation, -1 means full attention.
        op_name (str): Name of the operator.
    """

    def __init__(self, is_causal, is_prefill, page_size, softmax_scale, gqa_layout, window_size, op_name):
        super().__init__(op_name)
        self.is_causal = is_causal
        self.is_prefill = is_prefill
        self.page_size = page_size


class MojoPagedDecodeGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        q_scale_factor: int = 1,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        kv_layout: str = VALID_KV_LAYOUTS[0],
        tp_size: int = 1,
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Initialize the Paged Decode GQA attention operator with common parameters.
        Parameter descriptions:
        - q_scale_factor (int): Multiplier for q heads (integer, default 1), no scaling applied to q.
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - window_size (int): Attention window length; -1 means full window, or >=1 means sliding window length, default -1.
        - softmax_scale (Optional[float]): Scaling factor for attention scores, must be >0; default None.
        - kv_layout (str): KV storage layout indicator, values defined by VALID_KV_LAYOUTS, default VALID_KV_LAYOUTS[0].
        - tp_size (int): Tensor parallel size, default 1.
        - is_varlen (bool): When True, use TND (variable length) priority path; when False, use BSND; default True.
        - op_name (str): Operator name placeholder for registration and diagnostics.
        """
        super().__init__(op_name, layer_idx)

        if not isinstance(q_scale_factor, int) or q_scale_factor <= 0:
            raise ValueError(f"q_scale_factor must be a positive integer, got {q_scale_factor}")

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        if not isinstance(window_size, int) or (window_size != -1 and window_size < 1):
            raise ValueError(f"window_size must be -1 or >= 1, got {window_size}")

        if kv_layout not in VALID_KV_LAYOUTS:
            raise ValueError(f"kv_layout must be one of {VALID_KV_LAYOUTS}, got {kv_layout}")

        if not isinstance(tp_size, int) or tp_size <= 0:
            raise ValueError(f"tp_size must be a positive integer, got {tp_size}")

        if not isinstance(is_varlen, bool):
            raise ValueError(f"is_varlen must be a boolean, got {is_varlen}")

        self.is_causal = is_causal
        self.q_scale_factor = q_scale_factor
        self.gqa_layout = gqa_layout
        self.window_size = window_size
        self.kv_layout = kv_layout
        self.tp_size = tp_size
        self.is_varlen = is_varlen

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ):
        """
        Paged decode attention with grouped q heads (GQA) using a blocked KV cache.

        Args:
            q (torch.Tensor): Query of shape (B, Hq, D).
            k_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            v_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            cu_seqlens_q (torch.Tensor): Cumulative q lengths (unused here; see Notes).
            block_tables (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - If Hq > Hkv, K/V heads are repeated to match q heads.
            - Causal mask uses per-batch sequence lengths `seqlens`.
            - Softmax is computed in float32 and cast back to the input dtype.
            - This implementation references variables `q` and `seqlens`; ensure they
              correspond to `q` and the sequence-lengths tensor in the caller.
        """
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


class MojoPrefillGQA(MojoOperator):
    pass


class MojoPagedPrefillGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        q_scale_factor: int = 1,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        kv_layout: str = VALID_KV_LAYOUTS[0],
        tp_size: int = 1,
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Initialize the Paged Prefill GQA attention operator with common parameters.
        Parameter descriptions:
        - q_scale_factor (int): Multiplier for q heads (integer, default 1), no scaling applied to q.
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - window_size (int): Attention window length; -1 means full window, or >=1 means sliding window length, default -1.
        - kv_layout (str): KV storage layout indicator, values defined by VALID_KV_LAYOUTS, default VALID_KV_LAYOUTS[0].
        - tp_size (int): Tensor parallel size, default 1.
        - is_varlen (bool): When True, use TND (variable length) priority path; when False, use BSND; default True.
        - op_name (str): Operator name placeholder for registration and diagnostics.
        """
        super().__init__(op_name, layer_idx)

        if not isinstance(q_scale_factor, int) or q_scale_factor <= 0:
            raise ValueError(f"q_scale_factor must be a positive integer, got {q_scale_factor}")

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        if not isinstance(window_size, int) or (window_size != -1 and window_size < 1):
            raise ValueError(f"window_size must be -1 or >= 1, got {window_size}")

        if kv_layout not in VALID_KV_LAYOUTS:
            raise ValueError(f"kv_layout must be one of {VALID_KV_LAYOUTS}, got {kv_layout}")

        if not isinstance(tp_size, int) or tp_size <= 0:
            raise ValueError(f"tp_size must be a positive integer, got {tp_size}")

        if not isinstance(is_varlen, bool):
            raise ValueError(f"is_varlen must be a boolean, got {is_varlen}")

        self.is_causal = is_causal
        self.q_scale_factor = q_scale_factor
        self.gqa_layout = gqa_layout
        self.window_size = window_size
        self.kv_layout = kv_layout
        self.tp_size = tp_size
        self.is_varlen = is_varlen

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[Any]:
        """
        Paged prefill attention with grouped q heads (GQA) using a blocked KV cache.

        Args:
            q (torch.Tensor): Query tokens of shape (T, Hq, D).
            k_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            v_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            cu_seqlens_q (torch.Tensor): Cumulative q lengths, shape (B+1,);
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
        total_q_tokens, num_q_heads, head_dim = q.shape
        num_total_blocks, num_kv_heads, block_size, _ = k_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        total_kv_tokens = total_q_tokens

        k_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
        v_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)

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

        attn_mask = torch.ones(total_q_tokens, total_q_tokens, device=q.device, dtype=torch.bool).tril(diagonal=0)

        tok_to_seq = torch.repeat_interleave(torch.arange(batch_size, device=q.device), q_lens)

        seq_mask = tok_to_seq[:, None] == tok_to_seq[None, :]
        final_mask = attn_mask & seq_mask

        attn_scores = torch.einsum("thd,khd->thk", q, k_expanded) * softmax_scale
        attn_scores.masked_fill_(~final_mask.unsqueeze(1), -torch.inf)

        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        output = torch.einsum("thk,khd->thd", attn_probs, v_expanded)
        return output


class MojoBlockQuest(MojoOperator):
    """
    Paged Quest indexing operator for LLM Prefill.
    """

    def __init__(
        self,
        block_q: int,
        block_kv: int,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.block_q = block_q
        self.block_kv = block_kv

    def forward(self, curr_query_seg, mins, maxs, top_k_page):
        curr_query_seg = curr_query_seg[:, :: self.block_q]

        # [num_heads, num_segs, 1, head_size] * [num_heads, 1, num_pages, head_size]
        # --> [num_heads, num_segs, num_pages, head_size]
        q_min_k = curr_query_seg.unsqueeze(-2) * mins.unsqueeze(-3)
        q_max_k = curr_query_seg.unsqueeze(-2) * maxs.unsqueeze(-3)

        # [num_heads, num_segs, num_pages]
        page_score = torch.maximum(q_min_k, q_max_k).sum(dim=-1)
        # [num_heads, num_segs, top_k_page]
        _, topk_page_indices = page_score.topk(top_k_page, dim=-1)

        return topk_page_indices


class MojoPagedPrefillBlockSparseAttention(MojoOperator):
    """
    Paged Block Sparse Attention operator for LLM Prefill.
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
        full_mask = torch.ones(self.q_seg_size, self.page_size, device=causal_mask.device, dtype=torch.bool)
        empty_mask = torch.zeros(self.q_seg_size, self.page_size, device=causal_mask.device, dtype=torch.bool)
        session_mask = torch.ones(
            self.q_seg_size, self.q_seg_size * 3, device=causal_mask.device, dtype=torch.bool
        ).tril(diagonal=self.q_seg_size)
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


class MojoDecodeMLA(MojoOperator):
    pass


class MojoPagedDecodeMLA(MojoOperator):
    pass


class MojoDecodeNSA(MojoOperator):
    """
    MojoDecodeNSA operator.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoPagedDecodeNSA(MojoOperator):
    """
    Paged MLA attention operator for LLM Decode.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoPrefillMLA(MojoOperator):
    pass


class MojoPagedPrefillMLA(MojoOperator):
    pass


class MojoPrefillNSA(MojoOperator):
    """
    MLA attention operator for LLM Prefill.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoPagedPrefillNSA(MojoOperator):
    """
    Paged MLA attention operator for LLM Prefill.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoSdpa(MojoOperator):
    def __init__(
        self,
        mask: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        enable_gqa: bool = False,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
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
