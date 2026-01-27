import math

from typing import Any
from typing import Optional
from typing import Tuple

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


class MojoDecodeMLA(MojoOperator):
    pass


class MojoPagedDecodeMLA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        window_size: int = -1,
    ):
        super().__init__()

        if not isinstance(window_size, int) or (window_size != -1 and window_size < 1):
            raise ValueError(f"window_size must be -1 or >= 1, got {window_size}")

        self.is_causal = is_causal
        self.window_size = window_size

    def forward(
        self,
        query: torch.Tensor,
        key_query: torch.Tensor,
        value_query: torch.Tensor,
        current_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        kv_up_proj_weight: torch.Tensor | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        v_head_dim: int | None = None,
    ):
        """
        Paged decode MLA attention operator.

        Reconstructs per-token K/V from blocked caches and computes attention for
        all tokens (flattened across batch), returning a tensor of shape
        (B, Hq, D).
        """
        if kv_up_proj_weight is None or kv_lora_rank is None or qk_nope_head_dim is None or v_head_dim is None:
            raise ValueError("MojoPagedDecodeMLA requires kv_up_proj_weight, kv_lora_rank, qk_nope_head_dim, v_head_dim for MLA decode mode")

        batch_size, num_q_heads, qk_head_dim = query.shape
        num_total_blocks, block_size, kv_rank = key_query.shape

        if kv_rank != kv_lora_rank:
            raise ValueError(
                f"key_query last dim (kv_rank={kv_rank}) != kv_lora_rank={kv_lora_rank}"
            )

        max_len_in_batch = current_seq_lens.max().item()

        # Allocate buffers for the compressed KV and positional (PE) parts.
        kv_comp = torch.zeros(
            batch_size, max_len_in_batch, kv_lora_rank, device=query.device, dtype=key_query.dtype
        )
        pe_comp = torch.zeros(
            batch_size, max_len_in_batch, value_query.shape[-1], device=query.device, dtype=value_query.dtype
        )

        # Fill per-batch buffers by mapping logical -> physical blocks
        for i in range(batch_size):
            seq_len = current_seq_lens[i].item()
            num_blocks_for_seq = (seq_len + block_size - 1) // block_size
            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()
                start_pos = j * block_size
                tokens_in_block = min(block_size, seq_len - start_pos)

                # `k_slice`: (tokens_in_block, kv_lora_rank)
                k_slice = key_query[physical_block_id, :tokens_in_block, :]
                # `p_slice`: (tokens_in_block, pe_dim)
                p_slice = value_query[physical_block_id, :tokens_in_block, :]

                kv_comp[i, start_pos : start_pos + tokens_in_block, :] = k_slice
                pe_comp[i, start_pos : start_pos + tokens_in_block, :] = p_slice

        # Split the query into non-positional (nope) and positional (pe/rotary) components.
        q_nope = query[..., :qk_nope_head_dim]
        q_pe = query[..., qk_nope_head_dim:]

        if q_nope.shape[-1] != qk_nope_head_dim:
            raise ValueError(
                f"q_nope dim {q_nope.shape[-1]} != qk_nope_head_dim {qk_nope_head_dim}"
            )
        if q_pe.shape[-1] != pe_comp.shape[-1]:
            raise ValueError(
                f"q_pe dim {q_pe.shape[-1]} != pe_comp dim {pe_comp.shape[-1]}"
            )

        # Reshape the up-projection weight into per-head blocks.
        num_heads = num_q_heads
        w = kv_up_proj_weight
        if w.ndim == 2 and w.shape[0] == num_heads * (qk_nope_head_dim + v_head_dim):
            wkv_b = w.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
        elif w.ndim == 3 and w.shape[0] == num_heads:
            wkv_b = w
        else:
            # Conservative reshape with informative error if it fails later.
            wkv_b = w.reshape(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)

        wkv_b = wkv_b.to(query.dtype).to(query.device)

        # Project `q_nope` into the kv_lora space on a per-head basis.
        q_nope_proj = torch.einsum("bhd,hdc->bhc", q_nope, wkv_b[:, :qk_nope_head_dim, :])

        # Compute attention scores as the sum of two contributions.
        # Resulting shape: (B, H, L)
        attn_scores = (
            torch.einsum("bhc,blc->bhl", q_nope_proj, kv_comp)
            + torch.einsum("bhr,blr->bhl", q_pe, pe_comp)
        ) * softmax_scale

        # Mask positions beyond each sequence length per batch and apply softmax for numerical stability.
        k_len = kv_comp.shape[1]
        mask = torch.arange(k_len, device=query.device)[None, :] >= current_seq_lens[:, None]
        attn_scores.masked_fill_(mask[:, None, :], -torch.inf)

        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

        # Weighted sum over compressed kv -> (B, H, kv_lora_rank)
        weighted = torch.einsum("bhl,blc->bhc", attn_probs, kv_comp)

        # Project `weighted` back to value space using the v slice of `wkv_b`.
        v_proj = wkv_b[:, qk_nope_head_dim : qk_nope_head_dim + v_head_dim, :]
        if v_proj.shape[2] != kv_lora_rank:
            raise ValueError(
                f"v_proj last dim {v_proj.shape[2]} != kv_lora_rank {kv_lora_rank}"
            )

        out = torch.einsum("bhc,hdc->bhd", weighted, v_proj)
        return out


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
