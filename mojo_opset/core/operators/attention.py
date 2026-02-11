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
        gqa_layout: str = "AABB",
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
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Paged decode attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query of shape (B, Hq, D).
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
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
        _, num_kv_heads, block_size, head_dim = key_cache.shape

        num_share_q_heads = num_q_heads // num_kv_heads
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        outputs = torch.zeros(batch_size, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        for i in range(batch_size):
            seq_len = seqlens[i].item()

            q = query[i]

            k_ref = torch.zeros(
                seq_len, num_kv_heads, head_dim, device=query.device, dtype=query.dtype
            )
            v_ref = torch.zeros(
                seq_len, num_kv_heads, head_dim, device=query.device, dtype=query.dtype
            )
            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()

                start_pos = j * block_size
                tokens_in_block = min(block_size, seq_len - start_pos)

                k_slice = key_cache[physical_block_id, :, :tokens_in_block, :]
                v_slice = value_cache[physical_block_id, :, :tokens_in_block, :]

                k_ref[start_pos : start_pos + tokens_in_block, :, :] = k_slice.permute(1, 0, 2)
                v_ref[start_pos : start_pos + tokens_in_block, :, :] = v_slice.permute(1, 0, 2)

            if num_share_q_heads > 1:
                if self.gqa_layout == "AABB":
                    k_ref = k_ref.repeat_interleave(num_share_q_heads, dim=1)
                    v_ref = v_ref.repeat_interleave(num_share_q_heads, dim=1)
                else:
                    k_ref = k_ref.repeat((1, num_share_q_heads, 1))
                    v_ref = v_ref.repeat((1, num_share_q_heads, 1))

            attn_scores = torch.einsum("hd,khd->hk", q, k_ref) * softmax_scale
            # Note: if is_causal=True, we just do full attention over 1 query to seq_len key/value
            if not self.is_causal and mask is not None:
                if mask.dim() == 2:
                    attn_mask = mask
                else:
                    attn_mask = mask[i]
                attn_mask = attn_mask[seq_len, :seq_len]
                attn_scores.masked_fill_(attn_mask.unsqueeze(0), -torch.inf)

            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            outputs[i] = torch.einsum("hk,khd->hd", attn_probs, v_ref)
        return outputs


class MojoPrefillGQA(MojoOperator):
    pass


class MojoPagedPrefillGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
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
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        seqlens_kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Any]:
        """
        Paged prefill attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query tokens of shape (T, Hq, D).
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            cu_seqlens_q (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_seqlens_q[i]` is the start offset for query at batch i; `cu_seqlens_q[-1] == T`.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).
            seqlens_kv (Optional[torch.Tensor]): key/value lengths, shape (B,);
                `seqlens_kv[i]` is the length for key/value in key/value cache at batch i.
                If None, defaults to `cu_seqlens_q[i+1] - cu_seqlens_q[i]` for each batch i.
            mask (Optional[torch.Tensor]): Attention mask; defaults to None.
                If mask is None, it means a full mask or causal mask based on `is_causal`.
                If mask is not None, and is_causal=False, applies the mask to the attention scores.
                Currently we do not constrain the shape of mask, it is recommended be of shape (B, T, T) or (T, T),
                where B is the block size, and T >= max(max(seqlens_kv), max(seqlens_q)).

        Returns:
            torch.Tensor: Attention output of shape (T, Hq, D).

        Notes:
            - If Hq != Hkv, expands K/V heads to match Hq via repeat_interleave.
            - Applies a causal lower-triangular mask and restricts attention within each sequence.
            - Softmax is computed in float32 and cast back to the input dtype.
            - Despite the type annotation Tuple[Any], this implementation returns a single tensor.
        """
        total_q_tokens, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = key_cache.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        outputs = torch.zeros(total_q_tokens, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        batch_size = len(q_lens)

        for i in range(batch_size):
            q_seq_len = q_lens[i].item()
            start_loc = cu_seqlens_q[i].item()
            end_loc = cu_seqlens_q[i + 1].item()
            q = query[start_loc:end_loc]
            if seqlens_kv is None:
                kv_seq_len = q_seq_len
            else:
                kv_seq_len = seqlens_kv[i].item()
        

            num_blocks_for_seq = (kv_seq_len + block_size - 1) // block_size
            k_unpadded = torch.zeros(kv_seq_len, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)
            v_unpadded = torch.zeros(kv_seq_len, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()

                start_pos_in_seq = j * block_size
                end_pos_in_seq = min(start_pos_in_seq + block_size, kv_seq_len)
                tokens_in_block = end_pos_in_seq - start_pos_in_seq

                k_slice = key_cache[physical_block_id, :, :tokens_in_block, :]

                k_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = k_slice.permute(1, 0, 2)

                v_slice = value_cache[physical_block_id, :, :tokens_in_block, :]
                v_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = v_slice.permute(1, 0, 2)

            if num_q_heads != num_kv_heads:
                if self.gqa_layout == "AABB":
                    k_expanded = k_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                    v_expanded = v_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                else:
                    k_expanded = k_unpadded.repeat((1, num_q_heads // num_kv_heads, 1))
                    v_expanded = v_unpadded.repeat((1, num_q_heads // num_kv_heads, 1))
            else:
                k_expanded = k_unpadded
                v_expanded = v_unpadded

            attn_scores = torch.einsum("thd,khd->thk", q, k_expanded).float() * softmax_scale
            if self.is_causal:
                attn_mask = torch.ones(q_seq_len, kv_seq_len, device=query.device, dtype=torch.bool).tril(kv_seq_len - q_seq_len)
                attn_scores.masked_fill_(~attn_mask.unsqueeze(1), -torch.inf)
            elif mask is not None:
                if mask.dim() == 2:
                    attn_mask = mask
                else:
                    attn_mask = mask[i]
                attn_mask = attn_mask[kv_seq_len - q_seq_len:kv_seq_len, :kv_seq_len]
                attn_scores.masked_fill_(~attn_mask.unsqueeze(1), -torch.inf)

            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            outputs[start_loc:end_loc] = torch.einsum("thk,khd->thd", attn_probs, v_expanded)
        return outputs


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


class MojoPagedPrefillAttention(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        window_size: int = -1,
    ):
        """
        Initialize the Paged Prefill Attention operator.

        Args:
            is_causal (bool, default=True): Enable causal masking (lower-triangular) if True.
            window_size (int, default=-1): Attention window length. 
                Note: Currently only supports full attention (window_size=-1).

        Raises:
            ValueError: If `window_size` is not -1.
        """
        super().__init__()

        if window_size != -1:
            # 保留接口但不实现功能，仅支持全注意力
            pass

        self.is_causal = is_causal
        self.window_size = window_size

    def forward(
        self,
        query: torch.Tensor,
        key_query: torch.Tensor,
        value_query: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Paged prefill attention with separate query and KV sequence lengths.

        Args:
            query (torch.Tensor): Query tokens of shape (T_q, Hq, D).
            key_query (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_query (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            cu_seqlens_q (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_seqlens_q[i]` is the start offset for batch i; `cu_seqlens_q[-1] == T_q`.
            cu_seqlens_kv (torch.Tensor): Cumulative KV lengths, shape (B+1,);
                `cu_seqlens_kv[i]` is the start offset for batch i; `cu_seqlens_kv[-1] == T_kv`.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (T_q, Hq, D).

        Notes:
            - Supports different sequence lengths for queries and KV (e.g., for cross-attention).
            - Applies causal masking within each sequence if is_causal=True.
            - window_size parameter is currently ignored (only full attention is supported).
        """
        total_q_tokens, num_q_heads, head_dim = query.shape
        num_total_blocks, num_kv_heads, block_size, _ = key_query.shape
        
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        
        # Calculate batch sizes from cumulative sequence lengths
        batch_size = len(cu_seqlens_q) - 1
        if len(cu_seqlens_kv) - 1 != batch_size:
            raise ValueError(f"Batch size mismatch: cu_seqlens_q has {batch_size} batches, "
                           f"cu_seqlens_kv has {len(cu_seqlens_kv)-1} batches")
        
        # Prepare unpadded KV tensors
        total_kv_tokens = cu_seqlens_kv[-1].item()
        k_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, 
                               dtype=query.dtype, device=query.device)
        v_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim,
                               dtype=query.dtype, device=query.device)
        
        # Fill KV cache from blocks for each batch
        for i in range(batch_size):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            kv_start = cu_seqlens_kv[i].item()
            kv_end = cu_seqlens_kv[i + 1].item()
            
            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            
            # Get block table for this batch
            batch_block_table = block_tables[i]
            
            # Calculate number of blocks needed for this KV sequence
            num_blocks_for_seq = (kv_len + block_size - 1) // block_size
            
            # Extract KV from blocks
            for j in range(num_blocks_for_seq):
                if j >= batch_block_table.shape[0]:
                    break
                    
                physical_block_id = batch_block_table[j].item()
                
                start_pos_in_seq = j * block_size
                tokens_in_block = min(block_size, kv_len - start_pos_in_seq)
                
                if tokens_in_block <= 0:
                    continue
                
                # Get KV slices from the cache block
                k_slice = key_query[physical_block_id, :, :tokens_in_block, :]
                v_slice = value_query[physical_block_id, :, :tokens_in_block, :]
                
                # Permute to (tokens, heads, dim) and copy to unpadded tensors
                start_loc_in_kv = kv_start + start_pos_in_seq
                end_loc_in_kv = start_loc_in_kv + tokens_in_block
                
                k_unpadded[start_loc_in_kv:end_loc_in_kv, :, :] = k_slice.permute(1, 0, 2)
                v_unpadded[start_loc_in_kv:end_loc_in_kv, :, :] = v_slice.permute(1, 0, 2)
        
        # Expand KV heads to match query heads if needed
        if num_q_heads != num_kv_heads:
            if num_q_heads % num_kv_heads != 0:
                raise ValueError(f"num_q_heads ({num_q_heads}) must be a multiple of "
                               f"num_kv_heads ({num_kv_heads})")
            expand_factor = num_q_heads // num_kv_heads
            k_expanded = k_unpadded.repeat_interleave(expand_factor, dim=1)
            v_expanded = v_unpadded.repeat_interleave(expand_factor, dim=1)
        else:
            k_expanded = k_unpadded
            v_expanded = v_unpadded
        
        # Prepare output tensor
        output = torch.zeros_like(query)
        
        # Process each batch separately
        for i in range(batch_size):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            kv_start = cu_seqlens_kv[i].item()
            kv_end = cu_seqlens_kv[i + 1].item()
            
            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            
            if q_len == 0 or kv_len == 0:
                continue
            
            # Get query and KV slices for this batch
            query_i = query[q_start:q_end]
            k_i = k_expanded[kv_start:kv_end]
            v_i = v_expanded[kv_start:kv_end]
            
            # Compute attention scores
            # query_i: (q_len, num_q_heads, head_dim)
            # k_i: (kv_len, num_q_heads, head_dim)
            # attn_scores: (q_len, num_q_heads, kv_len)
            attn_scores = torch.einsum("thd,khd->thk", query_i.to(torch.float32), k_i.to(torch.float32)) * softmax_scale
            
            # Apply attention mask
            # 注意：对应rightDownCausal模式
            if self.is_causal:
                # Create causal mask
                # In prefill, queries can only attend to keys up to their position
                # For different query and KV lengths, we need to handle offset
                
                # If kv_len >= q_len, it means we have cached KV + current KV
                # The first (kv_len - q_len) tokens are from cache
                # The remaining q_len tokens correspond to current query positions
                
                cache_len = kv_len - q_len
                
                if cache_len >= 0:
                    # Create causal mask for this batch
                    # Query position t can attend to KV positions 0 to (cache_len + t)
                    mask = torch.zeros(q_len, kv_len, dtype=torch.bool, device=query.device)
                    for t in range(q_len):
                        mask[t, :(cache_len + t + 1)] = True
                else:
                    # If kv_len < q_len, we're in a cross-attention scenario
                    # Queries can attend to all KV tokens
                    mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device)
                    for t in range(kv_len):
                        mask[:(-cache_len + t + 1), t] = False
                # 注意：这里不处理window_size参数，仅支持全注意力
                # Apply mask to attention scores
                attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            
            # Compute attention probabilities
            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
            
            # Compute output for this batch
            # attn_probs: (q_len, num_q_heads, kv_len)
            # v_i: (kv_len, num_q_heads, head_dim)
            # output_i: (q_len, num_q_heads, head_dim)
            output_i = torch.einsum("thk,khd->thd", attn_probs, v_i.to(torch.float32))
            
            # Store output
            output[q_start:q_end] = output_i.to(query.dtype)
        
        return output
    

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
