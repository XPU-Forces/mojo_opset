from typing import Optional

import torch

from ..operator import MojoOperator


class MojoLightningIndexer(MojoOperator):
    def forward(
        self,
        query: torch.Tensor,
        query_scale: torch.Tensor,
        key: torch.Tensor,
        key_scale: Optional[torch.Tensor] = None,
    ):
        """
        Lightning index calculation with query and optional key scaling.

        Args:
            query: Query tensor. Shape ``[B, M, H, K]``, where B is batch size,
                M is the sequence length of query, H is head number, K is head dimension.
            query_scale: Query scaling factors. Shape ``[B, M, H]``.
            key: Key tensor. Shape ``[B, N, K]``, where N is the sequence length of key.
            key_scale: Optional scaling factors for key. Shape can be ``[B, N]`` or ``[N]``.

        Returns:
            index_score: Index score tensor. Shape ``[B, M, N]``.
        """
        batch_size, q_seq_len, head_num, head_dim = query.shape
        k_seq_len = key.shape[1]

        assert query_scale.size() == (
            batch_size,
            q_seq_len,
            head_num,
        ), f"query_scale must be [B, M, H], got {query_scale.size()}"

        if key_scale is None:
            key_scale = torch.ones(
                (batch_size, k_seq_len),
                dtype=torch.float32,
                device=query.device,
            )
        else:
            key_scale_shape = key_scale.shape
            if len(key_scale_shape) == 1:
                assert key_scale_shape[0] == k_seq_len, (
                    f"key_scale [N] must have N={k_seq_len}, got {key_scale_shape[0]}"
                )
                key_scale = key_scale.to(torch.float32).unsqueeze(0).expand(batch_size, -1)
            elif len(key_scale_shape) == 2:
                assert key_scale_shape == (batch_size, k_seq_len), f"key_scale must be [B, N], got {key_scale_shape}"
            else:
                raise ValueError(f"Invalid key_scale shape {key_scale_shape}")

        index_score = torch.zeros(
            (batch_size, q_seq_len, k_seq_len),
            dtype=torch.float32,
            device=query.device,
        )

        for batch_id in range(batch_size):
            key_batch = key[batch_id].to(torch.float32)  # [N, K]
            key_scale_batch = key_scale[batch_id].unsqueeze(-1)  # [N, 1]
            key_scaled = key_batch * key_scale_batch  # [N, K]

            for i in range(q_seq_len):
                q_slice = query[batch_id, i].to(torch.float32)  # [H, K]
                dot_product = torch.matmul(q_slice, key_scaled.transpose(0, 1))  # [H, N]
                relu_out = torch.maximum(dot_product, torch.tensor(0.0))
                q_scale_slice = query_scale[batch_id, i].unsqueeze(-1)  # [H, 1]
                scaled_out = relu_out * q_scale_slice
                index_score[batch_id, i] = torch.sum(scaled_out, dim=0)

        return index_score


class MojoIndexerCompressEpilog(MojoOperator):
    def forward(
        self,
        attn: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ):
        """
        Indexer compress epilog operation.

        Args:
            attn: Attention output tensor.
            x: Input tensor.
            residual: Residual tensor.
            gamma: Gamma scaling tensor.

        Returns:
            Output tensor.
        """
        return attn + x * gamma.unsqueeze(-1) + residual


class MojoKvCompressEpilog(MojoOperator):
    def forward(
        self,
        attn: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        kv_s: torch.Tensor = None,
        kv_z: torch.Tensor = None,
    ):
        """
        KV compress epilog operation with optional quantization parameters.

        Args:
            attn: Attention output tensor.
            x: Input tensor.
            residual: Residual tensor.
            gamma: Gamma scaling tensor.
            kv_s: Optional scale parameters for KV quantization.
            kv_z: Optional zero point parameters for KV quantization.

        Returns:
            Output tensor.
        """
        # Dequantize if quantization parameters are provided
        if kv_s is not None and kv_z is not None:
            residual = residual.to(torch.float32) * kv_s.unsqueeze(-1) + kv_z.unsqueeze(-1)
        return attn + x * gamma.unsqueeze(-1) + residual


class MojoSparseAttnSharedkv(MojoOperator):
    def forward(
        self,
        q: torch.Tensor,
        *,
        ori_kv: torch.Tensor = None,
        cmp_kv: torch.Tensor = None,
        ori_sparse_indices: torch.Tensor = None,
        cmp_sparse_indices: torch.Tensor = None,
        ori_block_table: torch.Tensor = None,
        cmp_block_table: torch.Tensor = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_ori_kv: torch.Tensor = None,
        cu_seqlens_cmp_kv: torch.Tensor = None,
        seqused_q: torch.Tensor = None,
        seqused_kv: torch.Tensor = None,
        sinks: torch.Tensor = None,
        metadata: torch.Tensor = None,
        softmax_scale: float = 0,
        cmp_ratio: int = 0,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout_q: str = 'BSND',
        layout_kv: str = 'PA_ND',
        return_softmax_lse: bool = False,
    ):
        """
        Sparse attention with shared KV cache.

        Args:
            q: Query tensor. Shape depends on layout_q: 'BSND' -> [B, S1, N1, D], 'TND' -> [T1, N1, D].
            ori_kv: Original KV tensor. Shape: [ori_block_num, ori_block_size, KV_N, D].
            cmp_kv: Compressed KV tensor. Shape: [cmp_block_num, cmp_block_size, KV_N, D].
            ori_sparse_indices: Original sparse indices. Shape: [Q_T, KV_N, K1].
            cmp_sparse_indices: Compressed sparse indices. Shape: [Q_T, KV_N, K2].
            ori_block_table: Original block table for PageAttention. Shape: [B, num_blocks].
            cmp_block_table: Compressed block table for PageAttention. Shape: [B, num_blocks].
            cu_seqlens_q: Cumulative sequence lengths for query. Shape: [B+1].
            cu_seqlens_ori_kv: Cumulative sequence lengths for original KV.
            cu_seqlens_cmp_kv: Cumulative sequence lengths for compressed KV.
            seqused_q: Sequence used flags for query. Shape: [B].
            seqused_kv: Sequence used flags for KV. Shape: [B].
            sinks: Sinks tensor. Shape: [N1].
            metadata: Metadata tensor from npu_sparse_attn_sharedkv_metadata. Shape: [1024]. Currently required.
            softmax_scale: Softmax scaling factor. Default 0 means 1/sqrt(D).
            cmp_ratio: Compression ratio for ori_kv. Support 4/128. Default 0.
            ori_mask_mode: Mask mode for original attention. Only support 4 (band mode).
            cmp_mask_mode: Mask mode for compressed attention. Only support 3 (rightDownCausal).
            ori_win_left: Window left size for original attention. Only support 127.
            ori_win_right: Window right size for original attention. Only support 0.
            layout_q: Layout of query tensor. 'BSND' or 'TND'.
            layout_kv: Layout of KV tensor. Only 'PA_ND' supported.
            return_softmax_lse: Whether to return softmax lse. Currently not supported.

        Returns:
            attn_out: Attention output tensor.
            softmax_lse: Optional softmax lse tensor.
        """
        # Reference implementation
        if layout_q == 'BSND':
            batch_size, seq_len, num_heads, head_dim = q.shape
            attn_out = torch.zeros((batch_size, seq_len, num_heads, head_dim), dtype=q.dtype, device=q.device)
        else:  # TND
            seq_len, num_heads, head_dim = q.shape
            attn_out = torch.zeros((seq_len, num_heads, head_dim), dtype=q.dtype, device=q.device)
        
        if return_softmax_lse:
            softmax_lse = torch.zeros((num_heads, seq_len), dtype=torch.float32, device=q.device)
            return attn_out, softmax_lse
        return attn_out


class MojoSparseAttnSharedkvMetadata(MojoOperator):
    def forward(
        self,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        *,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_ori_kv: torch.Tensor = None,
        cu_seqlens_cmp_kv: torch.Tensor = None,
        seqused_q: torch.Tensor = None,
        seqused_kv: torch.Tensor = None,
        batch_size: int = 0,
        max_seqlen_q: int = 0,
        max_seqlen_kv: int = 0,
        cmp_topk: int = 0,
        cmp_ratio: int = 0,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout_q: str = 'BSND',
        layout_kv: str = 'PA_ND',
        has_ori_kv: bool = True,
        has_cmp_kv: bool = False,
        device: str = 'npu:0',
    ):
        """
        Metadata calculation for sparse attention with shared KV cache.

        Args:
            num_heads_q: Number of query heads.
            num_heads_kv: Number of KV heads.
            head_dim: Dimension of each head.
            cu_seqlens_q: Cumulative sequence lengths for query. Shape: [B+1].
            cu_seqlens_ori_kv: Cumulative sequence lengths for original KV.
            cu_seqlens_cmp_kv: Cumulative sequence lengths for compressed KV.
            seqused_q: Sequence used flags for query. Shape: [B].
            seqused_kv: Sequence used flags for KV. Shape: [B].
            batch_size: Batch size.
            max_seqlen_q: Maximum sequence length for query.
            max_seqlen_kv: Maximum sequence length for KV.
            cmp_topk: TopK for compressed attention.
            cmp_ratio: Compression ratio.
            ori_mask_mode: Mask mode for original attention. Only support 4.
            cmp_mask_mode: Mask mode for compressed attention. Only support 3.
            ori_win_left: Window left size for original attention.
            ori_win_right: Window right size for original attention.
            layout_q: Layout of query tensor. 'BSND' or 'TND'.
            layout_kv: Layout of KV tensor. Only 'PA_ND' supported.
            has_ori_kv: Whether original KV is present.
            has_cmp_kv: Whether compressed KV is present.
            device: Target device.

        Returns:
            metadata: Metadata tensor of shape [1024].
        """
        # Reference implementation - return dummy metadata
        return torch.ones(1024, dtype=torch.int32)


class MojoKvQuantSparseAttnSharedkv(MojoOperator):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_s: torch.Tensor,
        kv_z: torch.Tensor,
        scale: float,
        causal: bool = False,
    ):
        """
        KV quantized sparse attention with shared KV cache.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            kv_s: Scale parameters for KV quantization.
            kv_z: Zero point parameters for KV quantization.
            scale: Scaling factor.
            causal: Whether to apply causal masking.

        Returns:
            Output tensor.
        """
        # Dequantize KV
        key = key.to(torch.float32) * kv_s.unsqueeze(-1) + kv_z.unsqueeze(-1)
        value = value.to(torch.float32) * kv_s.unsqueeze(-1) + kv_z.unsqueeze(-1)

        # Standard attention calculation
        # query: [B, M, H, K] or [B, H, M, K]
        # key: [B, N, K] -> need to reshape to [B, H, N, K] or handle differently
        query_dtype = query.dtype
        query = query.to(torch.float32)
        if query.dim() == 4 and key.dim() == 3:
            # Expand key to match query heads
            key = key.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
            value = value.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones_like(scores), diagonal=1)
            scores = scores.masked_fill(mask == 1, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output


class MojoKvQuantSparseAttnSharedkvMetadata(MojoOperator):
    def forward(
        self,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        kv_quant_mode: int = 0,
        *,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_ori_kv: torch.Tensor = None,
        cu_seqlens_cmp_kv: torch.Tensor = None,
        seqused_q: torch.Tensor = None,
        seqused_kv: torch.Tensor = None,
        batch_size: int = 0,
        max_seqlen_q: int = 0,
        max_seqlen_kv: int = 0,
        ori_topk: int = 0,
        cmp_topk: int = 0,
        tile_size: int = 0,
        rope_head_dim: int = 0,
        cmp_ratio: int = -1,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout_q: str = "BSND",
        layout_kv: str = "PA_ND",
        has_ori_kv: bool = True,
        has_cmp_kv: bool = True,
    ):
        """
        KV quantized sparse attention with shared KV cache and metadata.

        Args:
            num_heads_q: Number of query heads.
            num_heads_kv: Number of KV heads.
            head_dim: Dimension of each head.
            kv_quant_mode: KV quantization mode.
            cu_seqlens_q: Cumulative sequence lengths for query.
            cu_seqlens_ori_kv: Cumulative sequence lengths for original KV.
            cu_seqlens_cmp_kv: Cumulative sequence lengths for compressed KV.
            seqused_q: Sequence used flags for query.
            seqused_kv: Sequence used flags for KV.
            batch_size: Batch size.
            max_seqlen_q: Maximum sequence length for query.
            max_seqlen_kv: Maximum sequence length for KV.
            ori_topk: TopK for original attention.
            cmp_topk: TopK for compressed attention.
            tile_size: Tile size for computation.
            rope_head_dim: RoPE head dimension.
            cmp_ratio: Compression ratio.
            ori_mask_mode: Mask mode for original attention.
            cmp_mask_mode: Mask mode for compressed attention.
            ori_win_left: Original window left size.
            ori_win_right: Original window right size.
            layout_q: Layout of query tensor.
            layout_kv: Layout of KV tensor.
            has_ori_kv: Whether has original KV.
            has_cmp_kv: Whether has compressed KV.

        Returns:
            metadata tensor.
        """
        # Reference implementation returns dummy metadata
        if cu_seqlens_q is not None:
            return torch.zeros((batch_size, max_seqlen_q, num_heads_q), dtype=torch.int32, device=cu_seqlens_q.device)
        return torch.zeros((batch_size, max_seqlen_q, num_heads_q), dtype=torch.int32)


class MojoQuantLightningIndexerMetadata(MojoOperator):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_dequant_scale: torch.Tensor,
        key_dequant_scale: torch.Tensor,
        query_quant_mode: int,
        key_quant_mode: int,
        metadata: torch.Tensor = None,
        *,
        actual_seq_lengths_query: Optional[torch.Tensor] = None,
        actual_seq_lengths_key: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        layout_query: str = "BSND",
        layout_key: str = "PA_BSND",
        sparse_count: int = 2048,
        sparse_mode: int = 3,
        pre_tokens: int = 9223372036854775807,
        next_tokens: int = 9223372036854775807,
        cmp_ratio: int = 1,
        return_value: bool = False,
    ):
        """
        Quantized lightning index calculation with metadata support.

        Args:
            query: Query tensor. Shape ``[B, M, H, K]``.
            key: Key tensor. Shape ``[B, N, K]``.
            weights: Value tensor or weights tensor.
            query_dequant_scale: Query dequantization scale.
            key_dequant_scale: Key dequantization scale.
            query_quant_mode: Query quantization mode (0: no quant, 1: per-tensor, 2: per-channel).
            key_quant_mode: Key quantization mode (0: no quant, 1: per-tensor, 2: per-channel).
            metadata: Optional metadata tensor for advanced processing.
            actual_seq_lengths_query: Actual sequence lengths for query.
            actual_seq_lengths_key: Actual sequence lengths for key.
            block_table: Block table for sparse attention.
            layout_query: Layout of query tensor.
            layout_key: Layout of key tensor.
            sparse_count: Sparse count parameter.
            sparse_mode: Sparse mode parameter.
            pre_tokens: Pre-tokens parameter.
            next_tokens: Next tokens parameter.
            cmp_ratio: Comparison ratio.
            return_value: Whether to return value.

        Returns:
            index_score: Index score tensor. Shape ``[B, M, N]``.
        """
        batch_size, q_seq_len, head_num, head_dim = query.shape
        k_seq_len = key.shape[1]

        # Dequantize query if needed
        if query_quant_mode != 0:
            query = query.to(torch.float32) * query_dequant_scale.unsqueeze(-1).unsqueeze(-1)

        # Dequantize key if needed
        if key_quant_mode != 0:
            key = key.to(torch.float32) * key_dequant_scale.unsqueeze(-1)

        index_score = torch.zeros(
            (batch_size, q_seq_len, k_seq_len),
            dtype=torch.float32,
            device=query.device,
        )

        index_score = torch.zeros(
            (batch_size, q_seq_len, k_seq_len),
            dtype=torch.float32,
            device=query.device,
        )

        for batch_id in range(batch_size):
            key_batch = key[batch_id].to(torch.float32)
            key_scale_batch = key_dequant_scale[batch_id].unsqueeze(-1) if key_quant_mode != 0 else torch.ones(k_seq_len, 1, device=key.device)
            key_scaled = key_batch * key_scale_batch

            for i in range(q_seq_len):
                q_slice = query[batch_id, i].to(torch.float32)
                dot_product = torch.matmul(q_slice, key_scaled.transpose(0, 1))
                relu_out = torch.maximum(dot_product, torch.tensor(0.0))
                q_scale_slice = query_dequant_scale[batch_id, i].unsqueeze(-1) if query_quant_mode != 0 else torch.ones(head_num, 1, device=query.device)
                scaled_out = relu_out * q_scale_slice
                index_score[batch_id, i] = torch.sum(scaled_out, dim=0)

        return index_score


class MojoQuantLightningIndexer(MojoOperator):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_dequant_scale: torch.Tensor,
        key_dequant_scale: torch.Tensor,
        query_quant_mode: int,
        key_quant_mode: int,
        *,
        actual_seq_lengths_query: Optional[torch.Tensor] = None,
        actual_seq_lengths_key: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        layout_query: str = "BSND",
        layout_key: str = "PA_BSND",
        sparse_count: int = 2048,
        sparse_mode: int = 3,
        pre_tokens: int = 9223372036854775807,
        next_tokens: int = 9223372036854775807,
        cmp_ratio: int = 1,
        return_value: bool = False,
    ):
        """
        Quantized lightning index calculation with query and key scaling.

        Args:
            query: Query tensor. Shape ``[B, M, H, K]``, where B is batch size,
                M is the sequence length of query, H is head number, K is head dimension.
            key: Key tensor. Shape ``[B, N, K]`` or other layout.
            weights: Value tensor or weights tensor.
            query_dequant_scale: Query dequantization scale.
            key_dequant_scale: Key dequantization scale.
            query_quant_mode: Query quantization mode (0: no quant, 1: per-tensor, 2: per-channel).
            key_quant_mode: Key quantization mode (0: no quant, 1: per-tensor, 2: per-channel).
            actual_seq_lengths_query: Actual sequence lengths for query.
            actual_seq_lengths_key: Actual sequence lengths for key.
            block_table: Block table for sparse attention.
            metadata: Metadata tensor for advanced processing.
            layout_query: Layout of query tensor.
            layout_key: Layout of key tensor.
            sparse_count: Sparse count parameter.
            sparse_mode: Sparse mode parameter.
            pre_tokens: Pre-tokens parameter.
            next_tokens: Next tokens parameter.
            cmp_ratio: Comparison ratio.
            return_value: Whether to return value.

        Returns:
            index_score: Index score tensor. Shape ``[B, M, N]``.
            (Optional) value_output: Value output tensor if return_value is True.
        """
        batch_size, q_seq_len, head_num, head_dim = query.shape
        k_seq_len = key.shape[1]

        # Dequantize query if needed
        if query_quant_mode != 0:
            # query: [B, M, H, K], query_dequant_scale: [B, M]
            query = query.to(torch.float32) * query_dequant_scale.unsqueeze(-1).unsqueeze(-1)

        # Dequantize key if needed
        if key_quant_mode != 0:
            # key: [B, N, K], key_dequant_scale: [B, N]
            key = key.to(torch.float32) * key_dequant_scale.unsqueeze(-1)

        index_score = torch.zeros(
            (batch_size, q_seq_len, k_seq_len),
            dtype=torch.float32,
            device=query.device,
        )

        for batch_id in range(batch_size):
            key_batch = key[batch_id].to(torch.float32)  # [N, K]
            key_scale_batch = key_dequant_scale[batch_id].unsqueeze(-1) if key_quant_mode != 0 else torch.ones_like(key_batch[:, :1])
            key_scaled = key_batch * key_scale_batch  # [N, K]

            for i in range(q_seq_len):
                q_slice = query[batch_id, i].to(torch.float32)  # [H, K]
                dot_product = torch.matmul(q_slice, key_scaled.transpose(0, 1))  # [H, N]
                relu_out = torch.maximum(dot_product, torch.tensor(0.0))
                q_scale_slice = query_dequant_scale[batch_id, i].unsqueeze(-1) if query_quant_mode != 0 else torch.ones(head_num, 1, device=query.device)
                scaled_out = relu_out * q_scale_slice
                index_score[batch_id, i] = torch.sum(scaled_out, dim=0)

        if return_value:
            return index_score, weights
        return index_score
