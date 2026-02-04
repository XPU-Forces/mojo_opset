import torch
import torch_npu

from typing import Optional

from mojo_opset.core import MojoPagedPrefillAttention

class TorchNpuPagedPrefillAttention(MojoPagedPrefillAttention):
    supported_platforms_list = ["npu"]

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
        actual_seq_qlen = cu_seqlens_q[1:]
        actual_seq_kvlen = torch.diff(cu_seqlens_kv)
        _, num_query_heads, _ = query.shape
        _, num_key_value_heads, block_size, _ = key_query.shape
        if self.is_causal:
            out_npu, _ = torch_npu.npu_fused_infer_attention_score_v2(query, key_query, value_query, 
                actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, block_table=block_tables,
                num_query_heads=num_query_heads, num_key_value_heads=num_key_value_heads, softmax_scale=softmax_scale,
                sparse_mode=3, input_layout="TND", block_size=block_size)
        else:
            out_npu, _ = torch_npu.npu_fused_infer_attention_score_v2(query, key_query, value_query, 
                actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, block_table=block_tables,
                num_query_heads=num_query_heads, num_key_value_heads=num_key_value_heads, softmax_scale=softmax_scale,
                input_layout="TND", block_size=block_size)
        return out_npu
