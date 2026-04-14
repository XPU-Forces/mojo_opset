from __future__ import annotations

from typing import Optional

import torch

from mojo_opset.core import MojoPagedPrefillGQA

from ixformer.contrib import vllm_flash_attn as  ix_fa

class IxformerPagedPrefillGQA(MojoPagedPrefillGQA):
    """Ixformer implementation for paged prefill GQA."""

    supported_platforms_list = ["ilu"]

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
    ) -> torch.Tensor:
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

        if mask is not None:
            raise NotImplementedError("IxformerPagedPrefillGQA does not support custom mask yet.")
        if self.gqa_layout == "ABAB":
            raise NotImplementedError("IxformerPagedPrefillGQA does not support ABAB layout.")
        page_block_size = key_cache.shape[2]
        if page_block_size > 256:
            raise NotImplementedError(
                f"IxformerPagedPrefillGQA only supports page_block_size <= 256, got {page_block_size}."
            )

        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5
        if seqlens_kv is None:
            seqlens_kv = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        if self.window_size != -1:
            raise NotImplementedError(
                f"IxformerPagedPrefillGQA only supports window_size=-1, got {self.window_size}."
            )

        # ixformer paged-kernel requires int32 index tensors.
        cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32)
        block_tables = block_tables.to(dtype=torch.int32)

        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()) if cu_seqlens_q.numel() > 1 else 0
        max_seqlen_k = int(seqlens_kv.max().item()) if seqlens_kv.numel() > 0 else 0
        window_size = (-1, -1)

        cu_seqlens_kv = torch.zeros_like(cu_seqlens_q)
        cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv.to(dtype=cu_seqlens_q.dtype), dim=0)

        return ix_fa.flash_attn_varlen_func(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_k,
            causal=self.is_causal,
            softmax_scale=softmax_scale,
            block_table=block_tables,
            window_size=window_size,
        )
