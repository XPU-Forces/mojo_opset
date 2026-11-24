"""Configuration-only modules for inference attention."""

from typing import Optional

import torch

from mojo_opset import functions


class PagedDecodeGQAInfer(torch.nn.Module):
    def __init__(self, *, is_causal=True, gqa_layout="AABB", implementation=None):
        super().__init__()
        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.implementation = implementation

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        return functions.paged_decode_gqa_infer(
            query,
            key_cache,
            value_cache,
            total_seq_lens,
            block_tables,
            softmax_scale,
            mask,
            max_total_seq_len,
            is_causal=self.is_causal,
            gqa_layout=self.gqa_layout,
            implementation=self.implementation,
        )


class PrefillGQAInfer(torch.nn.Module):
    def __init__(self, *, is_causal=True, gqa_layout="ABAB", implementation=None):
        super().__init__()
        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.implementation = implementation

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        return functions.prefill_gqa_infer(
            query,
            k_cache,
            v_cache,
            cu_q_lens,
            softmax_scale,
            is_causal=self.is_causal,
            gqa_layout=self.gqa_layout,
            implementation=self.implementation,
        )


class PagedPrefillGQAInfer(torch.nn.Module):
    def __init__(self, *, is_causal=True, gqa_layout="AABB", implementation=None):
        super().__init__()
        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.implementation = implementation
        self.scheduler_metadata = None

    def prepare_metadata(self, cu_q_lens, cu_total_seq_lens, num_q_heads, num_kv_heads, page_size):
        """Prepare scheduler tables before graph capture; rebuild after changing lengths."""
        from mojo_opset.functions.paged_attention import _prepare_paged_prefill_metadata

        self.scheduler_metadata = _prepare_paged_prefill_metadata(
            cu_q_lens,
            cu_total_seq_lens,
            num_q_heads,
            num_kv_heads,
            page_size,
            self.gqa_layout,
            self.implementation,
        )
        return self.scheduler_metadata

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        return functions.paged_prefill_gqa_infer(
            query,
            key_cache,
            value_cache,
            cu_q_lens,
            block_tables,
            softmax_scale,
            cu_total_seq_lens,
            mask,
            max_q_len,
            max_total_seq_len,
            is_causal=self.is_causal,
            gqa_layout=self.gqa_layout,
            scheduler_metadata=self.scheduler_metadata,
            implementation=self.implementation,
        )


class PagedPrefillSWAInfer(torch.nn.Module):
    def __init__(
        self, *, is_causal=True, gqa_layout="AABB", global_window_size=None, local_window_size=None, implementation=None
    ):
        super().__init__()
        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.global_window_size = global_window_size
        self.local_window_size = local_window_size
        self.implementation = implementation

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        return functions.paged_prefill_swa_infer(
            query,
            key_cache,
            value_cache,
            cu_q_lens,
            block_table,
            softmax_scale,
            cu_total_seq_lens,
            max_q_len,
            max_total_seq_len,
            is_causal=self.is_causal,
            gqa_layout=self.gqa_layout,
            global_window_size=self.global_window_size,
            local_window_size=self.local_window_size,
            implementation=self.implementation,
        )


class PagedDecodeSWAInfer(torch.nn.Module):
    def __init__(
        self, *, is_causal=True, gqa_layout="AABB", global_window_size=None, local_window_size=None, implementation=None
    ):
        super().__init__()
        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.global_window_size = global_window_size
        self.local_window_size = local_window_size
        self.implementation = implementation

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: Optional[float] = None,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        return functions.paged_decode_swa_infer(
            query,
            key_cache,
            value_cache,
            total_seq_lens,
            block_table,
            softmax_scale,
            max_total_seq_len,
            is_causal=self.is_causal,
            gqa_layout=self.gqa_layout,
            global_window_size=self.global_window_size,
            local_window_size=self.local_window_size,
            implementation=self.implementation,
        )
