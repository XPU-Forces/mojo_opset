from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import paged_attention_decode
from mojo_opset.backends.ttx.kernels import paged_attention_prefill
from mojo_opset.backends.ttx.kernels import sdpa_infer
from mojo_opset.backends.ttx.kernels import block_sparse_attention
from mojo_opset.backends.ttx.kernels import block_sparse_attention_paged_prefill

from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoSdpa
from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoSdpa
from mojo_opset.core import MojoBlockSparseAttention
from mojo_opset.core import MojoPagedPrefillBlockSparseAttention


class TTXPagedPrefillGQA(MojoPagedPrefillGQA):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ):
        assert (
            self.window_size == -1
        ), f"[TTXPagedPrefillGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        assert (
            self.gqa_layout == "ABAB"
        ), f"[TTXPagedPrefillGQA] TTX only support ABAB layout, but got gqa_layout={self.gqa_layout}"
        assert (
            self.is_causal
        ), f"[TTXPagedPrefillGQA] TTX only support causal attention, but got is_causal={self.is_causal}"

        output = paged_attention_prefill(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            block_tables=block_tables,
            sm_scale=softmax_scale,
        )

        return output


class TTXPagedDecodeGQA(MojoPagedDecodeGQA):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ):
        assert (
            self.window_size == -1
        ), f"[TTXPagedPrefillGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        assert (
            self.gqa_layout == "ABAB"
        ), f"[TTXPagedPrefillGQA] TTX only support ABAB layout, but got gqa_layout={self.gqa_layout}"
        assert (
            self.is_causal
        ), f"[TTXPagedPrefillGQA] TTX only support causal attention, but got is_causal={self.is_causal}"

        output = paged_attention_decode(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            seqlens=seqlens,
            block_tables=block_tables,
            sm_scale=softmax_scale,
        )

        return output


class TTXBlockSparseAttention(MojoBlockSparseAttention):
    supported_platforms_list = ["npu"]

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
        output = block_sparse_attention(
            curr_query_seg,
            key,
            value,
            self.scale,
            self.mask,
            topk_page_indices,
            q_seg_id,
            q_chunk_size,
            self.q_seg_size,
            self.page_size,
        )
        return output


class TTXPagedPrefillBlockSparseAttention(MojoPagedPrefillBlockSparseAttention):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        whole_causal_mask,
        block_table,
        q_chunk_indices,
        selected_page_indices,
        cu_num_selected_pages_per_chunk,
    ):
        output = block_sparse_attention_paged_prefill(
            query,
            key_cache,
            value_cache,
            self.scale,
            cu_seqlens_q,
            cu_seqlens_k,
            self.mask,
            block_table,
            q_chunk_indices,
            selected_page_indices,
            cu_num_selected_pages_per_chunk,
            self.q_seg_size,
            self.page_size,
        )
        return output


class TTXSdpa(MojoSdpa):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        output = sdpa_infer(
            q=query,
            k=key,
            v=value,
            mask=self.mask,
            scale=self.scale,
            enable_gqa=self.enable_gqa,
        )
        return output
