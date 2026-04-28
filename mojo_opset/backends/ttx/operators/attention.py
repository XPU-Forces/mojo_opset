from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import paged_attention_prefill
from mojo_opset.backends.ttx.kernels import paged_attention_decode
from mojo_opset.backends.ttx.kernels import sdpa_infer
from mojo_opset.backends.ttx.kernels import swa_paged_prefill
from mojo_opset.backends.ttx.kernels import swa_paged_decode
from mojo_opset.backends.ttx.kernels import swa_infer
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoSdpa
from mojo_opset.core import MojoPagedPrefillSWA
from mojo_opset.core import MojoPagedDecodeSWA
from mojo_opset.core import MojoSWA
from mojo_opset.core.operators.attention import assert_paged_decode_contract
from mojo_opset.core.operators.attention import assert_paged_prefill_contract


class TTXPagedPrefillGQA(MojoPagedPrefillGQA):
    supported_platforms_list = ["npu", "ilu", "mlu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AUX_MASK_SIZE = 1024
        self.aux_mask = None

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_total_seqlen: Optional[int] = None,
    ):
        assert_paged_prefill_contract(cu_seqlens_q, block_tables, cu_total_seqlens)
        assert self.window_size == -1, (
            f"[TTXPagedPrefillGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        )
        assert self.is_causal, (
            f"[TTXPagedPrefillGQA] TTX only support causal attention, but got is_causal={self.is_causal}"
        )
        assert mask is None, f"[TTXPagedPrefillGQA] TTX does not support mask, but got mask={mask}"
        total_seq_lens = (
            cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            if cu_total_seqlens is None
            else cu_total_seqlens[1:] - cu_total_seqlens[:-1]
        )
        # max_seqlen_q / max_total_seqlen / kwargs: core·Ixformer API compatibility; kernel uses per-seq lengths only.
        if self.aux_mask is None:
            self.aux_mask = torch.ones(
                self.AUX_MASK_SIZE,
                self.AUX_MASK_SIZE * 3,
                dtype=torch.bool,
                device=query.device,
            ).tril(self.AUX_MASK_SIZE)

        output = paged_attention_prefill(
            q=query,
            key_cache=key_cache,
            value_cache=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqlens_kv=total_seq_lens,
            block_tables=block_tables,
            gqa_interleave=self.gqa_layout == "ABAB",
            softmax_scale=softmax_scale,
            aux_mask=self.aux_mask,
        )

        return output


class TTXPagedDecodeGQA(MojoPagedDecodeGQA):
    supported_platforms_list = ["npu", "ilu", "mlu"]

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ):
        assert total_seq_lens.dtype == torch.int32
        assert block_tables.dtype == torch.int32
        assert_paged_decode_contract(block_tables, total_seq_lens)
        assert self.window_size == -1, (
            f"[TTXPagedDecodeGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        )
        assert self.is_causal, (
            f"[TTXPagedDecodeGQA] TTX only support causal attention, but got is_causal={self.is_causal}"
        )
        assert mask is None, f"[TTXPagedDecodeGQA] TTX does not support mask, but got mask={mask}"
        assert cu_q_lens is None, "varlen is not supported"
        _ = max_total_seq_len  # API parity with core / Ixformer; ILU kernel uses per-row total sequence lengths.
        output = paged_attention_decode(
            q=query,
            key_cache=key_cache,
            value_cache=value_cache,
            seqlens=total_seq_lens,
            block_tables=block_tables,
            gqa_interleave=self.gqa_layout == "ABAB",
            softmax_scale=softmax_scale,
        )

        return output


class TTXSdpa(MojoSdpa):
    supported_platforms_list = ["npu", "ilu"]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        output = sdpa_infer(
            q=query,
            k=key,
            v=value,
            mask=attn_mask,
            scale=self.scale,
            enable_gqa=self.enable_gqa,
        )
        return output

class TTXPagedPrefillSWA(MojoPagedPrefillSWA):
    supported_platforms_list = ["npu", "mlu", "ilu"]

    def forward(
        self,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k_cache: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v_cache: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        block_table: torch.Tensor,  # [bsz, num_kv_blocks]
        softmax_scale: Optional[float] = None,
        cu_total_seqlens: Optional[torch.Tensor] = None,  # [bsz + 1]
    ) -> torch.Tensor:
        assert_paged_prefill_contract(cu_seqlens_q, block_table, cu_total_seqlens)
        total_seq_lens = (
            cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            if cu_total_seqlens is None
            else cu_total_seqlens[1:] - cu_total_seqlens[:-1]
        )

        o = swa_paged_prefill(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            total_seq_lens,
            block_table,
            self.is_causal,
            self.local_window_size,
            self.global_window_size,
            softmax_scale,
            self.gqa_interleave,
        )
        return o


class TTXPagedDecodeSWA(MojoPagedDecodeSWA):
    supported_platforms_list = ["npu", "mlu", "ilu"]

    def forward(
        self,
        q: torch.Tensor,  # [bsz, n_q_heads, head_dim]
        k_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        v_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        total_seq_lens: torch.Tensor,  # [bsz]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        # Note: is_causal = False should never happen
        assert total_seq_lens.dtype == torch.int32
        assert block_table.dtype == torch.int32
        assert_paged_decode_contract(block_table, total_seq_lens)
        o = swa_paged_decode(
            q,
            k_cache,
            v_cache,
            total_seq_lens,
            block_table,
            self.local_window_size,
            self.global_window_size,
            self.gqa_interleave,
            softmax_scale,
        )

        return o


class TTXSWA(MojoSWA):
    supported_platforms_list = ["npu", "ilu"]

    def forward(
        self,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        cu_total_seqlens: torch.Tensor,  # [bsz + 1]
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        assert cu_seqlens_q.dtype == torch.int32
        assert cu_total_seqlens.dtype == torch.int32
        o = swa_infer(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_total_seqlens,
            self.is_causal,
            self.local_window_size,
            self.global_window_size,
            softmax_scale,
            self.gqa_interleave,
        )
        return o
