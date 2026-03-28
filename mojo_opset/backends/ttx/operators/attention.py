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


class TTXPagedPrefillGQA(MojoPagedPrefillGQA):
    supported_platforms_list = ["npu"]

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
        seqlens_kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        assert self.window_size == -1, (
            f"[TTXPagedPrefillGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        )
        assert self.is_causal, (
            f"[TTXPagedPrefillGQA] TTX only support causal attention, but got is_causal={self.is_causal}"
        )
        assert mask is None, f"[TTXPagedPrefillGQA] TTX does not support mask, but got mask={mask}"
        if self.aux_mask is None:
            self.aux_mask = torch.ones(self.AUX_MASK_SIZE, self.AUX_MASK_SIZE * 3, dtype=torch.bool).tril(self.AUX_MASK_SIZE).npu()

        output = paged_attention_prefill(
            q=query,
            key_cache=key_cache,
            value_cache=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqlens_kv=seqlens_kv,
            block_tables=block_tables,
            gqa_interleave=self.gqa_layout == "ABAB",
            softmax_scale=softmax_scale,
            aux_mask=self.aux_mask,
        )

        return output


class TTXPagedDecodeGQA(MojoPagedDecodeGQA):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        assert self.window_size == -1, (
            f"[TTXPagedDecodeGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        )
        assert self.is_causal, (
            f"[TTXPagedDecodeGQA] TTX only support causal attention, but got is_causal={self.is_causal}"
        )
        assert mask is None, f"[TTXPagedDecodeGQA] TTX does not support mask, but got mask={mask}"

        output = paged_attention_decode(
            q=query,
            key_cache=key_cache,
            value_cache=value_cache,
            seqlens=seqlens,
            block_tables=block_tables,
            gqa_interleave=self.gqa_layout == "ABAB",
            softmax_scale=softmax_scale,
        )

        return output


class TTXSdpa(MojoSdpa):
    supported_platforms_list = ["npu"]

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
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k_cache: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v_cache: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        block_table: torch.Tensor,  # [bsz, num_kv_blocks]
        softmax_scale: Optional[float] = None,
        seqlens_kv: Optional[torch.Tensor] = None,  # [bsz]
    ) -> torch.Tensor:
        if seqlens_kv is None:
            seqlens_kv = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

        o = swa_paged_prefill(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            seqlens_kv,
            block_table,
            self.is_causal,
            self.local_window_size,
            self.global_window_size,
            softmax_scale,
            self.gqa_interleave,
        )
        return o


class TTXPagedDecodeSWA(MojoPagedDecodeSWA):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,  # [bsz, n_q_heads, head_dim]
        k_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        v_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        seq_lens: torch.Tensor,  # [bsz]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        # Note: is_causal = False should never happen

        o = swa_paged_decode(
            q,
            k_cache,
            v_cache,
            seq_lens,
            block_table,
            self.local_window_size,
            self.global_window_size,
            self.gqa_interleave,
            softmax_scale,
        )
        
        return o


class TTXSWA(MojoSWA):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        cu_seqlens_kv: torch.Tensor,  # [bsz + 1]
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:

        o = swa_infer(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            self.is_causal,
            self.local_window_size,
            self.global_window_size,
            softmax_scale,
            self.gqa_interleave,
        )
        return o

