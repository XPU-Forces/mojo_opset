from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import paged_attention_decode
from mojo_opset.backends.ttx.kernels import paged_attention_prefill
from mojo_opset.backends.ttx.kernels import sdpa_infer
from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoSdpa


class TTXPagedPrefillGQA(MojoPagedPrefillGQA):
    supported_platforms_list = ["npu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        AUX_MASK_SIZE=1024
        self.aux_mask = torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE*3, dtype=torch.bool).tril(AUX_MASK_SIZE).npu()


    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqlens_kv: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ):
        assert self.window_size == -1, (
            f"[TTXPagedPrefillGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        )
        assert self.is_causal, (
            f"[TTXPagedPrefillGQA] TTX only support causal attention, but got is_causal={self.is_causal}"
        )
        output = paged_attention_prefill(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqlens_kv=seqlens_kv,
            block_tables=block_tables,
            gqa_interleave=self.gqa_layout == "ABAB",
            sm_scale=softmax_scale,
            aux_mask=self.aux_mask,
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
        assert self.window_size == -1, (
            f"[TTXPagedPrefillGQA] TTX does not support sliding window, but got window_size={self.window_size}"
        )
        assert self.gqa_layout == "ABAB", (
            f"[TTXPagedPrefillGQA] TTX only support ABAB layout, but got gqa_layout={self.gqa_layout}"
        )
        assert self.is_causal, (
            f"[TTXPagedPrefillGQA] TTX only support causal attention, but got is_causal={self.is_causal}"
        )

        output = paged_attention_decode(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            seqlens=seqlens,
            block_tables=block_tables,
            sm_scale=softmax_scale,
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
