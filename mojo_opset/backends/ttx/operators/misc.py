import torch
from mojo_opset.backends.ttx.kernels import quest
from mojo_opset.backends.ttx.kernels import paged_prefill_block_quest
from mojo_opset.core import MojoQuest
from mojo_opset.core import MojoPagedPrefillBlockQuest


class TTXQuest(MojoQuest):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        mins: torch.Tensor,
        maxs: torch.Tensor,
        top_k_page: int,
    ):
        return quest(query, mins, maxs, top_k_page)


class TTXPagedPrefillBlockQuest(MojoPagedPrefillBlockQuest):
    supported_platforms_list = ["npu"]

    def forward(self, query, cu_seqlens_q, page_k_mins, page_k_maxs, kv_cache_indices, cu_seqlens_k, num_topk_pages):
        return paged_prefill_block_quest(
            query,
            cu_seqlens_q,
            page_k_mins,
            page_k_maxs,
            kv_cache_indices,
            cu_seqlens_k,
            num_topk_pages,
            self.chunk_size,
            self.page_size,
        )
