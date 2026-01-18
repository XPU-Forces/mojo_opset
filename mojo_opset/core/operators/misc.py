import torch

from ..operator import MojoOperator


class MojoQuant(MojoOperator):
    pass


class MojoDequant(MojoOperator):
    pass


class MojoEmbedding(MojoOperator):
    pass


class MojoParallelEmbedding(MojoOperator):
    pass


class MojoQuest(MojoOperator):
    """
    Quest indexing operator for LLM Prefill.
    """

    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

    def forward(self, query, mins, maxs, top_k_page):

        # [num_heads, q_len, 1, head_size] * [num_heads, 1, num_pages, head_size]
        # --> [num_heads, q_len, num_pages, head_size]
        query = query.float()
        mins = mins.float()
        maxs = maxs.float()
        q_min_k = query.unsqueeze(-2) * mins.unsqueeze(-3)
        q_max_k = query.unsqueeze(-2) * maxs.unsqueeze(-3)

        # [num_heads, num_segs, num_pages]
        page_score = torch.maximum(q_min_k, q_max_k).sum(dim=-1)
        # [num_heads, num_segs, top_k_page]
        _, topk_page_indices = page_score.topk(top_k_page, dim=-1)

        return topk_page_indices


class MojoPagedPrefillBlockQuest(MojoOperator):
    def __init__(
        self,
        chunk_size: int,
        page_size: int,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.chunk_size = chunk_size
        self.page_size = page_size

    def forward(self, query, cu_seqlens_q, page_k_mins, page_k_maxs, kv_cache_indices, cu_seqlens_k, num_topk_pages):
        q_head_num, _, head_size = query.shape
        kv_head_num = page_k_mins.shape[1]

        topk_pages = []
        for i in range(len(kv_cache_indices)):
            kv_len = cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item()
            top_k_page = num_topk_pages[i].item()

            sublist = kv_cache_indices[i]
            valid_mask = sublist != -1
            valid_indices = sublist[valid_mask]

            # if prefill_sparse and kv_len[i] > 0 and valid_kv_seq_length > sparse_limit:

            valid_num_pages = kv_len // self.page_size
            mins = (
                page_k_mins.index_select(0, valid_indices)[:valid_num_pages]
                .permute(1, 0, 2)
                .repeat_interleave(q_head_num // kv_head_num, dim=0)
                .contiguous()
            )
            maxs = (
                page_k_maxs.index_select(0, valid_indices)[:valid_num_pages]
                .permute(1, 0, 2)
                .repeat_interleave(q_head_num // kv_head_num, dim=0)
                .contiguous()
            )

            # [num_heads, q_len, 1, head_size] * [num_heads, 1, num_pages, head_size]
            # --> [num_heads, q_len, num_pages, head_size]
            curr_query = query[:, cu_seqlens_q[i] : cu_seqlens_q[i + 1] : self.chunk_size].float()
            q_min_k = curr_query.unsqueeze(-2) * mins.float().unsqueeze(-3)
            q_max_k = curr_query.unsqueeze(-2) * maxs.float().unsqueeze(-3)

            # [num_heads, num_segs, num_pages]
            page_score = torch.maximum(q_min_k, q_max_k).sum(dim=-1)
            # [num_heads, num_segs, top_k_page]
            _, topk_page_indices = page_score.topk(top_k_page, dim=-1)
            topk_pages.append(topk_page_indices.reshape(q_head_num, -1))

        return torch.cat(topk_pages, dim=1)
