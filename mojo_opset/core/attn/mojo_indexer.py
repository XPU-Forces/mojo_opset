from typing import Any
from typing import Optional
from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoLightningIndexer(MojoOperator):
    def __init__(
        self,
        top_k: int = 2048,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.top_k = top_k

    def forward_std(
        self, query, query_scale, key, key_scale: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(
        self, query, query_scale, key, key_scale: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Any]:
        batch_size, q_seq_len, _, _ = query.shape
        k_seq_len = key.shape[1]
        index_score = torch.zeros((batch_size, q_seq_len, k_seq_len), dtype=torch.float32, device=query.device)

        for batch_id in range(batch_size):
            for i in range(q_seq_len):
                q_slice = query[batch_id, i].to(torch.float32)
                k_slice = key[batch_id].to(torch.float32)
                relu_out = torch.maximum(
                    torch.matmul(q_slice.to(torch.float32), k_slice.to(torch.float32).transpose(0, 1)),
                    torch.tensor(0),
                )
                weight_out = relu_out * query_scale[batch_id, i].unsqueeze(-1)
                reduce_out = torch.sum(weight_out, dim=0)
                index_score[batch_id, i] = reduce_out

        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(self.top_k, dim=-1)[1]

        return topk_indices

    def forward_analysis(
        self, query, query_scale, key, key_scale: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, int, int]:
        pass
