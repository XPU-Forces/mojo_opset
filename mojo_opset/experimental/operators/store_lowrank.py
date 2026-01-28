import torch

from mojo_opset.core.operator import MojoOperator


class MojoStoreLowrank(MojoOperator):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        label_cache: torch.Tensor,
        key_lr: torch.Tensor,
        block_idxs: torch.Tensor,
        token_idxs: torch.Tensor,
        token_num: int,
    ) -> torch.Tensor:
        assert label_cache.dim() == 4, "Expected label_cache is BNSD"
        assert key_lr.dim() == 3, "Expected key_lr is SND"

        label_cache[block_idxs, :, token_idxs, :] = key_lr[:token_num]
        return label_cache
