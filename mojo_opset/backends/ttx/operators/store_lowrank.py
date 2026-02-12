import torch

from mojo_opset.backends.ttx.kernels import store_label_cache_infer
from mojo_opset.experimental import MojoStoreLowrank


class TTXStoreLowrank(MojoStoreLowrank):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        label_cache: torch.Tensor,
        key_lr: torch.Tensor,
        block_idxs: torch.Tensor,
        token_idxs: torch.Tensor,
        token_num: int,
    ):
        return store_label_cache_infer(
            label_cache,
            key_lr,
            block_idxs,
            token_idxs,
            token_num,
        )
