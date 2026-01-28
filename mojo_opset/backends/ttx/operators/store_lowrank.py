import torch

from mojo_opset.backends.ttx.kernels.npu.store_lowrank import store_label_cache_fwd_impl
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
        return store_label_cache_fwd_impl(
            label_cache,
            key_lr,
            block_idxs,
            token_idxs,
            token_num,
        )
