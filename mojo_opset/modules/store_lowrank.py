import torch

from mojo_opset import functions


class StoreLowrank(torch.nn.Module):
    def __init__(self, *, implementation=None):
        super().__init__()
        self.implementation = implementation

    def forward(self, label_cache, key_lr, block_idxs, token_idxs, token_num):
        return functions.store_lowrank(
            label_cache, key_lr, block_idxs, token_idxs, token_num, implementation=self.implementation
        )
