import torch


def store_lowrank_fwd(
    label_cache: torch.Tensor, key_lr: torch.Tensor, block_idxs: torch.Tensor, token_idxs: torch.Tensor, token_num: int
) -> None:
    label_cache[block_idxs, :, token_idxs, :] = key_lr[:token_num]
