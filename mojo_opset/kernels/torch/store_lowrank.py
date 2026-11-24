import torch


def store_lowrank_fwd(
    label_cache: torch.Tensor, key_lr: torch.Tensor, block_idxs: torch.Tensor, token_idxs: torch.Tensor, token_num: int
) -> None:
    # Keep advanced indices adjacent. NPU index_put mishandles the implicit
    # head slice between block/token indices when the cache has multiple heads.
    heads = torch.arange(label_cache.shape[1], device=label_cache.device)
    label_cache[
        block_idxs[:token_num].long()[:, None], heads[None, :], token_idxs[:token_num].long()[:, None], :
    ] = key_lr[:token_num]
