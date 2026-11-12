"""Shared, deterministic workloads for request-to-global index conversion."""

import torch

# T, requests, blocks/request, block size, K, prefill, prefill tokens, prefill requests.
CASES = [
    (8, 4, 10, 16, 128, False, 0, 0),
    (8, 4, 10, 16, 256, False, 0, 0),
    (8, 4, 10, 16, 512, False, 0, 0),
    (8, 4, 10, 64, 128, False, 0, 0),
    (8, 4, 10, 64, 256, False, 0, 0),
    (8, 4, 10, 64, 512, False, 0, 0),
    (8, 4, 10, 128, 128, False, 0, 0),
    (8, 4, 10, 128, 256, False, 0, 0),
    (8, 4, 10, 128, 512, False, 0, 0),
    (8, 4, 10, 8, 128, False, 0, 0),
    (12, 4, 8, 16, 128, True, 6, 2),
    (1, 1, 10, 16, 128, False, 0, 0),
    (64, 8, 10, 16, 128, False, 0, 0),
    (512, 32, 64, 64, 128, False, 0, 0),
    (2048, 64, 128, 128, 128, False, 0, 0),
    (8192, 256, 1024, 64, 512, False, 0, 0),
    (512, 16, 64, 16, 512, False, 0, 0),
    (2048, 128, 1024, 64, 512, False, 0, 0),
    (2048, 128, 1024, 128, 2048, False, 0, 0),
    (512, 32, 64, 128, 2048, False, 0, 0),
    (8192, 512, 1024, 128, 2048, False, 0, 0),
    (2048, 64, 1024, 64, 512, True, 1024, 8),
]


def make_inputs(case, device, *, seed=42, strided=False):
    tokens, requests, blocks, block_size, topk, has_prefill, prefill_tokens, num_prefill = case
    generator = torch.Generator().manual_seed(seed)
    req = torch.randint(requests, (tokens,), dtype=torch.int32, generator=generator)
    table = torch.randint(4096, (requests, blocks), dtype=torch.int32, generator=generator)
    indices = torch.randint(blocks * block_size, (tokens, topk), dtype=torch.int32, generator=generator)
    if tokens:
        indices[0, :4] = -1
    if tokens > 1:
        indices[1, :4] = blocks * block_size
    ids = starts = None
    if has_prefill:
        ids = torch.full((tokens,), -1, dtype=torch.int32)
        if prefill_tokens:
            ids[-prefill_tokens:] = torch.arange(prefill_tokens, dtype=torch.int32) * num_prefill // prefill_tokens
        starts = torch.arange(num_prefill, dtype=torch.int32) * 100

    def transfer(tensor):
        if tensor is None:
            return None
        tensor = tensor.to(device)
        if not strided:
            return tensor
        # Exercise nonzero storage offsets and strides in every tensor dimension.
        storage = torch.full(tuple(2 * n + 1 for n in tensor.shape), -777, dtype=tensor.dtype, device=device)
        view = storage[tuple(slice(1, 2 * n + 1, 2) for n in tensor.shape)]
        view.copy_(tensor)
        return view

    return (
        transfer(req),
        transfer(table),
        transfer(indices),
        block_size,
        topk,
        has_prefill,
        transfer(ids),
        transfer(starts),
    )
