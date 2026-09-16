"""The original NPU Indexer used this device-independent Torch Hadamard formula."""

import torch


def rotate_activation_fwd(x):
    dim = x.shape[-1]
    padded_dim = 1 << (dim - 1).bit_length()
    rows = x.reshape(-1, dim)
    if dim != padded_dim:
        rows = torch.nn.functional.pad(rows, (0, padded_dim - dim))
    hadamard = x.new_ones((1, 1))
    for _ in range(padded_dim.bit_length() - 1):
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0)
    output = torch.nn.functional.linear(rows, hadamard) * dim**-0.5
    return output[..., :dim].reshape(x.shape)
