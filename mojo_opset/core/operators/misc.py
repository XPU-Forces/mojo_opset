import torch
import math

def hadamard(n: int, dtype, device):
    """
    Torch version hadamard matrix generation
    refer to https://pytorch.org/blog/hadacore/
    """
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))

    if 2**lg2 != n:
        raise ValueError(f"n must be a power of 2, but got {n}")

    H = torch.tensor([1], dtype=dtype, device=device)
    for _ in range(0, lg2):
        H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))
    return H
