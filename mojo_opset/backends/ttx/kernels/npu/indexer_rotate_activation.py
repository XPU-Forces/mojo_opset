import math

import torch
import torch.nn.functional as F


def indexer_rotate_activation_impl(x: torch.Tensor) -> torch.Tensor:
    from mojo_opset.core.operators.misc import hadamard

    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    dim_padded = 2 ** math.ceil(math.log2(dim))

    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))

    hadamard_tensor = hadamard(dim_padded, dtype=x.dtype, device=x.device)
    linear = torch.nn.Linear(dim_padded, dim_padded, bias=False, device=x.device, dtype=x.dtype)
    linear.weight = torch.nn.Parameter(hadamard_tensor, requires_grad=False)
    out = linear(x)
    out = out * dim**-0.5
    return out[..., :dim].reshape(*x_shape)
