import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime.libentry import libentry


def indexer_rotate_activation_impl(x: torch.Tensor) -> torch.Tensor:
    from mojo_opset.backends.ttx.operators.linear import matmul
    from mojo_opset.core.operators.misc import hadamard

    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    dim_padded = 2 ** math.ceil(math.log2(dim))

    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    # TODO fuse matmul and hadamard
    hadamard_tensor = hadamard(dim_padded, dtype=x.dtype, device=x.device)
    out = matmul(x, hadamard_tensor)
    out = out * dim**-0.5
    return out[..., :dim].reshape(*x_shape)
