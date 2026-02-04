import math
from typing import Tuple

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from triton.runtime.libentry import libentry


# hadamard
@triton.jit
def hadamard_kernel(
    output_ptr,
    n: tl.constexpr,
    log2_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    output shape: [n, n]
    """
    pid = tl.program_id(axis=0)
    
    row_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offs < n
    
    for col_block in range(0, n, BLOCK_SIZE):
        col_offs = col_block + tl.arange(0, BLOCK_SIZE)
        col_mask = col_offs < n
        
        row_indices = row_offs[:, None]
        col_indices = col_offs[None, :]
        
        result = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1, dtype=tl.int8)
        
        for k in range(log2_n):
            row_bit = (row_indices >> k) & 1
            col_bit = (col_indices >> k) & 1
            
            condition = (row_bit == 1) & (col_bit == 1)
            result = tl.where(condition, -result, result)
        
        output_offset = row_offs[:, None] * n + col_offs[None, :]
        output_ptr_block = output_ptr + output_offset
        
        block_mask = row_mask[:, None] & col_mask[None, :]
        
        tl.store(output_ptr_block, result, mask=block_mask)


def hadamard_triton(n: int, dtype, device):
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, but got {n}")
    
    log2_n = int(math.log2(n))
    
    output = torch.empty((n, n), dtype=dtype, device=device)
    
    if n <= 32:
        BLOCK_SIZE = 16
    elif n <= 128:
        BLOCK_SIZE = 32
    elif n <= 512:
        BLOCK_SIZE = 64
    else:
        BLOCK_SIZE = 128
    
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    hadamard_kernel[grid](
        output,
        n=n,
        log2_n=log2_n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def indexer_rotate_activation_impl(x: torch.Tensor)-> torch.Tensor:
    hidden_size = x.size(-1)
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard_triton(dim_padded, dtype=x.dtype, device=x.device).transpose(0, 1))
    out = out * hidden_size**-0.5
    return out[..., :dim].reshape(*x_shape)