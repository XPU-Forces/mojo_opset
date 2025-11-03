import torch
import triton.language as tl
import triton

VEC_ALIGN_BYTES = 256


torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def ceil_div(x, y):
    return (x + y - 1) // y


def align(x, n_cols, aligned_bytes):
    aligned_bytes = ceil_div(x.element_size() * n_cols, aligned_bytes) * aligned_bytes
    return aligned_bytes // x.element_size()


@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)


@triton.jit
def store_with_pred_1d(ptr, value, skip_boundary_check: tl.constexpr, mask: tl.tensor):
    if not skip_boundary_check:
        tl.store(ptr, value, mask)
    else:
        tl.store(ptr, value)
