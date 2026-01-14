import contextlib
import functools

from functools import lru_cache
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from packaging import version

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


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def input_guard(
    *,
    make_contiguous: bool = True,
    auto_to_device: bool = True,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """
    A decorator to optionally:
      1. make all input tensors contiguous
      2. set device context based on input tensors
    """

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if make_contiguous:
                new_args = tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
                new_kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            else:
                new_args = args
                new_kwargs = kwargs

            tensor = None
            for a in new_args:
                if isinstance(a, torch.Tensor):
                    tensor = a
                    break
            if tensor is None:
                for v in new_kwargs.values():
                    if isinstance(v, torch.Tensor):
                        tensor = v
                        break

            if auto_to_device and tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*new_args, **new_kwargs)

        return wrapper

    return decorator


contiguous = input_guard


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = "2.4") -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        return "meta_device"


device = get_available_device()
device_torch_lib = getattr(torch, device)


if check_pytorch_version("2.4"):
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)

else:
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.npu.device(index)


if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Just make triton compiler happy.
    """

    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    lens = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    total = lens.sum()
    flat = torch.arange(total, device=cu_seqlens.device)
    seq_ids = torch.repeat_interleave(torch.arange(lens.numel(), device=cu_seqlens.device), lens)
    offsets = torch.cumsum(lens, 0) - lens
    indices = flat - offsets[seq_ids]
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], dim=1)
