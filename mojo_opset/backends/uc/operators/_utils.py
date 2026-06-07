from functools import lru_cache

import torch


_DTYPE_API_SUFFIX = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


@lru_cache(maxsize=1)
def _uc_kernels():
    import uc_kernel

    return uc_kernel.load()


def _matrix_shape(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.dim() == 0:
        return 1, 1
    if tensor.dim() == 1:
        return 1, tensor.numel()
    return tensor.numel() // tensor.shape[-1], tensor.shape[-1]


def _typed_api(api: str, dtype: torch.dtype) -> str:
    suffix = _DTYPE_API_SUFFIX.get(dtype)
    if suffix is None:
        raise NotImplementedError(f"UC backend {api} does not support dtype {dtype}.")

    kernels = _uc_kernels()
    typed_api = f"{api}_{suffix}"
    if typed_api in kernels.keys():
        return typed_api
    raise NotImplementedError(f"UC backend {api} does not provide a {suffix} kernel artifact.")


# ---------------------------------------------------------------------------
# W7-B-followup (P-Wave-7): module-level FastKernel cache for the generic
# run_unary_kernel / run_binary_kernel / run_kernel dispatch path.
#
# Per W7-B profile (docs/project-ops/perf-debug/wrapper-overhead-reduction-
# pwave7-w7b.md §1.3), ``uc_kernel.runtime.KernelFunction.__call__`` spends
# ~38 µs in ``init_workspace()`` (re-parses _manifest.json every call) +
# ~8.5 µs in ``_stream_ptr(None)`` (torch_npu API query). A wrapper-side
# ``FastKernel`` binds workspace + stream exactly once per (api, dtype),
# then dispatches via the cached extension callable — saving ~80 µs / call
# regardless of shape.
#
# This module-level cache automatically broadcasts that save to every UC
# wrapper that goes through ``run_unary_kernel`` / ``run_binary_kernel`` /
# ``run_kernel`` (16+ ops across activation / quant / dequant /
# over_encoding / etc). No per-wrapper code change required.
#
# Fallback semantics preserved: ``_get_or_create_fast`` raises
# ``NotImplementedError`` matching ``_typed_api()`` if dtype unsupported or
# kernel missing from the wheel — caller's ``super().forward()`` fallback
# path (if any) is untouched.
# ---------------------------------------------------------------------------

# Lazy import to avoid module load order issues (some tests import _utils
# before uc_kernel is importable).
_FAST_KERNEL_CACHE: dict = {}


def _get_or_create_fast(api: str, dtype: torch.dtype):
    """Return a cached :class:`FastKernel` for ``(api, dtype)``.

    Raises ``NotImplementedError`` if ``dtype`` is unsupported or the
    typed-API kernel is not present in the loaded wheel — matching the
    behaviour of :func:`_typed_api`. Callers that previously relied on
    ``_typed_api`` raising will see identical exception semantics.
    """
    key = (api, dtype)
    fast = _FAST_KERNEL_CACHE.get(key)
    if fast is not None:
        return fast
    typed = _typed_api(api, dtype)  # raises NotImplementedError if missing
    from ._fast_dispatch import FastKernel  # local import = no circular
    fast = FastKernel(typed)
    _FAST_KERNEL_CACHE[key] = fast
    return fast


def _marshal_to_raw(args):
    """Inline tensor → data_ptr() conversion; passes scalars through.

    Replicates ``uc_kernel.runtime._marshal_args`` minus the
    ``kernel_sig.args[i].is_ptr`` lookup (which itself adds ~1 µs / call).
    All current ``run_kernel`` callers pass either ``torch.Tensor`` or
    int / float / bool scalars, so the ``hasattr`` check is sound.
    """
    return [a.data_ptr() if hasattr(a, "data_ptr") else a for a in args]


def run_unary_kernel(api: str, x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.empty_like(x)

    kernel_input = x.contiguous()
    kernel_output = torch.empty_like(kernel_input)
    rows, cols = _matrix_shape(kernel_input)
    fast = _get_or_create_fast(api, kernel_input.dtype)
    fast(kernel_input.data_ptr(), kernel_output.data_ptr(), rows, cols)
    return kernel_output.reshape(x.shape)


def run_binary_kernel(api: str, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.shape != rhs.shape:
        raise ValueError(f"UC backend {api} expects matching input shapes, got {lhs.shape} and {rhs.shape}.")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"UC backend {api} expects matching input dtypes, got {lhs.dtype} and {rhs.dtype}.")
    if lhs.numel() == 0:
        return torch.empty_like(lhs)

    kernel_lhs = lhs.contiguous()
    kernel_rhs = rhs.contiguous()
    kernel_output = torch.empty_like(kernel_lhs)
    rows, cols = _matrix_shape(kernel_lhs)
    fast = _get_or_create_fast(api, kernel_lhs.dtype)
    fast(
        kernel_lhs.data_ptr(),
        kernel_rhs.data_ptr(),
        kernel_output.data_ptr(),
        rows,
        cols,
    )
    return kernel_output.reshape(lhs.shape)


def run_kernel(api: str, dtype: torch.dtype, *args) -> None:
    fast = _get_or_create_fast(api, dtype)
    fast(*_marshal_to_raw(args))
