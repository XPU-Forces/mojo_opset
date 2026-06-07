"""W7-B Phase 2-3: Cache-based fast dispatch helper for UC wrappers.

Profiling (Section D of ``w7b_profile.py``) showed that
``uc_kernel.runtime.KernelFunction.__call__`` spends ~69 µs / call:
  * ~38 µs in ``init_workspace()`` — the "no-op" fast path still re-parses
    ``_manifest.json`` via ``_load_package_manifest()`` every call to compute
    ``_max_workspace_size()``.
  * ~8.5 µs in ``_stream_ptr(None)`` — torch_npu API query per call.
  * ~2.6 µs in the actual ``extension.<api>(*ptrs, stream)`` call.

This helper exposes a ``FastKernel`` wrapper that pays the workspace bind +
stream query exactly ONCE (per ``(api, dtype)`` cache key), then issues all
subsequent calls via the cached ``extension.<api>`` callable + cached stream
pointer.

Side effects / assumptions:
  * Workspace tensor is bound once per FastKernel instance. If the user
    swaps to a different workspace mid-program, behaviour diverges. This is
    the same staleness model as ``_uc_kernels()`` (lru_cache(maxsize=1)).
  * NPU stream is cached. If the user switches to a non-default stream, the
    cache becomes stale. We expose ``invalidate_stream()`` for that case.
    For typical inference workloads (single default stream) the cache is
    safe.
  * The cached extension callable is bound to the underlying C extension —
    valid as long as the wheel isn't reloaded.

Constraint compliance (W7-B task §硬约束):
  * Pure mojo_opset wrapper-side optimization; no edit to
    ``uc_kernel/runtime.py``, no edit to compiled kernels, no compiler
    changes.
  * Wrapper's existing ``super().forward(...)`` fallback path is preserved
    by callers (FastKernel is opt-in per-op).
  * Accuracy unchanged — we still call the same C extension entry with the
    same data pointers; we only skip the redundant per-call setup.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from ._utils import _uc_kernels


def _bind_workspace_once(kernel_workspace_size: int) -> None:
    """Bind the global UC workspace exactly once.

    ``uc_kernel.runtime.init_workspace`` is itself idempotent on the "no-op"
    path but still walks the manifest each call. We call it exactly once per
    FastKernel cache miss.
    """
    from uc_kernel.runtime import init_workspace

    init_workspace(required_nbytes=kernel_workspace_size)


def _current_stream_ptr() -> int:
    """Resolve the current NPU stream pointer.

    Equivalent to ``uc_kernel.runtime._stream_ptr(None)`` but called only on
    cache miss / explicit invalidation, not per call.
    """
    import torch_npu

    return torch_npu.npu.current_stream().npu_stream


class FastKernel:
    """A cached, low-overhead callable for a single UC kernel.

    The first ``__init__`` resolves the kernel API, binds the workspace, and
    caches the bound extension entry + stream pointer. Subsequent calls
    invoke the extension directly with pre-cached state, skipping the
    per-call ``init_workspace()`` walk and ``_stream_ptr`` query.
    """

    __slots__ = ("api", "_ext_call", "_stream_ptr", "_workspace_bound")

    def __init__(self, api: str):
        kernels = _uc_kernels()
        kfn = kernels[api]  # raises KeyError if missing
        _bind_workspace_once(kfn.workspace_size)
        self.api = api
        self._ext_call = getattr(kfn.extension, kfn.api)
        self._stream_ptr = _current_stream_ptr()
        self._workspace_bound = True

    def invalidate_stream(self) -> None:
        """Re-query the current NPU stream. Call after explicit stream switch."""
        self._stream_ptr = _current_stream_ptr()

    def __call__(self, *raw_args) -> None:
        """Invoke the extension with pre-marshalled raw args + cached stream.

        Callers MUST pass ``int`` (data_ptr or scalar) for every arg in the
        kernel signature order. No type validation is performed — this is
        the hot path.
        """
        self._ext_call(*raw_args, self._stream_ptr)


# ---------------------------------------------------------------------------
# Per-instance cache helpers (used by wrapper subclasses)
# ---------------------------------------------------------------------------

class _DtypeKernelCache:
    """Maps ``torch.dtype`` → (FastKernel, optional cast-param tensors).

    Stored as an instance attribute on each UC operator. Survives many
    ``forward()`` calls; invalidated only on first miss for a new dtype.
    """

    __slots__ = ("_by_dtype",)

    def __init__(self):
        self._by_dtype: dict[torch.dtype, tuple] = {}

    def get(self, dtype: torch.dtype):
        return self._by_dtype.get(dtype)

    def set(self, dtype: torch.dtype, value: tuple) -> None:
        self._by_dtype[dtype] = value

    def clear(self) -> None:
        self._by_dtype.clear()
