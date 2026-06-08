"""W7-B Phase 3: Prototype wrappers with FastKernel cache pattern.

Side-by-side variants of UCResidualAddRMSNorm and UCStaticQuant that:
  1. Cache the resolved kernel API + FastKernel handle per-dtype on the
     instance (avoids ``_resolve_api()`` + ``_uc_kernels()[api]`` lookup
     every forward).
  2. Pre-cast weight to kernel dtype + contiguous() once per dtype, stored
     on the instance (avoids ``_cast_param()`` allocation if param dtype
     differs from kernel dtype).
  3. Bypass ``uc_kernel.runtime.KernelFunction.__call__`` slow path
     (init_workspace + stream_ptr ~46 µs/call) via ``FastKernel`` which
     binds workspace + stream once.

These are *prototypes* and use the ``Analysis*`` class-name prefix so they
are NOT registered as a backend (the backend_registry special-cases the
``analysis`` prefix). Production wrappers in ``normalization.py`` /
``quant.py`` remain untouched. Once we validate accuracy + perf, the
pattern can be folded back into the production wrappers.

Accuracy contract: pre-cast weight is re-computed if ``self.weight.dtype``
or ``data_ptr()`` changes between calls (covers
``module.to(dtype=...)``/load_state_dict swap).
"""

from __future__ import annotations

import torch

from ._fast_dispatch import FastKernel, _DtypeKernelCache
from ._utils import _matrix_shape, _uc_kernels
from .normalization import UCResidualAddRMSNorm
from .quant import UCStaticQuant


# ===========================================================================
# AnalysisResidualAddRMSNormFast — fast variant
# ===========================================================================

_RES_RMS_API = {
    torch.bfloat16: "mojo_residual_add_rmsnorm_bf16",
    torch.float16: "mojo_residual_add_rmsnorm_fp16",
}


def _resolve_api_fast(api_map: dict, dtype: torch.dtype):
    """Same semantics as ``normalization._resolve_api`` but uses
    ``kernel in kernels.keys()`` since the system wheel may not expose
    ``__contains__`` on KernelRegistry (P-Wave-6 A2 patch not yet
    deployed)."""
    api = api_map.get(dtype)
    if api is None:
        return None
    if api not in _uc_kernels().keys():
        return None
    return api


class AnalysisResidualAddRMSNormFast(UCResidualAddRMSNorm):
    """Fast-dispatch variant of UCResidualAddRMSNorm.

    Caches per-dtype: (FastKernel, pre-cast weight tensor + ``data_ptr`` /
    dtype tag for invalidation).
    """

    supported_platforms_list = ["npu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: _DtypeKernelCache = _DtypeKernelCache()
        # Track weight identity so cache is invalidated if weight is swapped.
        self._last_weight_dataptr: int = 0
        self._last_weight_dtype: torch.dtype | None = None

    def _ensure_cache(self, dtype: torch.dtype):
        """Return (fast_kernel, weight_cached, weight_ptr) or (None,None,None)."""
        weight = self.weight
        weight_changed = (
            weight.data_ptr() != self._last_weight_dataptr
            or weight.dtype != self._last_weight_dtype
        )
        entry = self._cache.get(dtype)
        if entry is not None and not weight_changed:
            return entry

        api = _resolve_api_fast(_RES_RMS_API, dtype)
        if api is None:
            return (None, None, None)

        try:
            fast = FastKernel(api)
        except KeyError:
            return (None, None, None)

        # Pre-cast weight to kernel dtype and make contiguous.
        if weight.dtype != dtype:
            weight_cached = weight.detach().to(dtype).contiguous()
        else:
            weight_cached = weight.detach().contiguous()
        weight_ptr = weight_cached.data_ptr()

        entry = (fast, weight_cached, weight_ptr)
        self._cache.set(dtype, entry)
        self._last_weight_dataptr = weight.data_ptr()
        self._last_weight_dtype = weight.dtype
        return entry

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if residual is None:
            raise ValueError("UC backend MojoResidualAddRMSNorm requires residual.")
        if hidden_state.shape != residual.shape:
            raise ValueError(
                f"UC backend MojoResidualAddRMSNorm expects matching shapes, "
                f"got {hidden_state.shape} and {residual.shape}."
            )
        if hidden_state.dtype != residual.dtype:
            raise ValueError(
                f"UC backend MojoResidualAddRMSNorm expects matching dtypes, "
                f"got {hidden_state.dtype} and {residual.dtype}."
            )

        dtype = hidden_state.dtype
        fast, weight_cached, weight_ptr = self._ensure_cache(dtype)
        if fast is None:
            return super().forward(hidden_state, residual)

        if hidden_state.numel() == 0:
            empty = torch.empty_like(hidden_state)
            return empty, empty

        kernel_input = hidden_state.contiguous()
        kernel_residual = residual.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_output = torch.empty_like(kernel_input)
        kernel_residual_output = torch.empty_like(kernel_input)
        eps = float(self.variance_epsilon)

        fast(
            kernel_input.data_ptr(),
            kernel_residual.data_ptr(),
            weight_ptr,
            kernel_output.data_ptr(),
            kernel_residual_output.data_ptr(),
            rows,
            cols,
            eps,
        )

        output = kernel_output.reshape(hidden_state.shape)
        updated_residual = kernel_residual_output.reshape(hidden_state.shape)
        if self.norm_pos == "pre":
            return output, updated_residual
        return output, output


# ===========================================================================
# AnalysisStaticQuantFast — fast variant
# ===========================================================================

_STATIC_QUANT_API = {
    torch.bfloat16: "mojo_static_quant_bf16",
    torch.float16: "mojo_static_quant_fp16",
}


class AnalysisStaticQuantFast(UCStaticQuant):
    """Fast-dispatch variant of UCStaticQuant.

    Caches:
      * FastKernel handle per input dtype.
      * Pre-cast scale tensor (device-matched, fp32, contiguous) cached on
        instance — avoids the ``self.scale.to(device=..., dtype=fp32)``
        round-trip per call. Cache invalidates if ``self.scale.data_ptr()``
        changes.
    """

    supported_platforms_list = ["npu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: _DtypeKernelCache = _DtypeKernelCache()
        self._last_scale_dataptr: int = 0
        self._scale_cached: torch.Tensor | None = None
        self._scale_cached_ptr: int = 0

    def _ensure_cache(self, dtype: torch.dtype, target_device: torch.device):
        entry = self._cache.get(dtype)
        if entry is None:
            api = _STATIC_QUANT_API.get(dtype)
            if api is None or api not in _uc_kernels().keys():
                return None
            try:
                fast = FastKernel(api)
            except KeyError:
                return None
            entry = fast
            self._cache.set(dtype, entry)

        # Scale cache (independent of kernel dtype — scale is always fp32).
        scale = self.scale
        if (
            scale.data_ptr() != self._last_scale_dataptr
            or self._scale_cached is None
            or self._scale_cached.device != target_device
        ):
            self._scale_cached = scale.to(device=target_device, dtype=torch.float32).contiguous()
            self._scale_cached_ptr = self._scale_cached.data_ptr()
            self._last_scale_dataptr = scale.data_ptr()

        return entry

    def forward(self, input: torch.Tensor):
        if self.quant_dtype != torch.int8:
            raise NotImplementedError(f"UCStaticQuantFast only supports torch.int8, got {self.quant_dtype}.")
        if input.dim() < len(self.input_size):
            raise ValueError(
                f"input must have at least {len(self.input_size)} dims for scale shape "
                f"{self.input_size}, got {tuple(input.shape)}."
            )
        if tuple(input.shape[-len(self.input_size):]) != self.input_size:
            raise ValueError(
                f"input trailing dims {tuple(input.shape[-len(self.input_size):])} must "
                f"match scale shape {self.input_size}."
            )

        fast = self._ensure_cache(input.dtype, input.device)
        if fast is None:
            return super().forward(input)

        if input.numel() == 0:
            return torch.empty_like(input, dtype=self.quant_dtype), self.scale

        kernel_input = input.contiguous()
        cols = self._scale_cached.numel()
        rows = kernel_input.numel() // cols
        kernel_input_2d = kernel_input.reshape(rows, cols)
        kernel_output = torch.empty_like(kernel_input_2d, dtype=self.quant_dtype)

        fast(
            kernel_input_2d.data_ptr(),
            self._scale_cached_ptr,
            kernel_output.data_ptr(),
            rows,
            cols,
        )
        return kernel_output.reshape(input.shape), self.scale
