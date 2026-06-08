import torch

from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm

from ._utils import _matrix_shape
from ._utils import _uc_kernels


# Deployed `mojo_*norm_{bf16,fp16}` wheel kernels accept:
#   * x / weight / bias / y / residual / residual_out as BFLOAT16 (or FLOAT16) ptrs,
#   * M, N as trailing INT32 scalars,
#   * eps as a trailing FLOAT32 *scalar* (NOT a tensor pointer).
#
# Earlier RCA in `docs/project-ops/perf-debug/bf16-cold-start.md` §2.2 quoted
# a stale `_manifest.json` snapshot that listed `eps` as a FLOAT32 *ptr*; the
# actually-deployed wheel uses a scalar eps (verified 2026-06-05 against the
# installed `_manifest.json` and the compiled `_kernels.so` binding signature).
# The arg ORDER in the previous version of this file was therefore correct;
# the real silent bug is that `self.weight` / `self.bias` were passed raw —
# usually as FLOAT32 — against a BFLOAT16-typed kernel pointer, so the kernel
# read garbage bytes and produced NaN-scale outputs.  See §C.2 of
# `docs/project-ops/lessons-learned.md`.
#
# In addition, the deployed wheel currently exposes only LayerNorm + Residual-
# AddLayerNorm (bf16/fp16); the RMSNorm variants live in the project's source
# `uc-kernel/manifest.json` but were not packaged into the installed wheel.
# We therefore look up APIs by key and fall back to `super().forward(...)`
# whenever the kernel is unavailable, so the wrappers behave correctly on any
# wheel build.

_LAYERNORM_API = {
    torch.bfloat16: "mojo_layernorm_bf16",
    torch.float16: "mojo_layernorm_fp16",
}

_RMSNORM_API = {
    torch.bfloat16: "mojo_rmsnorm_bf16",
    torch.float16: "mojo_rmsnorm_fp16",
}

# Shape-specialised RMSNorm variants (single-DRAM-pass).  Indexed by the
# trailing dim ``N``.  The wrapper falls back to the dynamic ``_RMSNORM_API``
# when no shape-specialised kernel matches.
_RMSNORM_API_SHAPE = {
    2048: {
        torch.bfloat16: "mojo_rmsnorm_n2048_bf16",
        torch.float16: "mojo_rmsnorm_n2048_fp16",
    },
}

_RESIDUAL_LN_API = {
    torch.bfloat16: "mojo_residual_add_layernorm_bf16",
    torch.float16: "mojo_residual_add_layernorm_fp16",
}

# Shape-specialised ResidualAddLayerNorm variants.  P-Wave-6 Track B1:
# N=2048 UB-resident apply pass + chunked variance/mean (P3-02 form A
# pattern + P2-01/P3-01 v6 n2048 schedule).  Wrapper falls back to the
# dynamic-N _RESIDUAL_LN_API when no shape match.
_RESIDUAL_LN_API_SHAPE = {
    2048: {
        torch.bfloat16: "mojo_residual_add_layernorm_n2048_bf16",
        torch.float16: "mojo_residual_add_layernorm_n2048_fp16",
    },
}

_RESIDUAL_RMS_API = {
    torch.bfloat16: "mojo_residual_add_rmsnorm_bf16",
    torch.float16: "mojo_residual_add_rmsnorm_fp16",
}


def _resolve_api(api_map: dict, dtype: torch.dtype) -> str:
    """Return the wheel API name for ``dtype`` or raise ``NotImplementedError``.

    Per project rule "wheel 没实现的就直接给报错" (2026-06-08), this helper
    raises on missing dtype or missing API instead of returning None — the
    previous "return None + caller fallback to ``super().forward``" pattern
    was a silent torch fallback that hides actual UC backend coverage.
    """
    api = api_map.get(dtype)
    if api is None:
        raise NotImplementedError(
            f"UC backend has no kernel registered for dtype {dtype}. "
            "Supported dtypes: " + ", ".join(str(d) for d in api_map.keys())
        )
    if api not in _uc_kernels():
        raise NotImplementedError(
            f"UC kernel {api!r} is not present in the loaded uc-kernel wheel. "
            "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
        )
    return api


def _cast_param(param: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast weight / bias to the kernel's native low-precision dtype if
    needed.  PyTorch's default for `nn.LayerNorm.weight` is fp32, but the
    bf16/fp16 wheel kernels read the buffer as the kernel-side native dtype;
    passing fp32 produces garbage / NaN outputs (silent miscompile)."""
    if param.dtype == dtype:
        return param.contiguous()
    return param.to(dtype).contiguous()


def _resolve_shape_or_dynamic(shape_map_or_none, dynamic_map, dtype: torch.dtype) -> str:
    """Pick the shape-specialised API when registered for ``dtype``; else fall
    back to the dynamic API.  Raises ``NotImplementedError`` only when the
    dynamic API is also missing (matches "wheel 没实现的就直接给报错")."""
    if shape_map_or_none is not None and dtype in shape_map_or_none:
        try:
            return _resolve_api(shape_map_or_none, dtype)
        except NotImplementedError:
            pass
    return _resolve_api(dynamic_map, dtype)


class UCRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        kernel_input = hidden_state.contiguous()
        rows, cols = _matrix_shape(kernel_input)

        # Prefer the shape-specialised kernel (single-DRAM-pass) when one
        # exists for the trailing dim; otherwise fall back to the dynamic-N
        # kernel.  Per "wheel 没实现的就直接给报错", if neither is registered
        # for the requested dtype, ``_resolve_api`` raises NotImplementedError.
        api = _resolve_shape_or_dynamic(_RMSNORM_API_SHAPE.get(cols), _RMSNORM_API, hidden_state.dtype)

        if hidden_state.numel() == 0:
            return torch.empty_like(hidden_state)

        kernel_output = torch.empty_like(kernel_input)
        weight = _cast_param(self.weight, kernel_input.dtype)
        eps = float(self.variance_epsilon)

        # wheel ABI: (x, weight, y, M, N, eps)
        _uc_kernels()[api](
            kernel_input,
            weight,
            kernel_output,
            rows,
            cols,
            eps,
        )
        return kernel_output.reshape(hidden_state.shape)


class UCResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu"]

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
        api = _resolve_api(_RESIDUAL_RMS_API, hidden_state.dtype)
        if hidden_state.numel() == 0:
            empty = torch.empty_like(hidden_state)
            return empty, empty

        kernel_input = hidden_state.contiguous()
        kernel_residual = residual.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_output = torch.empty_like(kernel_input)
        kernel_residual_output = torch.empty_like(kernel_input)
        weight = _cast_param(self.weight, kernel_input.dtype)
        eps = float(self.variance_epsilon)

        # wheel ABI: (x, residual, weight, y, residual_out, M, N, eps)
        _uc_kernels()[api](
            kernel_input,
            kernel_residual,
            weight,
            kernel_output,
            kernel_residual_output,
            rows,
            cols,
            eps,
        )

        output = kernel_output.reshape(hidden_state.shape)
        updated_residual = kernel_residual_output.reshape(hidden_state.shape)
        if self.norm_pos == "pre":
            return output, updated_residual
        return output, output


class UCResidualAddLayerNorm(MojoResidualAddLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if residual is None:
            raise ValueError("UC backend MojoResidualAddLayerNorm requires residual.")
        if hidden_state.shape != residual.shape:
            raise ValueError(
                f"UC backend MojoResidualAddLayerNorm expects matching shapes, "
                f"got {hidden_state.shape} and {residual.shape}."
            )
        if hidden_state.dtype != residual.dtype:
            raise ValueError(
                f"UC backend MojoResidualAddLayerNorm expects matching dtypes, "
                f"got {hidden_state.dtype} and {residual.dtype}."
            )
        if self.weight is None or self.bias is None:
            raise NotImplementedError("UC backend mojo_residual_add_layernorm requires weight and bias.")

        kernel_input = hidden_state.contiguous()
        kernel_residual = residual.contiguous()
        rows, cols = _matrix_shape(kernel_input)

        # Prefer the shape-specialised kernel (UB-resident apply) when one
        # exists for the trailing dim; otherwise use the dynamic-N kernel.
        # If neither is registered for the requested dtype, raise.
        api = _resolve_shape_or_dynamic(
            _RESIDUAL_LN_API_SHAPE.get(cols), _RESIDUAL_LN_API, hidden_state.dtype
        )

        if hidden_state.numel() == 0:
            empty = torch.empty_like(hidden_state)
            return empty, empty

        kernel_output = torch.empty_like(kernel_input)
        kernel_residual_output = torch.empty_like(kernel_input)
        weight = _cast_param(self.weight, kernel_input.dtype)
        bias = _cast_param(self.bias, kernel_input.dtype)
        eps = float(self.variance_epsilon)

        # wheel ABI: (x, residual, weight, bias, y, residual_out, M, N, eps)
        _uc_kernels()[api](
            kernel_input,
            kernel_residual,
            weight,
            bias,
            kernel_output,
            kernel_residual_output,
            rows,
            cols,
            eps,
        )

        output = kernel_output.reshape(hidden_state.shape)
        updated_residual = kernel_residual_output.reshape(hidden_state.shape)
        if self.norm_pos == "pre":
            return output, updated_residual
        return output, output


class UCLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            raise NotImplementedError("UC backend mojo_layernorm requires weight and bias.")
        if not self.elementwise_affine:
            raise NotImplementedError(
                "UC backend mojo_layernorm requires elementwise_affine=True; "
                "the wheel kernel always applies weight/bias."
            )
        api = _resolve_api(_LAYERNORM_API, hidden_state.dtype)
        if hidden_state.numel() == 0:
            return torch.empty_like(hidden_state)

        kernel_input = hidden_state.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_output = torch.empty_like(kernel_input)
        weight = _cast_param(self.weight, kernel_input.dtype)
        bias = _cast_param(self.bias, kernel_input.dtype)
        eps = float(self.variance_epsilon)

        # wheel ABI: (x, weight, bias, y, M, N, eps)
        _uc_kernels()[api](
            kernel_input,
            weight,
            bias,
            kernel_output,
            rows,
            cols,
            eps,
        )
        return kernel_output.reshape(hidden_state.shape)
