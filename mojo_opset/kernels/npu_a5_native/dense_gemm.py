"""Lazy A5 native wrappers for the migrated DeepGEMM dense kernels."""
from functools import lru_cache
from typing import Optional

import torch

from mojo_opset.kernels._dense_gemm import logical_shape


@lru_cache(maxsize=1)
def _launcher():
    import torch_npu  # noqa: F401
    try:
        from mojo_opset_lib.npu_a5 import _native
        launch = _native.dense_gemm
        return launch
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Build/install the matching A5 native library: bash native/build.sh npu_a5 dense_gemm; "
            "MOJO_NATIVE_PROVIDER=npu_a5 python -m pip install --no-build-isolation --no-deps -e native"
        ) from error


@torch.library.custom_op("mojo_npu_native_a5::dense_gemm", mutates_args=())
def _dense_gemm(input: torch.Tensor, weight: torch.Tensor,
                input_scale: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor],
                trans_input: bool, trans_weight: bool, output_dtype: torch.dtype,
                c: Optional[torch.Tensor], alpha: Optional[float], gran_m: int, gran_n: int,
                compiled_dims: str) -> torch.Tensor:
    return _launcher()(input, weight, input_scale, weight_scale, trans_input, trans_weight,
                       int(output_dtype == torch.float32), c, alpha, gran_m, gran_n, compiled_dims)


@_dense_gemm.register_fake
def _fake(input, weight, input_scale, weight_scale, trans_input, trans_weight,
          output_dtype, c, alpha, gran_m, gran_n, compiled_dims):
    a = input.t() if trans_input else input
    b = weight if trans_weight else weight.t()
    return input.new_empty((logical_shape(a)[0], logical_shape(b)[0]), dtype=output_dtype)


gemm_fwd = _dense_gemm
mxfp8_dequant_gemm_fwd = _dense_gemm
mxfp4_gemm_fwd = _dense_gemm
