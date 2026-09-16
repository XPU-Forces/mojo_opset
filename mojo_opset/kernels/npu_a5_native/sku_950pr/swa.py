from functools import lru_cache
from typing import Optional

import torch


@lru_cache(maxsize=1)
def _launcher():
    import torch_npu  # noqa: F401 -- load dependent runtime libraries first

    try:
        from mojo_opset_lib.npu_a5.sku_950pr import _native

        return _native.launch_swa
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Install byted-mojo-opset-lib-npu-a5-950pr matching your runtime, or build locally: "
            "bash native/build.sh npu_a5/sku_950pr swa; then "
            "MOJO_LIB_PROVIDER=npu_a5/sku_950pr python -m pip install --no-build-isolation --no-deps -e ./native"
        ) from error


@torch.library.custom_op("mojo_npu_native_a5_950pr::native_swa_infer", mutates_args=())
def native_swa_infer_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    is_causal: bool,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
    softmax_scale: float,
    gqa_interleave: bool,
) -> torch.Tensor:
    if local_window_size is None or global_window_size is None:
        raise ValueError("AscendC SWA requires explicit local and global window sizes")
    return _launcher()(
        q,
        k,
        v,
        cu_q_lens,
        cu_k_lens,
        is_causal,
        local_window_size,
        global_window_size,
        softmax_scale,
        int(gqa_interleave),
        0,
    )


@native_swa_infer_fwd.register_fake
def _native_swa_infer_fake(
    q, k, v, cu_q_lens, cu_k_lens, is_causal, local_window_size, global_window_size, softmax_scale, gqa_interleave
):
    return torch.empty_like(q)
