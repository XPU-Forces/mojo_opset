from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _launcher():
    import torch_npu  # noqa: F401 -- load dependent runtime libraries first
    try:
        from mojo_opset_lib.npu_a2 import _native

        return _native.launch_varlen_fa
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Install byted-mojo-opset-lib-npu-a2 matching your runtime, or build locally: "
            "bash native/build.sh npu_a2 varlen_fa; then "
            "MOJO_LIB_PROVIDER=npu_a2 python -m pip install --no-build-isolation --no-deps -e ./native"
        ) from error


@torch.library.custom_op("mojo_npu_native_a2::varlen_fa_infer", mutates_args=())
def varlen_fa_infer_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_q_lens: torch.Tensor, cu_k_lens: torch.Tensor,
    is_causal: bool, softmax_scale: float, gqa_interleave: bool,
) -> torch.Tensor:
    return _launcher()(
        q, k, v, cu_q_lens, cu_k_lens, is_causal, softmax_scale, int(gqa_interleave), 0,
    )


@varlen_fa_infer_fwd.register_fake
def _fake(q, k, v, cu_q_lens, cu_k_lens, is_causal, softmax_scale, gqa_interleave):
    return torch.empty_like(q)
