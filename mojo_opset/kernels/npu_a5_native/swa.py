from functools import lru_cache
import math
from typing import Optional
from typing import Tuple

import torch

from mojo_opset.utils.target import detect_target

from ._extension import load_ops


def _ops(symbols=("launch_swa",)):
    try:
        return load_ops("npu_a5", "mojo_native_a5", symbols)
    except ImportError as shared_error:
        target = detect_target()
        if target.platform == "npu" and target.sku != "950pr":
            raise ImportError(
                f"Native shared A5 SWA extension is unavailable for {target}: "
                f"{shared_error}. Build and install bash build.sh npu_a5; "
                "the public 950PR provider cannot serve this runtime."
            ) from shared_error
        try:
            return load_ops("npu_a5/sku_950pr", "mojo_native_a5_950pr", symbols)
        except ImportError as sku_error:
            raise ImportError(
                "Native A5 SWA extension could not be loaded from either "
                "mojo_opset_lib.npu_a5._native or "
                "mojo_opset_lib.npu_a5.sku_950pr._native. Build and install "
                "the provider for your runtime: bash build.sh npu_a5 "
                f"or bash build.sh npu_a5/sku_950pr. Shared import failed: {shared_error}"
            ) from sku_error


@lru_cache(maxsize=1)
def _launcher():
    return _ops().launch_swa.default


@lru_cache(maxsize=1)
def _training_launchers():
    ops = _ops(("launch_swa_training", "launch_swa_backward"))
    return ops.launch_swa_training.default, ops.launch_swa_backward.default


def _normalize_windows(
    k: torch.Tensor,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
) -> Tuple[int, int]:
    if local_window_size is not None and local_window_size < 0:
        raise ValueError("AscendC SWA requires a non-negative local_window_size")
    full_causal = local_window_size is None and global_window_size is None
    global_window_size = 0 if global_window_size is None else global_window_size
    if local_window_size is None:
        # The native launcher uses -1 for disabled, preserving explicit radius 0.
        local_window_size = k.shape[0] if full_causal else -1
    return local_window_size, global_window_size


def _validate_training_options(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
    softmax_scale: float,
    gqa_interleave: bool,
    output_f32: bool,
) -> None:
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("AscendC SWA training requires BF16 q, k, and v")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("AscendC SWA training requires contiguous q, k, and v")
    if q.shape[-1] != 128:
        raise ValueError("AscendC SWA training requires head_dim=128")
    if q.shape[1] <= 0 or k.shape[1] <= 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("AscendC SWA training requires Hq to be divisible by positive Hkv")
    if not is_causal:
        raise ValueError("AscendC SWA training supports causal attention only")
    if local_window_size is None or local_window_size < -1:
        raise ValueError("AscendC SWA training requires a local radius or disabled sentinel")
    if global_window_size is None or global_window_size < 0:
        raise ValueError("AscendC SWA training requires a non-negative global_window_size")
    if local_window_size < 0 and global_window_size == 0:
        raise ValueError("AscendC SWA training requires a local or global window")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("AscendC SWA training requires a finite positive softmax_scale")


@torch.library.custom_op("mojo_npu_native_a5::native_swa_infer", mutates_args=())
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


@torch.library.custom_op(
    "mojo_npu_native_a5::swa_fwd",
    mutates_args=(),
    schema="(Tensor q, Tensor k, Tensor v, Tensor cu_q_lens, Tensor cu_k_lens, "
           "bool is_causal, SymInt? local_window_size, SymInt? global_window_size, "
           "float softmax_scale, bool gqa_interleave, bool output_f32) -> (Tensor, Tensor, Tensor)",
)
def _swa_fwd_op(
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
    output_f32: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _training_launchers()[0](
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
        output_f32,
        0,
    )


@_swa_fwd_op.register_fake
def _swa_fwd_fake(
    q, k, v, cu_q_lens, cu_k_lens, is_causal, local_window_size, global_window_size,
    softmax_scale, gqa_interleave, output_f32
):
    output_cache = (
        q.new_empty(q.shape, dtype=torch.float32)
        if output_f32
        else q.new_empty((0,), dtype=torch.float32)
    )
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32), output_cache


def swa_fwd(
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
    output_f32: bool = False,
):
    local_window_size, global_window_size = _normalize_windows(
        k, local_window_size, global_window_size
    )
    _validate_training_options(
        q,
        k,
        v,
        is_causal,
        local_window_size,
        global_window_size,
        softmax_scale,
        gqa_interleave,
        output_f32,
    )
    output, softmax_lse, output_cache = _swa_fwd_op(
        q,
        k,
        v,
        cu_q_lens,
        cu_k_lens,
        is_causal,
        local_window_size,
        global_window_size,
        softmax_scale,
        gqa_interleave,
        output_f32,
    )
    return output, softmax_lse, output_cache if output_f32 else None


@torch.library.custom_op(
    "mojo_npu_native_a5::swa_bwd",
    mutates_args=(),
    schema="(Tensor grad_output, Tensor q, Tensor k, Tensor v, Tensor output, Tensor softmax_lse, "
           "Tensor cu_q_lens, Tensor cu_k_lens, SymInt local_window_size, bool gqa_interleave, "
           "SymInt global_window_size, float softmax_scale) -> (Tensor, Tensor, Tensor)",
)
def _swa_bwd_op(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    local_window_size: int,
    gqa_interleave: bool,
    global_window_size: int,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_q, grad_k, grad_v, _ = _training_launchers()[1](
        q,
        k,
        v,
        output,
        grad_output,
        softmax_lse,
        cu_q_lens,
        cu_k_lens,
        local_window_size,
        int(gqa_interleave),
        global_window_size,
        softmax_scale,
    )
    return grad_q, grad_k, grad_v


@_swa_bwd_op.register_fake
def _swa_bwd_fake(
    grad_output,
    q,
    k,
    v,
    output,
    softmax_lse,
    cu_q_lens,
    cu_k_lens,
    local_window_size,
    gqa_interleave,
    global_window_size,
    softmax_scale,
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def swa_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_k_lens: torch.Tensor,
    is_causal: bool,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
    softmax_scale: float,
    gqa_interleave: bool,
):
    local_window_size, global_window_size = _normalize_windows(
        k, local_window_size, global_window_size
    )
    _validate_training_options(
        q,
        k,
        v,
        is_causal,
        local_window_size,
        global_window_size,
        softmax_scale,
        gqa_interleave,
        False,
    )
    grad_q, grad_k, grad_v = _swa_bwd_op(
        grad_output,
        q,
        k,
        v,
        output,
        softmax_lse,
        cu_q_lens,
        cu_k_lens,
        local_window_size,
        gqa_interleave,
        global_window_size,
        softmax_scale,
    )
    return grad_q, grad_k, grad_v
