"""UC backend wrappers for dense / grouped / quantised matmul.

Status per kernel (2026-06-08, see ``docs/project-ops/uc-active-apis-2026-06-08.json``
and ``docs/project-ops/uc-kernel-fail-todo-2026-06-08.md``):

* ``UCGemm``       — DISABLED. The wheel no longer ships ``mojo_gemm_bf16``
  (dropped from manifest); per project rule "wheel 没实现的就直接给报错"
  this wrapper now raises ``NotImplementedError`` instead of silently
  falling back to ``torch.nn.functional.linear``.
* ``UCGroupGemm``  — DISABLED for the same reason (the host loop relies on
  ``mojo_gemm_bf16``).
* ``UCQuantGemm``  — ACTIVE. Calls ``mojo_quant_gemm_{bf16,fp16,fp32}``
  (all 3 dtype variants present in the 87-API active wheel).

Wheel ABI notes (lessons-learned §B.1)
--------------------------------------
``mojo_quant_gemm`` is dynamic (``M, N, K``); its empirically validated
trailing scalar order is ``(M, K, N)``.

It assumes ``F.linear``-style layout: weight is logical ``(N, K)``
and consumed with ``transpose_B`` inside the kernel.
"""

import torch

from mojo_opset.core import MojoGemm
from mojo_opset.core import MojoGroupGemm
from mojo_opset.core import MojoQuantGemm

from ._utils import _uc_kernels


_OUTPUT_DTYPE_SUFFIX = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


def _require_kernel(api: str):
    kernels = _uc_kernels()
    if api not in kernels:
        raise NotImplementedError(
            f"UC kernel {api!r} is not in the loaded uc-kernel wheel. "
            "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
        )
    return kernels[api]


_DISABLED_MSG_GEMM = (
    "UCGemm is disabled: ``mojo_gemm_bf16`` is not in the current uc-kernel "
    "wheel (manifest dropped this entry; only ``mojo_quant_gemm_*`` ships). "
    "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md. Per project rule "
    "'wheel 没实现的就直接给报错' (2026-06-08), this wrapper does not silently "
    "fall back to torch — use TTX / torch_npu / torch_native backend instead."
)

_DISABLED_MSG_GROUP_GEMM = (
    "UCGroupGemm is disabled: depends on ``mojo_gemm_bf16`` which is not in "
    "the current uc-kernel wheel. "
    "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md. Per project rule "
    "'wheel 没实现的就直接给报错' (2026-06-08), this wrapper does not silently "
    "fall back to torch — use TTX / torch_npu / torch_native backend instead."
)


class UCGemm(MojoGemm):
    """DISABLED — dense GEMM kernel removed from wheel manifest."""

    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(_DISABLED_MSG_GEMM)


class UCGroupGemm(MojoGroupGemm):
    """DISABLED — depends on missing ``mojo_gemm_bf16``."""

    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(_DISABLED_MSG_GROUP_GEMM)


class UCQuantGemm(MojoQuantGemm):
    """Int8 + dequant fused matmul via ``mojo_quant_gemm_{bf16,fp16,fp32}``."""

    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if input.dtype != torch.int8:
            raise NotImplementedError(f"UC QuantGemm supports int8 input, got {input.dtype}.")
        if self.trans_weight:
            weight = self.weight.t().contiguous()
        else:
            weight = self.weight
        input_scale = input_scale.flatten().float().contiguous()
        weight_scale = self.weight_scale.flatten().float().contiguous()
        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        M, K = input.shape
        K_w, N = weight.shape
        if K_w != K:
            raise ValueError(f"input K {K} must match weight K {K_w}.")
        if input_scale.numel() != M:
            raise ValueError(f"input_scale length {input_scale.numel()} must equal M {M}.")
        if weight_scale.numel() != N:
            raise ValueError(f"weight_scale length {weight_scale.numel()} must equal N {N}.")

        output = torch.empty((M, N), device=input.device, dtype=self.output_dtype)
        if output.numel() == 0:
            return output
        suffix = _OUTPUT_DTYPE_SUFFIX.get(self.output_dtype)
        if suffix is None:
            raise NotImplementedError(f"UC QuantGemm does not support output dtype {self.output_dtype}.")

        kernel = _require_kernel(f"mojo_quant_gemm_{suffix}")
        kernel(input, weight, input_scale, weight_scale, output, M, K, N)
        return output
