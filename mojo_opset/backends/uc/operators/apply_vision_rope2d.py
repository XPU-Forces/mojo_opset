"""UC backend for ``MojoApplyVisionRoPE2D`` — packed-token vision 2D RoPE.

Op contract (see ``mojo_opset.core.operators.position_embedding.MojoApplyVisionRoPE2D``)::

    q, k : [T, N, D]   (packed token-first, full-rope on head_dim)
    cos, sin : [T, D]  (per-token RoPE base; built as cat([freqs, freqs], -1))
    output = (x * cos) + (rotate_half(x) * sin)
    rotate_half(x) = cat([-x[..., D/2:], x[..., :D/2]], -1)

Fast path delivered by the UC backend:
  * dtype = bfloat16, D = 64 (vision config ``vision_448_27l_20h_h64``);
  * cos / sin fp32 of shape [T, D] (costoken mode);
  * arbitrary QH / KH (the per-head 1D-fragment kernel imposes no
    multiple-of-N head constraint).

All other shapes / dtypes fall back to ``super().forward`` (the torch
native reference in the parent class).
"""

from typing import Tuple

import torch

from mojo_opset.core import MojoApplyVisionRoPE2D

from ._utils import _uc_kernels, run_kernel


_APPLY_VISION_ROPE_KERNEL = "mojo_apply_vision_rope_tnh_d64_r64_costoken_bf16"
_SUPPORTED_HEAD_DIM = 64


class UCApplyVisionRoPE2D(MojoApplyVisionRoPE2D):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ----- input contract guards (mirror parent) -----
        if q.ndim != 3 or k.ndim != 3 or cos.ndim != 2 or sin.ndim != 2:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if q.dtype is not k.dtype:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if cos.shape != sin.shape:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if q.shape[0] != cos.shape[0] or k.shape[0] != cos.shape[0]:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if q.shape[-1] != cos.shape[-1] or k.shape[-1] != cos.shape[-1]:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ----- empty fast path -----
        if q.numel() == 0 or k.numel() == 0:
            return torch.empty_like(q), torch.empty_like(k)

        # ----- fast-path eligibility -----
        if q.dtype is not torch.bfloat16:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        head_dim = q.shape[-1]
        if head_dim != _SUPPORTED_HEAD_DIM:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # cos/sin must be f32 (kernel uses f32 trig table for accuracy).
        # ``MojoVisionRotaryEmbedding2D`` already emits f32; this guard
        # makes the wrapper robust to upstream dtype changes.
        if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        kernels = _uc_kernels()
        if _APPLY_VISION_ROPE_KERNEL not in kernels:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        rows, q_heads, _ = q.shape
        k_rows, k_heads, _ = k.shape
        if rows != k_rows:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        q_contig = q.contiguous()
        k_contig = k.contiguous()
        cos_contig = cos.contiguous()
        sin_contig = sin.contiguous()
        q_out = torch.empty_like(q_contig)
        k_out = torch.empty_like(k_contig)

        # Wheel ABI trailing scalar order = first-appearance of dynamic dim
        # names in tensor annotations: (M, QH, KH).
        run_kernel(
            "mojo_apply_vision_rope_tnh_d64_r64_costoken",
            q_contig.dtype,
            q_contig,
            k_contig,
            cos_contig,
            sin_contig,
            q_out,
            k_out,
            rows,
            q_heads,
            k_heads,
        )
        return q_out, k_out
