"""UC backend operator for MojoMRoPE (multimodal RoPE).

MojoMRoPE applies a 3D rotary embedding over the (T, H, W) axes for Qwen2-VL
family models. The 3-way axis selection on the ``cos_table`` /
``sin_table`` tensors (either non-interleaved concat-by-section or
interleaved gather-by-section) is performed on host because it is purely
indexing / slicing -- offloading it would add zero arithmetic per token but
significant kernel complexity.

After the host-side reduction, ``cos_table`` and ``sin_table`` are
``(num_tokens, half_rope_dim)`` and the on-device math is structurally
identical to the per-token (costoken) variant of ApplyRoPE.

The UC backend currently provides a single static kernel that matches the
dominant Qwen2-VL / Qwen2.5-VL / Qwen3-VL configuration:

    head_dim == rope_dim == 128, dtype == bfloat16

All other ``(head_dim, rope_dim, dtype)`` triples fall back to the torch
reference implementation in :class:`MojoMRoPE.forward`.
"""

from typing import List
from typing import Optional
from typing import Tuple

import functools
import os

import torch

from mojo_opset.core import MojoMRoPE
from mojo_opset.utils.logging import get_logger

from ._utils import run_kernel


logger = get_logger(__name__)


# Per-shape tile selection table for ``mojo_mrope_tnh_d128_r128_*_bf16``.
# Each entry maps a ``(n_qh % X == 0, n_kh % X == 0)`` membership to the
# specialised wheel API name + (XQ, XK) constants.  The wrapper picks the
# first variant whose tile divides both head counts (largest tile first).
#
# An env var ``UC_MROPE_FORCE_VARIANT`` (set to one of the suffix strings
# in the table -- e.g. ``"xq8_xk8_costoken"``) forces a specific variant for
# perf tile-sweep experiments.
_D128_R128_BF16_VARIANTS = (
    # (suffix, XQ, XK)
    ("xq14_xk4_costoken", 14, 4),  # Qwen2-VL-7B (QH=28, KH=4) - 2 Q iters
    ("xq20_xk8_costoken", 20, 8),  # Qwen2.5-VL-32B (QH=40, KH=8) - 2 Q iters
    ("xq16_xk8_costoken", 16, 8),  # Qwen3-VL-8B (QH=32, KH=8) - 2 Q iters
    ("xq8_xk8_costoken", 8, 8),    # Qwen3-VL-2B (QH=16, KH=8) - 2 Q iters / fallback
    ("costoken", 4, 4),            # universal fallback (XQ=XK=4)
)


def _select_mrope_variant(n_qh: int, n_kh: int) -> Optional[Tuple[str, int, int]]:
    forced = os.environ.get("UC_MROPE_FORCE_VARIANT", "").strip().lower()
    for suffix, xq, xk in _D128_R128_BF16_VARIANTS:
        if forced and suffix != forced:
            continue
        if n_qh % xq == 0 and n_kh % xk == 0:
            return suffix, xq, xk
    return None


@functools.lru_cache(maxsize=32)
def _interleaved_source_masks(
    half_rope_dim: int,
    sect_t: int,
    sect_h: int,
    sect_w: int,
    device_str: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute (mask_src1, mask_src2) for interleaved MRoPE.

    Mirrors ``MojoMRoPE._apply_interleaved_mrope``: position i where
    ``i % 3 == 1 and i < 3 * sect_h`` selects cos_table[1]; position i where
    ``i % 3 == 2 and i < 3 * sect_w`` selects cos_table[2]; otherwise
    cos_table[0]. Cached by section + half_rope_dim so we pay the host work
    only once per shape.
    """
    del sect_t  # unused; only sect_h / sect_w bound the index range.
    idx = torch.arange(half_rope_dim, device=device_str)
    mask_src1 = (idx % 3 == 1) & (idx < 3 * sect_h)
    mask_src2 = (idx % 3 == 2) & (idx < 3 * sect_w)
    return mask_src1.view(1, half_rope_dim), mask_src2.view(1, half_rope_dim)


@functools.lru_cache(maxsize=32)
def _split_offsets(sect_t: int, sect_h: int, sect_w: int) -> Tuple[int, int]:
    return sect_t, sect_t + sect_h


class UCMRoPE(MojoMRoPE):
    supported_platforms_list = ["npu"]

    # (head_dim, rope_dim, dtype) -> static kernel coverage.
    _STATIC_MROPE_KERNELS = frozenset(
        {
            (128, 128, torch.bfloat16),
        }
    )

    @staticmethod
    def _reduce_cos_sin_tables(
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collapse a 3D ``[3, S, half_rope_dim]`` cos / sin table to 2D.

        Mirrors the host-side branching inside :meth:`MojoMRoPE.forward` but
        uses index-free torch primitives (``torch.where`` for interleaved,
        ``torch.cat`` for non-interleaved) so the work compiles to a single
        broadcast / contig vec op instead of NPU stride-write scatter
        ops (which dominated overall latency at ~650 us in the slice-assign
        version -- see worker-reports/P2-14-mrope.md).
        """
        if cos_table.dim() != 3:
            return cos_table, sin_table

        sect_t, sect_h, sect_w = (int(s) for s in mrope_section)

        if is_interleaved:
            half_rope_dim = cos_table.shape[-1]
            mask_src1, mask_src2 = _interleaved_source_masks(
                half_rope_dim,
                sect_t,
                sect_h,
                sect_w,
                str(cos_table.device),
            )
            cos_out = torch.where(mask_src1, cos_table[1], cos_table[0])
            cos_out = torch.where(mask_src2, cos_table[2], cos_out)
            sin_out = torch.where(mask_src1, sin_table[1], sin_table[0])
            sin_out = torch.where(mask_src2, sin_table[2], sin_out)
            return cos_out, sin_out

        end_t, end_h = _split_offsets(sect_t, sect_h, sect_w)
        cos_out = torch.cat(
            [
                cos_table[0, :, :end_t],
                cos_table[1, :, end_t:end_h],
                cos_table[2, :, end_h:],
            ],
            dim=-1,
        )
        sin_out = torch.cat(
            [
                sin_table[0, :, :end_t],
                sin_table[1, :, end_t:end_h],
                sin_table[2, :, end_h:],
            ],
            dim=-1,
        )
        return cos_out, sin_out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        mrope_section: List[int],
        is_interleaved: bool = False,
        head_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- validation / fast-fallback fences (all 9 cover MojoMRoPE's
        # contract; failing any of them falls back to the torch reference).
        if query.dim() != 2 or key.dim() != 2:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if query.dtype != key.dtype:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if cos_table.shape != sin_table.shape:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if not isinstance(mrope_section, (list, tuple)) or len(mrope_section) != 3:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if query.numel() == 0 or key.numel() == 0:
            return torch.empty_like(query), torch.empty_like(key)

        rope_dim = int(sum(mrope_section)) * 2
        half_rope_dim = rope_dim // 2
        if head_dim is None:
            head_dim = rope_dim

        # The kernel currently only handles the fully-rotated case
        # (head_dim == rope_dim); partial-rotation with a non-empty pass
        # tail requires extra host copy logic, so we fall back.
        if head_dim != rope_dim:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        config_key = (head_dim, rope_dim, query.dtype)
        if config_key not in self._STATIC_MROPE_KERNELS:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        num_tokens, n_qh_head_dim = query.shape
        num_tokens_k, n_kh_head_dim = key.shape
        if num_tokens != num_tokens_k:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if n_qh_head_dim % head_dim != 0 or n_kh_head_dim % head_dim != 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        n_qh = n_qh_head_dim // head_dim
        n_kh = n_kh_head_dim // head_dim

        # Pick the largest head-pair tile that divides both head counts (or
        # fall back if none matches / variant overridden via env).
        variant = _select_mrope_variant(n_qh, n_kh)
        if variant is None:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        suffix, _xq, _xk = variant

        # ---- host-side cos/sin table reduction to (num_tokens, half_rope_dim) fp32.
        cos_reduced, sin_reduced = self._reduce_cos_sin_tables(
            cos_table, sin_table, list(mrope_section), is_interleaved
        )
        if cos_reduced.shape[-1] != half_rope_dim:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        cos_reduced = (
            cos_reduced.reshape(num_tokens, half_rope_dim).to(torch.float32).contiguous()
        )
        sin_reduced = (
            sin_reduced.reshape(num_tokens, half_rope_dim).to(torch.float32).contiguous()
        )

        # ---- reshape q/k to (M, H, D) for the kernel; outputs are allocated
        # in (M, H, D) layout and flattened back to (M, H*D) afterwards.
        q_in = query.reshape(num_tokens, n_qh, head_dim).contiguous()
        k_in = key.reshape(num_tokens, n_kh, head_dim).contiguous()

        q_out = torch.empty_like(q_in)
        k_out = torch.empty_like(k_in)

        api = f"mojo_mrope_tnh_d{head_dim}_r{rope_dim}_{suffix}"
        run_kernel(
            api,
            query.dtype,
            q_in,
            k_in,
            cos_reduced,
            sin_reduced,
            q_out,
            k_out,
            num_tokens,
            n_qh,
            n_kh,
        )

        return (
            q_out.reshape(num_tokens, n_qh_head_dim),
            k_out.reshape(num_tokens_k, n_kh_head_dim),
        )
