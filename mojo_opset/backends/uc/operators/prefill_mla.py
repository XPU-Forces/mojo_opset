"""UC backend wrapper for :class:`MojoPrefillMLA`.

Strategy
--------

``MojoPrefillMLA`` is variable-length packed MLA prefill: a single ``query``
tensor of shape ``(T, H, qk_nope + qk_rope)`` (with ``T = sum(L_i)`` across the
batch) is fed together with packed ``compressed_kv``, ``k_pe`` and a cumulative
``cu_q_lens`` index. The reference implementation:

1. Dequantises KV via ``kv = compressed_kv @ self.kv_b_proj.T`` and splits into
   ``k_nope`` / ``v_all``.
2. Builds the full key ``k_all = cat([k_nope, k_pe.expand(-1, H, -1)], dim=-1)``.
3. Runs per-batch SDPA on the resulting ``(L_i, H, D_QK)`` slabs with a
   per-segment causal mask.

The UC backend keeps all variable-length plumbing (KV dequant, ``k_pe``
broadcast, causal-bias build) on the host and routes per-segment cube + vector
flash attention to ``mojo_prefill_mla_bf16``, which is fixed-shape:

* ``L = 512`` per segment, ``H = 16`` heads, ``D_QK = 192``, ``D_V = 128``.
* ``bf16`` Q/K/V/O, fp32 additive bias.
* Causal mask is materialised on host as ``(L, L)`` fp32 (``0`` on-/-under
  diagonal, ``-1e30`` above), ``@lru_cache``-ed across calls so that lifter
  v0.3's ban on ``T.if_then_else`` / comparisons does not bleed into the
  kernel.

Any call that does not match the hot-path contract (different dtypes, head
counts, head dims, non-causal, ``attn_sink``, custom ``softmax_scale``, packed
length not divisible into uniform ``L = 512`` segments, ``device != npu``) falls
back to ``super().forward()`` so behaviour stays identical to the reference.
"""

from functools import lru_cache
from typing import Optional

import torch

from mojo_opset.experimental.operators.attention import MojoPrefillMLA
from mojo_opset.utils.logging import get_logger

from ._utils import _uc_kernels


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hot-path fixed-shape contract — must match mojo_prefill_mla_bf16.py.
# ---------------------------------------------------------------------------
_KERNEL_API = "mojo_prefill_mla_bf16"
_FIXED_L = 512
_FIXED_H = 16
_FIXED_D_QK_NOPE = 128
_FIXED_D_QK_ROPE = 64
_FIXED_D_QK = _FIXED_D_QK_NOPE + _FIXED_D_QK_ROPE  # 192
_FIXED_D_V = 128
_NEG_INF_BIAS = -1.0e30


@lru_cache(maxsize=8)
def _causal_bias(length: int, device_str: str) -> torch.Tensor:
    """Build a single-segment causal additive bias ``(L, L)`` fp32.

    Entries on or below the diagonal are ``0.0``; entries strictly above are
    ``-1e30`` so that ``exp(bias)`` collapses them to zero after the online
    softmax. Cached by ``(L, device_str)`` so repeated batches reuse the buffer.
    """
    device = torch.device(device_str)
    # Build on CPU to keep the kernel deterministic across devices, then move.
    bias = torch.zeros(length, length, dtype=torch.float32)
    upper = torch.triu(
        torch.ones(length, length, dtype=torch.bool), diagonal=1
    )
    bias.masked_fill_(upper, _NEG_INF_BIAS)
    return bias.to(device=device, non_blocking=True).contiguous()


def _hot_path_eligible(
    op: "UCPrefillMLA",
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    cu_q_lens: torch.Tensor,
    softmax_scale: Optional[float],
) -> bool:
    """Strict contract for the fixed-shape kernel path.

    Any mismatch returns ``False`` so the caller routes to ``super().forward()``
    (i.e. the torch reference in :class:`MojoPrefillMLA`).
    """
    if op.use_attn_sink:
        return False
    if not op.is_causal:
        return False
    if op.num_heads != _FIXED_H:
        return False
    if op.qk_nope_head_dim != _FIXED_D_QK_NOPE:
        return False
    if op.qk_rope_head_dim != _FIXED_D_QK_ROPE:
        return False
    if op.v_head_dim != _FIXED_D_V:
        return False

    if query.dtype != torch.bfloat16:
        return False
    if compressed_kv.dtype != torch.bfloat16:
        return False
    if k_pe.dtype != torch.bfloat16:
        return False
    if op.kv_b_proj.dtype != torch.bfloat16:
        return False

    if query.dim() != 3 or compressed_kv.dim() != 2 or k_pe.dim() != 3:
        return False
    if k_pe.shape[1] != 1 or k_pe.shape[2] != _FIXED_D_QK_ROPE:
        return False
    if query.shape[1] != _FIXED_H or query.shape[2] != _FIXED_D_QK:
        return False
    if compressed_kv.shape[1] != op.kv_lora_rank:
        return False
    if compressed_kv.shape[0] != query.shape[0] or k_pe.shape[0] != query.shape[0]:
        return False

    if cu_q_lens.dim() != 1 or cu_q_lens.numel() < 2:
        return False

    # Only honour the default scale; custom scale would diverge from the kernel
    # constant ``SCALE = (1/D_QK)**0.5``.
    if softmax_scale is not None:
        default_scale = (1.0 / _FIXED_D_QK) ** 0.5
        if abs(float(softmax_scale) - default_scale) > 1e-6:
            return False

    # Kernel API must actually be present in the installed uc-kernel wheel.
    try:
        kernels = _uc_kernels()
        if _KERNEL_API not in kernels.keys():
            return False
    except Exception:
        return False

    return True


def _segment_lengths(cu_q_lens: torch.Tensor) -> list[int]:
    cu = cu_q_lens.detach().cpu().tolist()
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


class UCPrefillMLA(MojoPrefillMLA):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        cu_q_lens: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if not _hot_path_eligible(
            self, query, compressed_kv, k_pe, cu_q_lens, softmax_scale
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # All segments must be exactly L_FIXED for the static kernel; otherwise
        # we fall back. ``cu_q_lens`` is small and lives on the device, so the
        # host transfer here is negligible.
        seg_lens = _segment_lengths(cu_q_lens)
        if not seg_lens or any(s != _FIXED_L for s in seg_lens):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        total_tokens = query.shape[0]
        if total_tokens != sum(seg_lens):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if cu_q_lens.detach().cpu()[0].item() != 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        device = query.device
        # ---- Host KV dequant + k_all build ---------------------------------
        # kv = compressed_kv @ kv_b_proj.T -> (T, H, D_QK_NOPE + D_V)
        kv = (compressed_kv @ self.kv_b_proj.T).view(
            total_tokens, _FIXED_H, _FIXED_D_QK_NOPE + _FIXED_D_V
        )
        k_nope = kv[..., :_FIXED_D_QK_NOPE]
        v_all = kv[..., _FIXED_D_QK_NOPE:]
        # k_pe is (T, 1, D_ROPE); broadcast to (T, H, D_ROPE) and concat onto
        # k_nope so the kernel sees a single contiguous K head dim.
        k_all = torch.cat(
            [k_nope, k_pe.expand(-1, _FIXED_H, -1)], dim=-1
        )  # (T, H, D_QK)

        # ---- Per-segment static kernel calls -------------------------------
        kernels = _uc_kernels()
        kernel = kernels[_KERNEL_API]
        bias = _causal_bias(_FIXED_L, str(device))
        outputs = torch.empty(
            total_tokens, _FIXED_H, _FIXED_D_V, dtype=query.dtype, device=device
        )

        for batch_idx, _ in enumerate(seg_lens):
            s = batch_idx * _FIXED_L
            e = s + _FIXED_L
            # Permute (L, H, D) -> (H, L, D) so per-head rows are contiguous,
            # matching the kernel's input layout. ``.contiguous()`` forces an
            # actual copy because torch's permute returns a view with the same
            # storage but non-canonical strides.
            q_seg = query[s:e].permute(1, 0, 2).contiguous()
            k_seg = k_all[s:e].permute(1, 0, 2).contiguous()
            v_seg = v_all[s:e].permute(1, 0, 2).contiguous()
            out_seg = torch.empty(
                _FIXED_H, _FIXED_L, _FIXED_D_V, dtype=query.dtype, device=device
            )
            kernel(q_seg, k_seg, v_seg, bias, out_seg)
            outputs[s:e] = out_seg.permute(1, 0, 2).contiguous()

        return outputs
