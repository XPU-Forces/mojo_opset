"""QK RMSNorm + half-split RoPE fusion (Ascend Triton).

Port of vLLM-Omni ``_rms_norm_rope_kernel``
(``vllm_omni/diffusion/layers/fused_qk_norm_rope.py``, main@717f8a3, lines 123-182), the
MiniMax-H3 contract: ``head_dim=128``, ``rotary_dim=96``, BF16, ``[tokens, heads, head_dim]``.

Numerical contract (must stay instruction-for-instruction identical to upstream)
---------------------------------------------------------------------------------
1. RMSNorm statistics are computed **once per head** and reused by both elements of every
   rotary pair, so ``(d, d + rotary_half)`` share one ``inv_rms``.
2. The normalized value is rounded to BF16 **before** the rotation, the rotation runs in
   FP32, and the result is rounded once when stored. Reordering any of these three steps
   changes the last bit and breaks the bit-exact cases.
3. The scale is applied left to right: ``x * inv_rms * weight``. Hoisting it to
   ``x * (inv_rms * weight)`` is *not* equivalent in FP32 and is therefore not done.
4. Dimensions ``>= rotary_dim`` pass through untouched: they are the BF16-rounded norm
   value and receive no second perturbation.

Ascend specific design
---------------------
* The kernel keeps every tile 1-D. Upstream launches a ``(8, head_dim)`` 2-D tile per
  ``(token, head-group)``; on triton-ascend 3.6.0 that shape costs ~11 us of fixed time per
  program (measured: 64-row tile, 235 MB case -> 40.3 ms = 11.2 us/program, 5.9 GB/s), the
  same signature as the 240x regression filed against the sibling operator
  (triton-lang/triton-ascend#2865). A 1-D row view is what that operator settled on too.
* One program therefore walks ``bt`` tokens and ``hpg`` heads, one 1-D row at a time, so the
  prologue (the weight loads and the ``pair_dims`` index chain) is amortised over
  ``bt * hpg`` rows (``rmsnorm.py`` uses the same schedule).
* Where the time goes, from the stage ablation on the 117 MB-per-stream case (cumulative, see
  ``_qk_norm_rope_kernel``'s docstring for the table): the partner re-read is 44% of the
  kernel and is the only stage with a plausible follow-up. Both halves are already inside
  ``x``, so a register-level swap would remove two loads per row; it is not done here because
  the only in-register shuffle available on 3.6.0 (``tl.gather`` in the index domain) is the
  53.2 ms form below, i.e. the same wall as #2865.
* ``head_dim`` is padded to the next power of two for ``tl.arange`` and the padded lanes are
  masked out of both the reduction and the store, so a non-power-of-two ``head_dim`` stays
  correct even though this task only requires 128. Only the MiniMax-H3 geometry has been
  measured; the other geometries are covered by the reference-path tests and listed as
  unverified in ``docs/vllm_ascend_qk_norm_rope.md``.
* Strided input/output are addressed through explicit strides; nothing calls
  ``.contiguous()`` on the activations, so a packed-QKV slice is consumed in place (task
  requirement 4). ``weight`` and ``rope_table`` are read with unit stride and are made
  contiguous by the wrapper.
* ``weight``, ``pair_weight``, ``pair_dims`` and ``freq_dims`` are loop-invariant and
  hoisted out of the row loop, which also keeps the ``tl.where`` index chains out of it.
  ``pair_dims`` is still an index-domain ``tl.where`` (i.e. a gather), which is why the
  partner itself is loaded with two constant shifts instead.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

__all__ = ["qk_norm_rope_infer_fwd"]


@libentry()
@triton.jit
def _qk_norm_rope_kernel(
    x_ptr,
    weight_ptr,
    rope_table_ptr,
    out_ptr,
    num_tokens,
    x_stride_t,
    x_stride_h,
    x_stride_d,
    rope_stride_t,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_half: tl.constexpr,
    d_pad: tl.constexpr,
    eps: tl.constexpr,
    bt: tl.constexpr,
    hpg: tl.constexpr,
):
    """One ``(head_dim,)`` row at a time, ``bt * hpg`` rows per program.

    Every tile is 1-D on purpose. Measured on 3.6.0 for the 235 MB case, in the same round:

    | shape                                        | time     |
    |----------------------------------------------|----------|
    | 1-D row, table hoisted, affine partner       | 7.8 ms   |
    | 1-D row, table per head, partner by index    | 53.2 ms  |
    | 2-D ``(8, head_dim)`` tile                   | 40.3 ms  |
    | flat 2 KB tile + ``tl.reshape`` + 2-D reduce  | 676 ms   |

    so a 2-D tile is not merely slower here, it is a different regime (the sibling operator hit
    the same wall, triton-lang/triton-ascend#2865). The remaining 1-D cost is dominated by the
    256 B transfer size: a pure row-wise copy of this layout reaches only 157 GB/s on the same
    box, against 288 GB/s for the ``npu_rms_norm`` + ``npu_rotary_mul`` composition.

    Cumulative stage ablation, one stream of 4096x56 (117 MB), 3891 us = 17.0 ns/row
    (``results/t1_a2_20261029/stage.json``):

    | stage                       | delta      | ns/row |
    |-----------------------------|------------|--------|
    | copy (one load + one store) | 747.8 us   | 3.3    |
    | + reduce (load ``x``)       | 802.8 us   | 3.5    |
    | + norm (in-register)        | -0.1 us    | 0.0    |
    | + partner (two shifted loads)| 1714.6 us | 7.5    |
    | + rotation (in-register)    | 422.2 us   | 1.8    |
    | + pass-through tail in store| 204.0 us   | 0.9    |

    Four of the five memory operations per row are therefore loads, and the two partner loads
    alone are 44% of the kernel. Reducing that count is the only lever left, but it needs an
    in-register half-swap, which on 3.6.0 means ``tl.gather`` (53.2 ms, above) — the same
    regression as #2865.

    Both optimisations that did work are kept: the partner is a pair of constant-shift loads
    with the select moved to the value domain, and cos/sin are loaded once per token and
    reused by all ``hpg`` heads. ``eps`` stays a ``tl.constexpr`` and is specialised per value,
    the same choice upstream makes; the contract's eps is a fixed scalar, so this costs one
    compilation in practice.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    dims = tl.arange(0, d_pad)
    dim_mask = dims < head_dim
    weight = tl.load(weight_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
    rotary_dim: tl.constexpr = rotary_half * 2
    first_half = dims < rotary_half
    rotary_mask = dims < rotary_dim
    second_half = (dims >= rotary_half) & rotary_mask
    pair_dims = tl.where(
        first_half, dims + rotary_half, tl.where(second_half, dims - rotary_half, dims)
    )
    pair_weight = tl.load(weight_ptr + pair_dims, mask=dim_mask, other=0.0).to(tl.float32)

    for i in range(bt):
        token = pid_t * bt + i
        t_valid = token < num_tokens
        table_off = token.to(tl.int64) * rope_stride_t
        table_mask = dim_mask & t_valid
        tab = tl.load(
            rope_table_ptr + table_off + dims, mask=table_mask & rotary_mask, other=0.0
        ).to(tl.float32)
        tab_hi = tl.load(
            rope_table_ptr + table_off + rotary_half + dims,
            mask=table_mask & first_half,
            other=0.0,
        ).to(tl.float32)
        tab_lo = tl.load(
            rope_table_ptr + table_off - rotary_half + dims,
            mask=table_mask & second_half,
            other=0.0,
        ).to(tl.float32)
        cos = tl.where(first_half, tab, tab_lo)
        sin = tl.where(first_half, tab_hi, tab)

        for j in range(hpg):
            head = pid_h * hpg + j
            row_mask = dim_mask & t_valid & (head < num_heads)
            row_in = token.to(tl.int64) * x_stride_t + head * x_stride_h
            row_out = token.to(tl.int64) * out_stride_t + head * out_stride_h

            x = tl.load(x_ptr + row_in + dims * x_stride_d, mask=row_mask, other=0.0).to(
                tl.float32
            )
            inv_rms = tl.rsqrt(tl.sum(x * x) / head_dim + eps)
            normalized = (x * inv_rms * weight).to(tl.bfloat16)

            shift = rotary_half * x_stride_d
            pair_a = tl.load(
                x_ptr + row_in + shift + dims * x_stride_d,
                mask=row_mask & first_half,
                other=0.0,
            ).to(tl.float32)
            pair_b = tl.load(
                x_ptr + row_in - shift + dims * x_stride_d,
                mask=row_mask & second_half,
                other=0.0,
            ).to(tl.float32)
            pair_normalized = (
                tl.where(first_half, pair_a, pair_b) * inv_rms * pair_weight
            ).to(tl.bfloat16)

            normalized_f = normalized.to(tl.float32)
            pair_f = pair_normalized.to(tl.float32)
            first = normalized_f * cos - pair_f * sin
            second = normalized_f * cos + pair_f * sin
            output = tl.where(rotary_mask, tl.where(first_half, first, second), normalized_f)
            tl.store(out_ptr + row_out + dims * out_stride_d, output, mask=row_mask)


def _pick_tuning(num_tokens: int, num_heads: int, cores: int) -> Tuple[int, int, int, int]:
    """Return ``(bt, hpg, grid_t, grid_h)``.

    ``hpg`` sizes the head group of the *grid*, not of a tile: this kernel still loads one
    1-D ``(head_dim,)`` row at a time, so ``hpg = 8`` means "reuse one cos/sin pair across
    eight rows and give ``grid_h`` the same 8-wide grouping upstream uses" — 56 heads split
    into 7 groups, 57 into 8 with a one-head tail. It does *not* build the 2 KB
    ``(8, head_dim)`` contiguous span, because that tile is the 40.3 ms form measured above.
    ``bt`` then absorbs the token axis so the prologue is amortised over ``bt * hpg`` rows.
    """
    hpg = 8 if num_heads >= 8 else 1
    grid_h = (num_heads + hpg - 1) // hpg
    # bt is capped at 16: sweeping 4/16/64 on the 235 MB case gave 8032/7806/8212 us, i.e. a
    # 5% band, so the schedule is not sensitive to it and a small cap keeps the constexpr tile
    # shared across neighbouring shapes instead of recompiling per token count. The cap also
    # wins over the cores * 4 target for long sequences, so the flat grid ends up well above
    # one wave (4096 x 56 -> 256 x 7 = 1792 programs on 40 vector cores); the 5% band says
    # that costs nothing, and dropping the cap was not faster.
    target = max((cores * 4) // max(grid_h, 1), 1)
    bt = max(1, min(-(-num_tokens // target), 16, num_tokens))
    return bt, hpg, (num_tokens + bt - 1) // bt, grid_h


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    bt: int,
    hpg: int,
    grid_t: int,
    grid_h: int,
) -> None:
    num_tokens, num_heads, head_dim = x.shape
    rotary_half = rope_table.shape[-1] // 2
    _qk_norm_rope_kernel[(grid_t, grid_h)](
        x,
        weight,
        rope_table,
        out,
        num_tokens,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rope_table.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads=num_heads,
        head_dim=head_dim,
        rotary_half=rotary_half,
        d_pad=triton.next_power_of_2(head_dim),
        eps=eps,
        bt=bt,
        hpg=hpg,
    )


def _norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    num_tokens, num_heads, _ = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if num_tokens == 0 or num_heads == 0:
        return out
    bt, hpg, grid_t, grid_h = _pick_tuning(num_tokens, num_heads, get_num_cores("vector"))
    _launch(x, weight, rope_table, eps, out, bt, hpg, grid_t, grid_h)
    return out


def qk_norm_rope_infer_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q and K each get one launch, matching the upstream kernel-count contract."""
    return (
        _norm_rope(q, q_weight, rope_table, eps),
        _norm_rope(k, k_weight, rope_table, eps),
    )
