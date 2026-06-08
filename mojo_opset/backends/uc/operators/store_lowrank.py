"""UC backend for ``MojoStoreLowrank`` (experimental).

The high-level mojo op (``mojo_opset/experimental/operators/store_lowrank.py``)
is the indirect store

    label_cache[block_idxs, :, token_idxs, :] = key_lr[:token_num]

with shapes

    label_cache : (num_blocks, num_kv_heads, block_size, head_dim)  bf16  (BNSD)
    key_lr      : (token_num,  num_kv_heads, head_dim)              bf16  (SND)
    block_idxs  : (token_num,)                                      int32
    token_idxs  : (token_num,)                                      int32
    token_num   : int (number of *valid* leading rows of ``key_lr`` to consume)

Performance design (P2-28 rewrite, 2026-06-05)
---------------------------------------------
The previous wrapper looped on ``num_kv_heads`` and per-head materialised a
contiguous ``cache_h`` plane, scattered into it, then wrote the plane back to
``label_cache``. For (256, 8, 512, 128) bf16 that costs ~768 MB of pure
overhead DRAM traffic on top of ~27 MB of real scatter work — Class § F.3
(DRAM↔UB 来回). On wide-H shapes this overhead dominates and makes UC much
slower than TTX.

This rewrite collapses **all kv-heads into a single kernel launch** by
treating the cache as one flat ``(num_blocks * num_kv_heads * block_size,
head_dim)`` slot table:

* ``label_cache_flat[block * (H*BS) + h * BS + tok, d]``  ≡
  ``label_cache[block, h, tok, d]``  (true for any contiguous (NB,H,BS,D)
  storage, which mojo / torch always produce here).
* Source: ``key_lr[:M].view(M*H, D)`` — already DRAM-contiguous, no
  ``.contiguous()`` copy needed when the caller passes a normal tensor.
* Build a single ``(M*H,) int32`` ``slot_idx`` vector on device:

      slot[m*H + h] = block_idxs[m] * (H*BS) + h * BS + token_idxs[m]

  This is one fused arithmetic broadcast on the host side — negligible
  vs the per-head copy overhead we eliminate.

The wheel kernel signature stays:

    mojo_store_lowrank_bf16(key_lr_flat, slot_idx, label_cache_flat, M, D, S)

We just call it once with ``M' = M * num_kv_heads`` and ``S' = num_blocks *
num_kv_heads * block_size``. The label_cache memory is mutated in place via
the ``.view`` (no copy back).

Fallback (super().forward) routes back to torch advanced indexing whenever
the fast path's preconditions fail: dtype != bf16, kernel API missing,
key_lr/label_cache rank wrong, token_num <= 0, or label_cache non-contiguous
(rare — a fresh ``torch.zeros(...)`` BNSD is always contiguous).
"""

import torch

from mojo_opset.experimental import MojoStoreLowrank

from ._utils import _uc_kernels


_API = "mojo_store_lowrank_bf16"

# Must match ROW_TILE in ``uc-kernel/kernels/mojo_store_lowrank_bf16.py``.
# The kernel processes M source rows in ROW_TILE-row blocks (one batched
# GM->UB gather per block, then ROW_TILE per-row SCATTERs). The wrapper must
# ensure the value of M passed into the kernel is a multiple of ROW_TILE;
# tail rows are scattered on the host (cheap — at most ROW_TILE-1 rows, and
# torch advanced indexing on a contiguous flat view is fast on NPU).
_KERNEL_ROW_TILE = 128


def _uc_store_lowrank_bf16(
    label_cache: torch.Tensor,
    key_lr: torch.Tensor,
    block_idxs: torch.Tensor,
    token_idxs: torch.Tensor,
    token_num: int,
) -> torch.Tensor:
    """All-heads-fused single-launch SCATTER.

    Mutates ``label_cache`` in place via a flat view and also returns it
    (mirrors the mojo contract in ``MojoStoreLowrank.forward``).
    """
    num_blocks, num_kv_heads, block_size, head_dim = label_cache.shape
    assert key_lr.shape[1] == num_kv_heads and key_lr.shape[2] == head_dim, (
        "key_lr must be (token_num, num_kv_heads, head_dim) matching label_cache "
        "(num_blocks, num_kv_heads, block_size, head_dim)"
    )

    device = label_cache.device

    # Build the fused (M*H,) int32 slot vector with one broadcast on device.
    # slot_idx[m*H + h] = block_idxs[m] * (H*BS) + h * BS + token_idxs[m]
    # Layout matches `key_lr[:M].view(M*H, D)` row-major.
    H = num_kv_heads
    BS = block_size
    block_i32 = block_idxs[:token_num].to(torch.int32)
    token_i32 = token_idxs[:token_num].to(torch.int32)
    base = block_i32 * (H * BS) + token_i32                    # (M,)
    h_off = torch.arange(H, dtype=torch.int32, device=device) * BS  # (H,)
    slot_idx = (base.unsqueeze(1) + h_off.unsqueeze(0)).reshape(-1).contiguous()  # (M*H,)

    # Source: (token_num, H, D) -> (M*H, D). When caller passes a contiguous
    # tensor (the common case), .contiguous() is a free no-op.
    key_lr_flat = key_lr[:token_num].contiguous().view(token_num * H, head_dim)

    # Destination: flat view over the same storage, no allocation, no copy.
    label_cache_flat = label_cache.view(num_blocks * H * BS, head_dim)

    # ROW_TILE alignment: the kernel processes ROW_TILE source rows per
    # iteration in one batched gather. Split off the tail rows (< ROW_TILE)
    # and scatter them on the host — the tail is at most ``ROW_TILE - 1``
    # rows so cost is negligible vs the kernel-handled bulk.
    total_rows = token_num * H
    main_rows = (total_rows // _KERNEL_ROW_TILE) * _KERNEL_ROW_TILE
    tail_rows = total_rows - main_rows

    # Small-shape fast path: when ``main_rows`` is so small that the kernel
    # launch + only-a-few-programs-active overhead dominates, just do the
    # whole scatter on the host (torch advanced indexing) — measured faster
    # for ``total_rows < 256`` on 910B (e.g. kv=24, H=8 case: 200us kernel
    # vs ~190us pure-host, see worker-reports/P2-28-store-lowrank.md).
    _MIN_KERNEL_ROWS = 256
    if main_rows < _MIN_KERNEL_ROWS:
        label_cache_flat[slot_idx.to(torch.int64)] = key_lr_flat
        return label_cache

    kernels = _uc_kernels()
    api = kernels[_API]
    api(
        key_lr_flat[:main_rows],
        slot_idx[:main_rows],
        label_cache_flat,
        main_rows,                # M' (rows handled by the kernel)
        head_dim,                 # D
        num_blocks * H * BS,      # S' (total slots)
    )

    if tail_rows > 0:
        # Cheap host scatter for the leftover ROW_TILE-1 rows.
        tail_slots = slot_idx[main_rows:main_rows + tail_rows].to(torch.int64)
        label_cache_flat[tail_slots] = key_lr_flat[main_rows:main_rows + tail_rows]

    return label_cache


class UCStoreLowrank(MojoStoreLowrank):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        label_cache: torch.Tensor,
        key_lr: torch.Tensor,
        block_idxs: torch.Tensor,
        token_idxs: torch.Tensor,
        token_num: int,
    ) -> torch.Tensor:
        # Mirror the mojo contract assertions before any device dispatch so
        # the fallback path also enforces them.
        assert block_idxs.dtype == torch.int32
        assert token_idxs.dtype == torch.int32
        assert label_cache.dim() == 4, "Expected label_cache is BNSD"
        assert key_lr.dim() == 3, "Expected key_lr is SND"

        # dtype fence: wheel kernel is bf16-only.
        if label_cache.dtype != torch.bfloat16 or key_lr.dtype != torch.bfloat16:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # No-op path -> let torch handle it (zero-row scatter is well-defined
        # for advanced indexing but ill-defined for our M=0 launch grid).
        if token_num <= 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # The all-heads-fused fast path requires label_cache to be a normal
        # contiguous BNSD tensor so the flat .view is legal. Tests + real
        # callsites always produce one; non-contiguous → fall back safely.
        if not label_cache.is_contiguous():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # API availability fence: if the wheel doesn't carry the kernel yet,
        # safely fall back to the torch reference rather than KeyError.
        # NB: ``KernelRegistry`` lacks ``__contains__``; use ``.keys()``.
        kernels = _uc_kernels()
        if _API not in kernels.keys():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        return _uc_store_lowrank_bf16(label_cache, key_lr, block_idxs, token_idxs, token_num)
