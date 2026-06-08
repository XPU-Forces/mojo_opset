"""UC backend for ``MojoStorePagedKVCache``.

Drives the wheel-side **twin block SCATTER** kernel
``mojo_store_paged_kv_cache_bf16`` which, for ``M = total_tokens * num_kv_heads``
tasks, scatters one K row + one V row per task into the flattened paged
cache view ``(num_blocks * num_kv_heads * block_size, head_dim)``.

Shape contract translation
--------------------------

* Mojo high-level layout (canonical BHTD, contiguous):
    ``key_cache, value_cache``: ``(num_blocks, kv_heads, block_size, head_dim)``
    ``key_states, value_states``: ``(token_num, kv_heads, head_dim)``

* Wheel kernel sees only flat 2-D forms (both zero-copy ``view`` of the
  contiguous originals):
    ``k_cache_flat, v_cache_flat``: ``(num_blocks * kv_heads * block_size, head_dim)``
    ``new_k, new_v``: ``(token_num * kv_heads, head_dim)``

* Per-task indices computed host-side (one 1-D int32 vector each):
    ``src_idx[m]  = src_token_row[m // kvh] * kvh + (m % kvh)``
    ``slot_idx[m] = block_id[m // kvh] * (kvh * block_size)
                   + (m % kvh) * block_size + offset[m // kvh]``

  where ``src_token_row``, ``block_id``, ``offset`` come from expanding
  ``chunk_metadata`` per token (``cumsum + searchsorted`` trick, no
  Python-side per-chunk loop).

Versus the prior wrapper
------------------------
Earlier UC wrapper did, per kv head ``h``:

    ``k_src = key_states[src_idx, h, :].contiguous()``                       (gather)
    ``k_cache_h = key_cache[:, h, :, :].contiguous().view(NB*BS, HD)``       (full-cache memcpy!)
    ``kernel(k_src, v_src, slot, k_cache_h, v_cache_h)``                     (per-head launch)
    ``key_cache[:, h, :, :] = k_cache_h.view(NB, BS, HD)``                   (full-cache writeback)

For a chunked prefill on ``num_kv_heads=2`` with a several-thousand-block
cache that's two ``O(num_blocks * block_size * head_dim)`` host bf16 copies
per call (10s of MB), serialised on the device stream — by far the
dominant cost. The new design eliminates every host memcpy on the cache
side (just ``view``) and folds all per-head launches into a single
kernel call.

Lifter / codegen constraints honoured (lessons §A.3 / §B):
    * Both ``src_idx`` and ``slot_idx`` are ``int32`` direct ``BufferLoad``
      (host pre-collapses block * kvh * BS + h * BS + off).
    * Indirect copies are GM <-> UB only (block GATHER for source row,
      block SCATTER for destination row).
    * Single runtime-varying region offset dim per copy (the row id).

Fallback: any of {dtype != bf16, kernel API unavailable, empty chunk
plan, non-contiguous states / cache that can't be made contiguous in
place, head_dim not a multiple of 128} routes back to ``super().forward``
(the reference torch implementation in
``mojo_opset/core/operators/kv_cache.py``).
"""

from typing import Optional, Tuple

import torch

from mojo_opset.core import MojoStorePagedKVCache
from mojo_opset.core.operators.kv_cache import (
    assert_paged_kv_store_contract,
    build_paged_kv_chunk_metadata,
)

from ._utils import _uc_kernels


_API = "mojo_store_paged_kv_cache_bf16"
# Must match constant ``D_TILE`` in
#   uc-kernel/kernels/mojo_store_paged_kv_cache.py
_D_TILE = 128


def _build_plan_cpu(
    chunk_metadata: torch.Tensor,
    block_size: int,
    kv_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the head-folded (src_idx, slot_idx) plan in one CPU pass.

    The plan is small (M = total_tokens * kv_heads int32 each), and each
    on-device torch op carries ~10-30µs of NPU launch overhead. The chain
    of `cumsum + searchsorted + arange + broadcast + cast` to expand
    chunk_metadata to per-(token, head) indices runs in <50µs on CPU but
    adds ~300µs of NPU launch latency when done on device.

    We therefore:
      1. Copy chunk_metadata to CPU once (small, ~few hundred int32s).
      2. Expand to per-token plan with numpy ops (sub-µs each).
      3. Broadcast per-token plan to per-(token, head) plan.
      4. Stack (src_idx, slot_idx) into one (2, M) int32 tensor and ship
         via a single H2D transfer.

    Returns ``(src_idx (M,) int32, slot_idx (M,) int32)`` with
    ``M = total_tokens * kv_heads``, both on the same device as
    ``chunk_metadata``.
    """
    import numpy as np
    device = chunk_metadata.device
    cm = chunk_metadata.detach().cpu().numpy().astype("int64", copy=False)
    n_chunks = cm.shape[0]
    if n_chunks == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty

    src_starts = cm[:, 0]
    dst_blocks = cm[:, 1]
    dst_offs = cm[:, 2]
    chunk_lens = cm[:, 3]

    total_tokens = int(chunk_lens.sum())
    if total_tokens == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty

    # Per-token (src_token_row, block_id, offset).
    cu_lens = np.empty(n_chunks + 1, dtype="int64")
    cu_lens[0] = 0
    np.cumsum(chunk_lens, out=cu_lens[1:])
    arange = np.arange(total_tokens, dtype="int64")
    chunk_idx = np.searchsorted(cu_lens, arange, side="right") - 1
    within = arange - cu_lens[chunk_idx]
    src_token_row = src_starts[chunk_idx] + within
    block_id = dst_blocks[chunk_idx]
    offset = dst_offs[chunk_idx] + within

    # Fold per-head into M = total_tokens * kv_heads tasks, token-major
    # ordering so ``states.view(T * kvh, HD)`` row m maps to
    # (m // kvh, m % kvh) and matches src_idx[m].
    block_stride = kv_heads * block_size
    head_arange = np.arange(kv_heads, dtype="int64")
    src_idx_2d = src_token_row[:, None] * kv_heads + head_arange[None, :]
    slot_idx_2d = (
        block_id[:, None] * block_stride
        + head_arange[None, :] * block_size
        + offset[:, None]
    )
    # Single contiguous int32 tensor + one H2D for both vectors. Stacking
    # along dim 0 lets us split with .view after device-side narrow.
    fused = np.empty((2, total_tokens * kv_heads), dtype="int32")
    fused[0] = src_idx_2d.reshape(-1)
    fused[1] = slot_idx_2d.reshape(-1)
    fused_t = torch.from_numpy(fused).to(device=device, non_blocking=True)
    return fused_t[0].contiguous(), fused_t[1].contiguous()


def _uc_store_paged_kv_bf16(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    chunk_metadata: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-launch host-side folded SCATTER. Mutates caches in place."""
    if chunk_metadata.shape[0] == 0:
        return key_cache, value_cache

    num_blocks, num_kv_heads, block_size, head_dim = key_cache.shape
    assert value_cache.shape == key_cache.shape, "key/value cache shapes must match"
    assert key_states.shape[1] == num_kv_heads and key_states.shape[2] == head_dim
    assert value_states.shape == key_states.shape

    src_idx, slot_idx = _build_plan_cpu(chunk_metadata, block_size, num_kv_heads)
    total_tasks = src_idx.shape[0]
    if total_tasks == 0:
        return key_cache, value_cache

    # Zero-copy flat views of contiguous tensors.
    new_k_flat = key_states.view(key_states.shape[0] * num_kv_heads, head_dim)
    new_v_flat = value_states.view(value_states.shape[0] * num_kv_heads, head_dim)
    k_cache_flat = key_cache.view(num_blocks * num_kv_heads * block_size, head_dim)
    v_cache_flat = value_cache.view(num_blocks * num_kv_heads * block_size, head_dim)

    # Wheel ABI: trailing INT32 scalars follow first-appearance order of
    # dynamic dims in the prim_func type annotations: (M, D, S).
    M_arg = int(total_tasks)
    D_arg = int(head_dim)
    S_arg = int(num_blocks * num_kv_heads * block_size)
    _uc_kernels()[_API](
        new_k_flat,
        new_v_flat,
        src_idx,
        slot_idx,
        k_cache_flat,
        v_cache_flat,
        M_arg,
        D_arg,
        S_arg,
    )
    return key_cache, value_cache


class UCStorePagedKVCache(MojoStorePagedKVCache):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        context_kv_lens: Optional[torch.Tensor] = None,
        *,
        chunk_metadata: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- 1. Cheap fast-path guards --------------------------------
        # Wheel kernel is bf16-only.
        if key_states.dtype is not torch.bfloat16 or key_cache.dtype is not torch.bfloat16:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # head_dim must be a multiple of the kernel's column tile.
        if key_cache.dim() != 4 or key_cache.shape[3] % _D_TILE != 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # The flat views require the source / cache tensors to be contiguous
        # on the canonical BHTD / THD layout; any custom stride layout
        # bails to the torch reference for safety.
        if (
            not key_states.is_contiguous()
            or not value_states.is_contiguous()
            or not key_cache.is_contiguous()
            or not value_cache.is_contiguous()
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        kernels = _uc_kernels()
        if _API not in kernels.keys():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ---- 2. Resolve chunk plan (same shape semantics as parent) ---
        if chunk_metadata is None:
            assert block_table is not None, "block_table is required when chunk_metadata is not provided."
            assert context_kv_lens is not None, "context_kv_lens is required when chunk_metadata is not provided."
            chunk_metadata = build_paged_kv_chunk_metadata(
                block_table, cu_q_lens, context_kv_lens, key_cache.shape[2],
            )
        else:
            assert block_table is None and cu_q_lens is None and context_kv_lens is None, (
                "chunk_metadata path should not be mixed with block_table/cu_q_lens/context_kv_lens."
            )
        assert_paged_kv_store_contract(chunk_metadata)

        return _uc_store_paged_kv_bf16(
            key_states,
            value_states,
            key_cache,
            value_cache,
            chunk_metadata,
        )
