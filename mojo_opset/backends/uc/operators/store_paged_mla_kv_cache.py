"""UC backend wrapper for ``MojoStorePagedMLAKVCache``.

The hot path drives ``mojo_store_paged_mla_kv_cache_bf16`` (a fixed-shape
twin block SCATTER of MLA latent ``compressed_kv`` + positional ``k_pe``
tokens into two paged caches sharing the same ``block_table``).

All bookkeeping (decoding ``block_table`` / ``cu_q_lens`` / ``context_kv_lens``
into per-token flat slot indices) is done host-side, so the kernel only sees
a single 1-D ``int32`` index tensor with one entry per token.  Any workload
that does not match the hard-coded fast-path shape, dtype, or that contains
invalid (out-of-cache) entries falls back to the parent's native torch
implementation.
"""

from typing import Optional
from typing import Tuple

import torch

from mojo_opset.core.operators.kv_cache import assert_paged_kv_layout_contract
from mojo_opset.experimental.operators.kv_cache import MojoStorePagedMLAKVCache
from mojo_opset.utils.logging import get_logger

from ._utils import _uc_kernels


logger = get_logger(__name__)


# Must match constants in
#   uc-kernel/kernels/mojo_store_paged_mla_kv_cache_bf16.py
_T_MAX = 2048
_D_CKV = 512
_D_KPE = 64
_CACHE_ROWS = 32768

_KERNEL_API = "mojo_store_paged_mla_kv_cache_bf16"


def _build_mla_flat_slot_indices(
    block_table: torch.Tensor,
    cu_q_lens: Optional[torch.Tensor],
    context_kv_lens: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> Optional[torch.Tensor]:
    """Compute per-token flat slot indices into the (num_blocks*block_size, D)
    view of the paged cache.

    Returns ``None`` if any token would map to an invalid (-1) slot, signalling
    the caller to fall back to the parent torch path.
    """
    device = block_table.device
    batch_size = context_kv_lens.shape[0]
    is_decode = cu_q_lens is None

    context_i32 = context_kv_lens.to(torch.int32)
    if is_decode:
        q_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
        cu = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(q_lens, dim=0, dtype=torch.int32),
            ]
        )
        total_tokens = int(batch_size)
    else:
        cu = cu_q_lens.to(torch.int32)
        q_lens = cu[1:] - cu[:-1]
        total_tokens = int(cu[-1].item())

    if total_tokens == 0:
        return torch.empty(0, dtype=torch.int32, device=device)

    # Drop padding batches (context_kv_lens == -1 / q_lens <= 0): give those
    # batches zero-length so they contribute no tokens.
    valid_batch = (context_i32 >= 0) & (q_lens > 0)
    if not bool(valid_batch.all().item()):
        return None  # fall back: parent already handles padding rows.

    # token -> batch_id via searchsorted on the cu_q boundaries.
    token_idx = torch.arange(total_tokens, dtype=torch.int32, device=device)
    # ``right=True`` so token_idx == cu[i] maps to batch i (not i-1).
    token_batch_id = (
        torch.searchsorted(cu[1:].contiguous(), token_idx, right=True)
        .clamp_max(batch_size - 1)
        .to(torch.int32)
    )
    token_off_in_seq = token_idx - cu[token_batch_id]

    write_pos = context_i32[token_batch_id] + token_off_in_seq  # (T,) int32
    logical_block = torch.div(write_pos, block_size, rounding_mode="floor")
    block_offset = write_pos - logical_block * block_size

    max_blocks_per_seq = block_table.shape[1]
    if int(logical_block.max().item()) >= max_blocks_per_seq:
        return None  # would index past the block_table column count.

    physical_block = block_table[token_batch_id.long(), logical_block.long()]
    if bool((physical_block < 0).any().item()):
        return None  # unmapped logical block -> torch path.

    flat_slot = physical_block.to(torch.int32) * block_size + block_offset
    if int(flat_slot.max().item()) >= num_blocks * block_size:
        return None  # would overrun the cache slot space.

    return flat_slot


class UCStorePagedMLAKVCache(MojoStorePagedMLAKVCache):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        compressed_kv_states: torch.Tensor,
        k_pe_states: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_q_lens: torch.Tensor,
        context_kv_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- 1. Cheap fast-path guards ---------------------------------
        # Dtype must be bf16 on the value tensors.
        if (
            compressed_kv_states.dtype is not torch.bfloat16
            or k_pe_states.dtype is not torch.bfloat16
            or compressed_kv_cache.dtype is not torch.bfloat16
            or k_pe_cache.dtype is not torch.bfloat16
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Layout contract (cheap asserts; same as parent).
        try:
            assert_paged_kv_layout_contract(block_table, cu_q_lens, context_kv_lens)
        except AssertionError:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Cache must be 4-D ``(num_blocks, 1, block_size, D)``.
        if compressed_kv_cache.dim() != 4 or k_pe_cache.dim() != 4:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if compressed_kv_cache.shape[1] != 1 or k_pe_cache.shape[1] != 1:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        num_blocks_ckv, _, block_size_ckv, d_ckv = compressed_kv_cache.shape
        num_blocks_kpe, _, block_size_kpe, d_kpe = k_pe_cache.shape

        # Both caches must agree on (num_blocks, block_size).
        if num_blocks_ckv != num_blocks_kpe or block_size_ckv != block_size_kpe:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Fixed-shape match: D_CKV / D_KPE / flat cache row count.
        if d_ckv != _D_CKV or d_kpe != _D_KPE:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        cache_rows = num_blocks_ckv * block_size_ckv
        if cache_rows != _CACHE_ROWS:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # State tensors must be 2-D matching (T, D).
        if compressed_kv_states.dim() != 2 or k_pe_states.dim() != 2:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        total_tokens = compressed_kv_states.shape[0]
        if (
            total_tokens != k_pe_states.shape[0]
            or compressed_kv_states.shape[1] != _D_CKV
            or k_pe_states.shape[1] != _D_KPE
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Kernel is hard-wired to T_MAX tokens per launch.
        if total_tokens != _T_MAX:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Wheel availability.
        kernels = _uc_kernels()
        if _KERNEL_API not in kernels:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ---- 2. Build per-token flat slot indices (host-side) ---------
        flat_slot = _build_mla_flat_slot_indices(
            block_table,
            cu_q_lens,
            context_kv_lens,
            block_size_ckv,
            num_blocks_ckv,
        )
        if flat_slot is None:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        if flat_slot.shape[0] != _T_MAX:
            # Either decode_mode/prefill produced fewer tokens than the kernel
            # is sized for, or extra padding tokens appeared; either way bail.
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # ---- 3. Launch the kernel -------------------------------------
        # Cache views: (num_blocks, 1, block_size, D) -> (num_blocks*block_size, D).
        # The cache is contiguous in the layout above, so the flatten is a
        # no-copy view.
        if not compressed_kv_cache.is_contiguous() or not k_pe_cache.is_contiguous():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        ckv_src = compressed_kv_states.contiguous()
        kpe_src = k_pe_states.contiguous()
        idx = flat_slot.contiguous().to(torch.int32)

        ckv_out = compressed_kv_cache.view(_CACHE_ROWS, _D_CKV)
        kpe_out = k_pe_cache.view(_CACHE_ROWS, _D_KPE)

        kernels[_KERNEL_API](ckv_src, kpe_src, idx, ckv_out, kpe_out)
        return compressed_kv_cache, k_pe_cache
