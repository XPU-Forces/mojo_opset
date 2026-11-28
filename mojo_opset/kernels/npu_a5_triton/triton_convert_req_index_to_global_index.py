"""Single-kernel request-index conversion on A5."""

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch_npu  # noqa: F401  (registers the npu backend)
import triton
import triton.language as tl
import triton.runtime.driver as driver


# Branch-free single-route A5 index remap with a one-tile look-ahead, one token
# per work item. The request id / page-table cache are scalar (one token per
# tile), so the row buffer is a plain 1D vector; the page-table row of the next
# tile is issued before the current tile's gather so its DMA overlaps the
# on-chip gather. No wave/tail branch, no request-id cache branch.
@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    prefill_ids_ptr,
    prefill_starts_ptr,
    num_tokens: tl.constexpr,
    max_blocks: tl.constexpr,
    MB_P2: tl.constexpr,
    LOG2_BS: tl.constexpr,
    BS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    NUM_PROGS: tl.constexpr,
    TPI: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
    bt_stride0: tl.constexpr,
    bt_stride1: tl.constexpr,
    ti_stride0: tl.constexpr,
    ti_stride1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    row_offs = tl.arange(0, MB_P2)
    row_mask = row_offs < max_blocks
    col = tl.arange(0, BLOCK_N)

    # Peel the first tile (one token per tile, so every index below is a scalar).
    cached = tl.load(req_id_ptr + pid).to(tl.int32)
    rows = tl.load(block_table_ptr + cached * bt_stride0 + row_offs * bt_stride1, mask=row_mask, other=0).to(tl.int32)

    for tile in range(pid, NUM_TILES, NUM_PROGS):
        tok = tl.load(token_indices_ptr + tile * ti_stride0 + col * ti_stride1).to(tl.int32)
        # Look-ahead: issue the next tile's row DMA so it overlaps this gather.
        nxt = tile + NUM_PROGS
        nreq = tl.load(req_id_ptr + tl.minimum(nxt, NUM_TILES - 1)).to(tl.int32)
        changed = (nxt < num_tokens) & (nreq != cached)
        next_rows = tl.load(
            block_table_ptr + nreq * bt_stride0 + row_offs * bt_stride1, mask=row_mask & changed, other=0
        ).to(tl.int32)
        block_id = tok >> LOG2_BS
        inblock_off = tok & BS_MASK
        gather_idx = block_id & (MB_P2 - 1)
        base = tl.gather(rows, gather_idx, axis=0)
        out_val = (base << LOG2_BS) + inblock_off
        invalid = block_id.to(tl.uint32) >= max_blocks
        if HAS_PREFILL:
            p_req = tl.load(prefill_ids_ptr + tile).to(tl.int32)
            is_prefill = p_req.to(tl.float32) >= 0
            workspace_start = tl.load(prefill_starts_ptr + tl.maximum(p_req, 0), mask=is_prefill, other=0).to(tl.int32)
            out_val = tl.where(is_prefill, workspace_start + tok, out_val)
            invalid = tl.where(is_prefill, tok == -1, invalid)
        out_val = tl.where(invalid, -1, out_val)
        tl.store(out_ptr + tile * out_stride0 + col * out_stride1, out_val)
        # Rotate only after this tile is done.
        rows = tl.where(changed, next_rows, rows)
        cached = tl.where(changed, nreq, cached)


class ModelNew(nn.Module):
    """One branch-free kernel route, fixed TPI=1, one-tile page-table look-ahead."""

    def __init__(self):
        super().__init__()
        self.num_cores = driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]

    def forward(
        self,
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
        has_prefill,
        prefill_workspace_request_ids=None,
        prefill_workspace_starts=None,
    ):
        req = req_id.contiguous()
        table = block_table.contiguous()
        indices = token_indices.contiguous()
        if has_prefill:
            assert prefill_workspace_request_ids is not None
            assert prefill_workspace_starts is not None
            prefill_ids = prefill_workspace_request_ids.contiguous()
            prefill_starts = prefill_workspace_starts.contiguous()
        else:
            prefill_ids = req
            prefill_starts = req
        output = torch.empty_like(indices)
        t = req.shape[0]
        m = table.shape[1]
        tpi = 1
        num_tiles = triton.cdiv(t, tpi)
        progs = min(num_tiles, self.num_cores)
        log2_bs = block_size.bit_length() - 1
        assert (1 << log2_bs) == block_size
        # Ascend gather lowering requires at least two lanes in the source.
        # The global page-table load still masks out padding lanes.
        gather_width = max(2, triton.next_power_of_2(m))
        _convert_req_index_to_global_index_kernel[(progs,)](
            req,
            table,
            indices,
            output,
            prefill_ids,
            prefill_starts,
            t,
            m,
            gather_width,
            log2_bs,
            block_size - 1,
            num_topk_tokens,
            num_tiles,
            progs,
            tpi,
            has_prefill,
            table.stride(0),
            table.stride(1),
            indices.stride(0),
            indices.stride(1),
            output.stride(0),
            output.stride(1),
        )
        return output


def _check_widths(block_table: torch.Tensor, token_indices: torch.Tensor) -> None:
    # The kernel keeps a whole page-table row and token tile in unified buffer.
    if token_indices.shape[1] > 8192 or block_table.shape[1] > 4096:
        raise ValueError("Ascend A5 Triton requires K <= 8192 and M <= 4096")


@lru_cache(maxsize=None)
def _model_for_device(device_index: int):
    # Cache only hardware metadata, never tensors or shape-dependent state.
    with torch.npu.device(device_index):
        return ModelNew()


def _conversion_impl(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    num_topk_tokens: int,
    has_prefill: bool,
    prefill_workspace_request_ids: Optional[torch.Tensor],
    prefill_workspace_starts: Optional[torch.Tensor],
) -> torch.Tensor:
    if req_id.numel() == 0:
        return torch.empty_like(token_indices)
    _check_widths(block_table, token_indices)
    with torch.npu.device(req_id.device):
        return _model_for_device(req_id.device.index).forward(
            req_id,
            block_table,
            token_indices,
            block_size,
            num_topk_tokens,
            has_prefill,
            prefill_workspace_request_ids,
            prefill_workspace_starts,
        )


# This integer-only inference op needs no autograd formula. Direct dispatcher
# registration keeps the graph boundary without CustomOpDef's per-call Python
# checks; schema, mutation and fake metadata are checked by the test suite.
_library = torch.library.Library("mojo_npu_triton_a5", "FRAGMENT")
_library.define(
    "triton_convert_req_index_to_global_index" + torch.library.infer_schema(_conversion_impl, mutates_args=())
)
_library.impl("triton_convert_req_index_to_global_index", _conversion_impl, "PrivateUse1")
triton_convert_req_index_to_global_index_fwd = (
    torch.ops.mojo_npu_triton_a5.triton_convert_req_index_to_global_index.default
)


@torch.library.register_fake("mojo_npu_triton_a5::triton_convert_req_index_to_global_index")
def _conversion_fake(
    req_id,
    block_table,
    token_indices,
    block_size,
    num_topk_tokens,
    has_prefill,
    prefill_workspace_request_ids,
    prefill_workspace_starts,
):
    if req_id.numel() == 0:
        return torch.empty_like(token_indices)
    _check_widths(block_table, token_indices)
    return torch.empty_like(token_indices, memory_format=torch.contiguous_format)
