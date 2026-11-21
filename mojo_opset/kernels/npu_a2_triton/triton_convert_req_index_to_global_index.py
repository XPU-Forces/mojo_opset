"""A2 request-index conversion; GLM-5 balanced workloads and generic fallback."""

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _query_vector_cores() -> int:
    import torch_npu  # noqa: F401
    import triton.runtime.driver as driver

    device = torch_npu.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    return int(props["num_vectorcore"])


@triton.jit
def _convert_req_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    prefill_req_ptr,
    workspace_ptr,
    num_tokens: tl.constexpr,
    num_programs: tl.constexpr,
    LOG2_BS: tl.constexpr,
    LOG2_MAX: tl.constexpr,
    BS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
    EARLY_LOAD: tl.constexpr,
    SINGLE_REQUEST: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, BLOCK_N)
    bt_cols = tl.arange(0, BLOCK_BT)
    r_off = tl.arange(0, ROWS)

    chunk = tl.cdiv(tl.cdiv(num_tokens, num_programs), ROWS) * ROWS
    start = pid * chunk
    end = tl.minimum(start + chunk, num_tokens)

    if SINGLE_REQUEST:
        table_single = tl.load(block_table_ptr + bt_cols, mask=bt_cols < MAX_BLOCKS, other=0)

    for row0 in range(start, end, ROWS):
        rows = row0 + r_off

        if EXACT:
            if EARLY_LOAD:
                tok = tl.load(token_indices_ptr + rows[:, None] * BLOCK_N + cols[None, :])
            reqs = tl.load(req_id_ptr + rows).to(tl.int32)
            if HAS_PREFILL:
                prefs = tl.load(prefill_req_ptr + rows).to(tl.int32)
                is_pref = prefs >= 0
                ws = tl.load(workspace_ptr + tl.maximum(prefs, 0), mask=is_pref, other=0).to(tl.int32)
            if SINGLE_REQUEST:
                bt_2d = tl.broadcast_to(table_single[None, :], (ROWS, BLOCK_BT))
            else:
                if MAX_BLOCKS == BLOCK_BT:
                    bt_2d = tl.load(block_table_ptr + reqs[:, None] * MAX_BLOCKS + bt_cols[None, :])
                else:
                    bt_2d = tl.load(
                        block_table_ptr + reqs[:, None] * MAX_BLOCKS + bt_cols[None, :],
                        mask=bt_cols[None, :] < MAX_BLOCKS,
                        other=0,
                    )
            if not EARLY_LOAD:
                tok = tl.load(token_indices_ptr + rows[:, None] * BLOCK_N + cols[None, :])
        else:
            r_mask = rows < end
            if EARLY_LOAD:
                tok = tl.load(
                    token_indices_ptr + rows[:, None] * BLOCK_N + cols[None, :],
                    mask=r_mask[:, None],
                    other=-1,
                )
            reqs = tl.load(req_id_ptr + rows, mask=r_mask, other=0).to(tl.int32)
            if HAS_PREFILL:
                prefs = tl.load(prefill_req_ptr + rows, mask=r_mask, other=-1).to(tl.int32)
                is_pref = prefs >= 0
                ws = tl.load(workspace_ptr + tl.maximum(prefs, 0), mask=is_pref, other=0).to(tl.int32)
            if SINGLE_REQUEST:
                bt_2d = tl.broadcast_to(table_single[None, :], (ROWS, BLOCK_BT))
            else:
                if MAX_BLOCKS == BLOCK_BT:
                    bt_2d = tl.load(
                        block_table_ptr + reqs[:, None] * MAX_BLOCKS + bt_cols[None, :],
                        mask=r_mask[:, None],
                        other=0,
                    )
                else:
                    bt_2d = tl.load(
                        block_table_ptr + reqs[:, None] * MAX_BLOCKS + bt_cols[None, :],
                        mask=(bt_cols[None, :] < MAX_BLOCKS) & r_mask[:, None],
                        other=0,
                    )
            if not EARLY_LOAD:
                tok = tl.load(
                    token_indices_ptr + rows[:, None] * BLOCK_N + cols[None, :],
                    mask=r_mask[:, None],
                    other=-1,
                )

        block_id = tok >> LOG2_BS
        base = tl.gather(bt_2d, block_id & (BLOCK_BT - 1), 1)

        out_val = tok + ((base - block_id) << LOG2_BS)
        if MAX_BLOCKS & (MAX_BLOCKS - 1) == 0:
            # Exact mask: zero tag -> 0; every nonzero int32 tag -> -1.
            if BLOCK_N <= 512 and EARLY_LOAD and (MAX_BLOCKS << LOG2_BS) <= 2147483648:
                # Bias the original token range directly; avoid a second index shift.
                invalid_mask = (tok | (tok + (2147483648 - (MAX_BLOCKS << LOG2_BS)))) >> 31
            else:
                tag = block_id >> LOG2_MAX
                invalid_mask = (tag | (tag + 2147483647)) >> 31
            out_val = out_val | invalid_mask
        else:
            invalid = (block_id < 0) | (block_id >= MAX_BLOCKS)
            out_val = tl.where(invalid, -1, out_val)

        if HAS_PREFILL:
            out_pref = ws[:, None] + tok
            out_val = tl.where(
                is_pref[:, None],
                tl.where(tok == -1, -1, out_pref),
                out_val,
            )

        if EXACT:
            tl.store(out_ptr + rows[:, None] * BLOCK_N + cols[None, :], out_val)
        else:
            tl.store(
                out_ptr + rows[:, None] * BLOCK_N + cols[None, :],
                out_val,
                mask=r_mask[:, None],
            )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._num_cores = None
        try:
            self._num_cores = _query_vector_cores()
        except Exception:
            self._num_cores = None

    def _cores(self) -> int:
        if self._num_cores is None:
            self._num_cores = _query_vector_cores()
        return self._num_cores

    def forward(
        self,
        req_ids,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
        HAS_PREFILL_WORKSPACE=False,
        prefill_workspace_request_ids=None,
        prefill_workspace_starts=None,
    ):
        assert req_ids.dtype == torch.int32
        assert block_table.dtype == torch.int32
        assert token_indices.dtype == torch.int32
        assert token_indices.shape[1] == num_topk_tokens
        assert req_ids.shape[0] == token_indices.shape[0]

        num_tokens = req_ids.shape[0]
        max_blocks = block_table.shape[1]
        block_n = triton.next_power_of_2(num_topk_tokens)
        block_bt = triton.next_power_of_2(max_blocks)
        log2_bs = block_size.bit_length() - 1

        req_c = req_ids.contiguous()
        bt_c = block_table.contiguous()
        ti_c = token_indices.contiguous()
        out = torch.empty_like(ti_c)

        if HAS_PREFILL_WORKSPACE:
            pref_c = prefill_workspace_request_ids.contiguous()
            ws_c = prefill_workspace_starts.contiguous()
        else:
            pref_c = req_c
            ws_c = req_c

        cores = self._cores()
        rows = 8 if block_n <= 512 else 4
        # A2 UB: large workspace tiles require fewer rows than decode tiles.
        if HAS_PREFILL_WORKSPACE and block_n >= 2048:
            rows = 2
        # Wide page-table tiles exceed A2 UB capacity with four rows.
        if block_n >= 1024 and block_bt >= 2048:
            rows = 2
        progs = min((num_tokens + rows - 1) // rows, cores)
        chunk = ((num_tokens + progs - 1) // progs + rows - 1) // rows * rows
        exact = (chunk % rows == 0) and (num_tokens % rows == 0)
        grid = (progs,)

        if num_tokens <= 12:
            _convert_small_req_index_kernel[(num_tokens,)](
                req_c,
                bt_c,
                ti_c,
                out,
                pref_c,
                ws_c,
                LOG2_BS=log2_bs,
                LOG2_MAX=max_blocks.bit_length() - 1,
                BLOCK_N=block_n,
                BLOCK_BT=block_bt,
                MAX_BLOCKS=max_blocks,
                HAS_PREFILL=bool(HAS_PREFILL_WORKSPACE),
            )
            return out

        _convert_req_index_kernel[grid](
            req_c,
            bt_c,
            ti_c,
            out,
            pref_c,
            ws_c,
            num_tokens,
            grid[0],
            LOG2_BS=log2_bs,
            LOG2_MAX=max_blocks.bit_length() - 1,
            BS_MASK=block_size - 1,
            BLOCK_N=block_n,
            BLOCK_BT=block_bt,
            MAX_BLOCKS=max_blocks,
            ROWS=rows,
            EXACT=exact,
            HAS_PREFILL=bool(HAS_PREFILL_WORKSPACE),
            EARLY_LOAD=num_tokens > 64,
            SINGLE_REQUEST=(block_table.shape[0] == 1 and not HAS_PREFILL_WORKSPACE),
        )
        return out


@triton.jit
def _convert_small_req_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    prefill_req_ptr,
    workspace_ptr,
    LOG2_BS: tl.constexpr,
    LOG2_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, BLOCK_N)
    bt_cols = tl.arange(0, BLOCK_BT)
    req = tl.load(req_id_ptr + row).to(tl.int32)
    tok = tl.load(token_indices_ptr + row * BLOCK_N + cols)
    if MAX_BLOCKS == BLOCK_BT:
        table = tl.load(block_table_ptr + req * MAX_BLOCKS + bt_cols)
    else:
        table = tl.load(
            block_table_ptr + req * MAX_BLOCKS + bt_cols,
            mask=bt_cols < MAX_BLOCKS,
            other=0,
        )
    bid = tok >> LOG2_BS
    base = tl.gather(table, bid & (BLOCK_BT - 1), 0)
    result = tok + ((base - bid) << LOG2_BS)
    if MAX_BLOCKS & (MAX_BLOCKS - 1) == 0:
        invalid = (bid >> LOG2_MAX) != 0
    else:
        invalid = (bid < 0) | (bid >= MAX_BLOCKS)
    result = tl.where(invalid, -1, result)
    if HAS_PREFILL:
        pref = tl.load(prefill_req_ptr + row).to(tl.int32)
        is_pref = pref >= 0
        ws = tl.load(workspace_ptr + tl.maximum(pref, 0), mask=is_pref, other=0)
        result = tl.where(is_pref, tl.where(tok == -1, -1, ws + tok), result)
    tl.store(out_ptr + row * BLOCK_N + cols, result)


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
_library = torch.library.Library("mojo_npu_triton_a2", "FRAGMENT")
_library.define(
    "triton_convert_req_index_to_global_index" + torch.library.infer_schema(_conversion_impl, mutates_args=())
)
_library.impl("triton_convert_req_index_to_global_index", _conversion_impl, "PrivateUse1")
triton_convert_req_index_to_global_index_fwd = (
    torch.ops.mojo_npu_triton_a2.triton_convert_req_index_to_global_index.default
)


@torch.library.register_fake("mojo_npu_triton_a2::triton_convert_req_index_to_global_index")
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
    return torch.empty_like(token_indices, memory_format=torch.contiguous_format)
