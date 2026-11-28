"""Single-kernel request-index conversion on A2."""

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    prefill_req_ptr,
    workspace_ptr,
    T: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    LOG2_BS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    ROWS: tl.constexpr,
    CHK: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    bt_cols = tl.arange(0, BLOCK_BT)
    r_off = tl.arange(0, ROWS)
    chk = tl.arange(0, CHK)
    col_ok = cols < K
    bt_ok = bt_cols < M
    chunk = tl.cdiv(tl.cdiv(T, P), ROWS) * ROWS
    start = pid * chunk
    end = tl.minimum(start + chunk, T)
    row = start
    while row < end:
        cur = tl.load(req_id_ptr + row).to(tl.int32)
        table = tl.load(block_table_ptr + cur * M + bt_cols, mask=bt_ok, other=0)
        nxt = end
        for c0 in range(row, end, CHK):
            idx = c0 + chk
            reqs = tl.load(req_id_ptr + idx, mask=idx < end, other=cur).to(tl.int32)
            bad = (reqs != cur) & (idx < end)
            nxt = tl.minimum(nxt, c0 + tl.min(tl.where(bad, chk, T), 0))
        for row0 in range(row, nxt, ROWS):
            rows = row0 + r_off
            r_ok = rows < nxt
            tok = tl.load(
                token_indices_ptr + rows[:, None] * K + cols[None, :],
                mask=r_ok[:, None] & col_ok[None, :],
                other=-1,
            )
            # Clamp the gather index and reject entries changed by the clamp.
            bid = tok >> LOG2_BS
            bid_c = tl.minimum(tl.maximum(bid, 0), M - 1)
            t2d = tl.broadcast_to(table[None, :], (ROWS, BLOCK_BT))
            base = tl.gather(t2d, bid_c, 1)
            result = tok + ((base - bid) << LOG2_BS)
            result = tl.where(bid == bid_c, result, -1)
            if HAS_PREFILL:
                pref = tl.load(prefill_req_ptr + rows, mask=r_ok, other=-1).to(tl.int32)
                ws = tl.load(
                    workspace_ptr + tl.maximum(pref, 0),
                    mask=pref >= 0,
                    other=0,
                ).to(tl.int32)
                result = tl.where(
                    pref[:, None] >= 0,
                    tl.where(tok == -1, -1, ws[:, None] + tok),
                    result,
                )
            tl.store(
                out_ptr + rows[:, None] * K + cols[None, :],
                result,
                mask=r_ok[:, None] & col_ok[None, :],
            )
        row = nxt


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        import torch_npu
        import triton.runtime.driver as driver

        self._num_cores = int(
            driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]
        )

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
        assert req_ids.dtype == block_table.dtype == token_indices.dtype == torch.int32
        assert req_ids.shape[0] == token_indices.shape[0]
        assert token_indices.shape[1] == num_topk_tokens
        assert block_size > 0 and (block_size & (block_size - 1)) == 0
        t = req_ids.shape[0]
        assert t > 0 and num_topk_tokens > 0 and block_table.shape[1] > 0
        req = req_ids.contiguous()
        table = block_table.contiguous()
        tok = token_indices.contiguous()
        out = torch.empty_like(tok)
        if HAS_PREFILL_WORKSPACE:
            pref = prefill_workspace_request_ids.contiguous()
            ws = prefill_workspace_starts.contiguous()
        else:
            pref = req
            ws = req
        rows = 2
        programs = min(triton.cdiv(t, rows), self._num_cores)
        _convert_req_index_to_global_index_kernel[(programs,)](
            req,
            table,
            tok,
            out,
            pref,
            ws,
            T=t,
            K=num_topk_tokens,
            M=block_table.shape[1],
            P=programs,
            LOG2_BS=block_size.bit_length() - 1,
            BLOCK_N=triton.next_power_of_2(num_topk_tokens),
            BLOCK_BT=max(2, triton.next_power_of_2(block_table.shape[1])),
            ROWS=rows,
            CHK=1024,
            HAS_PREFILL=bool(HAS_PREFILL_WORKSPACE),
        )
        return out


def _check_widths(block_table: torch.Tensor, token_indices: torch.Tensor) -> None:
    # The kernel keeps a whole page-table row and token tile in unified buffer.
    if token_indices.shape[1] > 2048 or block_table.shape[1] > 4096:
        raise ValueError("Ascend A2 Triton requires K <= 2048 and M <= 4096")


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
    _check_widths(block_table, token_indices)
    return torch.empty_like(token_indices, memory_format=torch.contiguous_format)
