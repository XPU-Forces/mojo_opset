"""opt_iter_3：opt_iter_2 + 尾部无掩码快路径 + 单条无效判定。

相对 opt_iter_2：
- EXACT 分支：当每个 program 的行区间长度都是 ROWS 的整数倍时（chunk % ROWS == 0
  且 num_tokens % ROWS == 0），去掉 req/token/out 的所有 r_mask，load/store 全宽展开
- MAX_BLOCKS 为 2 的幂时，无效判定合并为 `(block_id & -MAX_BLOCKS) != 0`
  （负 bid 与越界 bid 都会得到非 0，一条 and + 一条 cmp）
- ROWS 由 host 按 BLOCK_N 选择：BLOCK_N<=1024 用 4，否则用 2（控制 UB）

精度要求：22/22 全过才允许采用。
"""

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
    num_tokens,
    num_programs,
    LOG2_BS: tl.constexpr,
    BS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, BLOCK_N)
    bt_cols = tl.arange(0, BLOCK_BT)
    r_off = tl.arange(0, ROWS)

    chunk = tl.cdiv(tl.cdiv(num_tokens, num_programs), ROWS) * ROWS
    start = pid * chunk
    end = tl.minimum(start + chunk, num_tokens)

    for row0 in range(start, end, ROWS):
        rows = row0 + r_off

        if EXACT:
            reqs = tl.load(req_id_ptr + rows).to(tl.int32)
            if HAS_PREFILL:
                prefs = tl.load(prefill_req_ptr + rows).to(tl.int32)
                is_pref = prefs >= 0
                ws = tl.load(workspace_ptr + tl.maximum(prefs, 0), mask=is_pref, other=0).to(tl.int32)
            if MAX_BLOCKS == BLOCK_BT:
                bt_2d = tl.load(block_table_ptr + reqs[:, None] * MAX_BLOCKS + bt_cols[None, :])
            else:
                bt_2d = tl.load(
                    block_table_ptr + reqs[:, None] * MAX_BLOCKS + bt_cols[None, :],
                    mask=bt_cols[None, :] < MAX_BLOCKS,
                    other=0,
                )
            tok = tl.load(token_indices_ptr + rows[:, None] * BLOCK_N + cols[None, :])
        else:
            r_mask = rows < end
            reqs = tl.load(req_id_ptr + rows, mask=r_mask, other=0).to(tl.int32)
            if HAS_PREFILL:
                prefs = tl.load(prefill_req_ptr + rows, mask=r_mask, other=-1).to(tl.int32)
                is_pref = prefs >= 0
                ws = tl.load(workspace_ptr + tl.maximum(prefs, 0), mask=is_pref, other=0).to(tl.int32)
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
            tok = tl.load(
                token_indices_ptr + rows[:, None] * BLOCK_N + cols[None, :],
                mask=r_mask[:, None],
                other=-1,
            )

        block_id = tok >> LOG2_BS
        inblock = tok & BS_MASK
        base = tl.gather(bt_2d, block_id & (BLOCK_BT - 1), 1)

        out_val = (base << LOG2_BS) + inblock
        if MAX_BLOCKS & (MAX_BLOCKS - 1) == 0:
            invalid = (block_id & (0 - MAX_BLOCKS)) != 0
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
        progs = min(num_tokens, cores)
        rows = 4
        chunk = ((num_tokens + progs - 1) // progs + rows - 1) // rows * rows
        exact = (chunk % rows == 0) and (num_tokens % rows == 0)
        grid = (progs,)

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
            BS_MASK=block_size - 1,
            BLOCK_N=block_n,
            BLOCK_BT=block_bt,
            MAX_BLOCKS=max_blocks,
            ROWS=rows,
            EXACT=exact,
            HAS_PREFILL=bool(HAS_PREFILL_WORKSPACE),
        )
        return out


def triton_convert_req_index_to_global_index_fwd(
    req_id,
    block_table,
    token_indices,
    block_size,
    num_topk_tokens,
    has_prefill,
    prefill_workspace_request_ids,
    prefill_workspace_starts,
):
    return ModelNew()(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
        has_prefill,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
    )
