import torch
import torch.nn as nn
import torch_npu  # noqa: F401  (registers the npu backend)
import triton
import triton.language as tl
import triton.runtime.driver as driver


# A5 index remap: scalar-prefetch path for small T and medium K>512;
# vectorized token batches for the other workloads. A measured single-wave
# K<=128 path uses TPI=16; TPI=8 is used for larger K<=512 workloads
# only when T>=16*vectorcore_count; remaining batched workloads use TPI=4.
# Prefill workspace indices are independent of decode page-table capacity;
# token -1 remains invalid on both paths. See report.md for measurements.
@triton.jit
def _convert_req_index_kernel_small(
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
    NUM_PROGS: tl.constexpr,
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

    token_id = pid
    req = tl.load(req_id_ptr + token_id).to(tl.int32)

    tok = tl.load(token_indices_ptr + token_id * ti_stride0 + col * ti_stride1).to(tl.int32)

    row = tl.load(
        block_table_ptr + req * bt_stride0 + row_offs * bt_stride1,
        mask=row_mask,
        other=0,
    ).to(tl.int32)

    is_prefill = False
    workspace_start = 0
    if HAS_PREFILL:
        p_req = tl.load(prefill_ids_ptr + token_id).to(tl.int32)
        is_prefill = p_req.to(tl.float32) >= 0
        workspace_start = tl.load(
            prefill_starts_ptr + tl.maximum(p_req, 0),
            mask=is_prefill,
            other=0,
        ).to(tl.int32)

    block_id = tok >> LOG2_BS
    inblock_off = tok & BS_MASK

    gather_idx = block_id & (MB_P2 - 1)
    base = tl.gather(row, gather_idx, axis=0)

    out_val = (base << LOG2_BS) + inblock_off

    if HAS_PREFILL:
        prefill_val = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_val, out_val)

    invalid = block_id.to(tl.uint32) >= max_blocks
    if HAS_PREFILL:
        invalid = tl.where(is_prefill, tok == -1, invalid)
    out_val = tl.where(invalid, -1, out_val)

    tl.store(
        out_ptr + token_id * out_stride0 + col * out_stride1,
        out_val,
    )


@triton.jit
def _convert_req_index_kernel_per_token(
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
    NUM_PROGS: tl.constexpr,
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

    r0 = tl.load(req_id_ptr + pid, mask=pid < num_tokens, other=0)
    r1 = tl.load(req_id_ptr + pid + NUM_PROGS, mask=pid + NUM_PROGS < num_tokens, other=0)
    r2 = tl.load(req_id_ptr + pid + 2 * NUM_PROGS, mask=pid + 2 * NUM_PROGS < num_tokens, other=0)
    r3 = tl.load(req_id_ptr + pid + 3 * NUM_PROGS, mask=pid + 3 * NUM_PROGS < num_tokens, other=0)

    for token_id in range(pid, num_tokens, NUM_PROGS):
        req = r0.to(tl.int32)
        r0 = r1
        r1 = r2
        r2 = r3
        r3 = tl.load(
            req_id_ptr + token_id + 4 * NUM_PROGS,
            mask=token_id + 4 * NUM_PROGS < num_tokens,
            other=0,
        )

        tok = tl.load(token_indices_ptr + token_id * ti_stride0 + col * ti_stride1).to(tl.int32)

        row = tl.load(
            block_table_ptr + req * bt_stride0 + row_offs * bt_stride1,
            mask=row_mask,
            other=0,
        ).to(tl.int32)

        is_prefill = False
        workspace_start = 0
        if HAS_PREFILL:
            p_req = tl.load(prefill_ids_ptr + token_id).to(tl.int32)
            is_prefill = p_req.to(tl.float32) >= 0
            workspace_start = tl.load(
                prefill_starts_ptr + tl.maximum(p_req, 0),
                mask=is_prefill,
                other=0,
            ).to(tl.int32)

        block_id = tok >> LOG2_BS
        inblock_off = tok & BS_MASK

        gather_idx = block_id & (MB_P2 - 1)
        base = tl.gather(row, gather_idx, axis=0)

        out_val = (base << LOG2_BS) + inblock_off

        if HAS_PREFILL:
            prefill_val = workspace_start + tok
            out_val = tl.where(is_prefill, prefill_val, out_val)

        invalid = block_id.to(tl.uint32) >= max_blocks
        if HAS_PREFILL:
            invalid = tl.where(is_prefill, tok == -1, invalid)
        out_val = tl.where(invalid, -1, out_val)

        tl.store(
            out_ptr + token_id * out_stride0 + col * out_stride1,
            out_val,
        )


@triton.jit
def _convert_req_index_tile(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    prefill_ids_ptr,
    prefill_starts_ptr,
    tokens,
    tmask,
    max_blocks: tl.constexpr,
    MB_P2: tl.constexpr,
    LOG2_BS: tl.constexpr,
    BS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TPI: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
    EXACT: tl.constexpr,
    bt_stride0: tl.constexpr,
    bt_stride1: tl.constexpr,
    ti_stride0: tl.constexpr,
    ti_stride1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
):
    lanes = tl.arange(0, TPI)
    row_offs = tl.arange(0, MB_P2)
    col = tl.arange(0, BLOCK_N)
    if EXACT:
        reqs = tl.load(req_id_ptr + tokens).to(tl.int32)
        tok = tl.load(token_indices_ptr + tokens[:, None] * ti_stride0 + col[None, :] * ti_stride1).to(tl.int32)
    else:
        reqs = tl.load(req_id_ptr + tokens, mask=tmask, other=0).to(tl.int32)
        tok = tl.load(
            token_indices_ptr + tokens[:, None] * ti_stride0 + col[None, :] * ti_stride1, mask=tmask[:, None], other=0
        ).to(tl.int32)
    if EXACT:
        if MB_P2 == max_blocks:
            rows = tl.load(block_table_ptr + reqs[:, None] * bt_stride0 + row_offs[None, :] * bt_stride1).to(tl.int32)
        else:
            rows = tl.load(
                block_table_ptr + reqs[:, None] * bt_stride0 + row_offs[None, :] * bt_stride1,
                mask=row_offs[None, :] < max_blocks,
                other=0,
            ).to(tl.int32)
    else:
        rows = tl.load(
            block_table_ptr + reqs[:, None] * bt_stride0 + row_offs[None, :] * bt_stride1,
            mask=tmask[:, None] & (row_offs[None, :] < max_blocks),
            other=0,
        ).to(tl.int32)
    is_prefill = tl.zeros([TPI], dtype=tl.int1)
    workspace_start = tl.zeros([TPI], dtype=tl.int32)
    if HAS_PREFILL:
        if EXACT:
            p_req = tl.load(prefill_ids_ptr + tokens).to(tl.int32)
        else:
            p_req = tl.load(prefill_ids_ptr + tokens, mask=tmask, other=-1).to(tl.int32)
        is_prefill = p_req.to(tl.float32) >= 0
        workspace_start = tl.load(prefill_starts_ptr + tl.maximum(p_req, 0), mask=is_prefill & tmask, other=0).to(
            tl.int32
        )
    block_id = tok >> LOG2_BS
    inblock_off = tok & BS_MASK
    gather_idx = block_id & (MB_P2 - 1)
    rows_flat = tl.reshape(rows, [TPI * MB_P2])
    idx_flat = tl.reshape(gather_idx + (lanes * MB_P2)[:, None], [TPI * BLOCK_N])
    base = tl.reshape(tl.gather(rows_flat, idx_flat, axis=0), [TPI, BLOCK_N])
    out_val = (base << LOG2_BS) + inblock_off
    if HAS_PREFILL:
        out_val = tl.where(is_prefill[:, None], workspace_start[:, None] + tok, out_val)
    invalid = block_id.to(tl.uint32) >= max_blocks
    if HAS_PREFILL:
        invalid = tl.where(is_prefill[:, None], tok == -1, invalid)
    out_val = tl.where(invalid, -1, out_val)
    if EXACT:
        tl.store(out_ptr + tokens[:, None] * out_stride0 + col[None, :] * out_stride1, out_val)
    else:
        tl.store(out_ptr + tokens[:, None] * out_stride0 + col[None, :] * out_stride1, out_val, mask=tmask[:, None])


@triton.jit
def _convert_req_index_kernel_batched(
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
    lanes = tl.arange(0, TPI)
    # Full interior batches are provably in range; only the last wave needs masks.
    for batch in range(0, num_tokens // (NUM_PROGS * TPI)):
        tokens = (batch * NUM_PROGS + pid) * TPI + lanes
        _convert_req_index_tile(
            req_id_ptr,
            block_table_ptr,
            token_indices_ptr,
            out_ptr,
            prefill_ids_ptr,
            prefill_starts_ptr,
            tokens,
            tl.full([TPI], True, tl.int1),
            max_blocks,
            MB_P2,
            LOG2_BS,
            BS_MASK,
            BLOCK_N,
            TPI,
            HAS_PREFILL,
            True,
            bt_stride0,
            bt_stride1,
            ti_stride0,
            ti_stride1,
            out_stride0,
            out_stride1,
        )
    if num_tokens - (num_tokens // (NUM_PROGS * TPI)) * NUM_PROGS * TPI > 0:
        tokens = ((num_tokens // (NUM_PROGS * TPI)) * NUM_PROGS + pid) * TPI + lanes
        _convert_req_index_tile(
            req_id_ptr,
            block_table_ptr,
            token_indices_ptr,
            out_ptr,
            prefill_ids_ptr,
            prefill_starts_ptr,
            tokens,
            tokens < num_tokens,
            max_blocks,
            MB_P2,
            LOG2_BS,
            BS_MASK,
            BLOCK_N,
            TPI,
            HAS_PREFILL,
            False,
            bt_stride0,
            bt_stride1,
            ti_stride0,
            ti_stride1,
            out_stride0,
            out_stride1,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        self.num_cores = properties["num_vectorcore"]

    def forward(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        has_prefill: bool,
        prefill_workspace_request_ids: torch.Tensor = None,
        prefill_workspace_starts: torch.Tensor = None,
    ) -> torch.Tensor:
        num_tokens = req_id.shape[0]
        max_blocks = block_table.shape[1]

        req_id_c = req_id.contiguous()
        block_table_c = block_table.contiguous()
        token_indices_c = token_indices.contiguous()

        if has_prefill:
            assert prefill_workspace_request_ids is not None
            assert prefill_workspace_starts is not None
            prefill_ids = prefill_workspace_request_ids.contiguous()
            prefill_starts = prefill_workspace_starts.contiguous()
        else:
            prefill_ids = req_id_c
            prefill_starts = req_id_c

        out = torch.empty_like(token_indices_c)

        block_n = num_topk_tokens
        mb_p2 = triton.next_power_of_2(max_blocks)
        log2_bs = block_size.bit_length() - 1
        assert (1 << log2_bs) == block_size

        num_progs = min(num_tokens, self.num_cores)
        grid = (num_progs,)

        common = (
            req_id_c,
            block_table_c,
            token_indices_c,
            out,
            prefill_ids,
            prefill_starts,
            num_tokens,
            max_blocks,
            mb_p2,
            log2_bs,
            block_size - 1,
            block_n,
        )

        use_batched = (num_tokens >= self.num_cores) and (block_n <= 512 or num_tokens >= 128 * self.num_cores)

        if num_tokens <= self.num_cores:
            # Exactly one live token per program: no loop or speculative prefetch.
            _convert_req_index_kernel_small[grid](
                *common,
                num_progs,
                has_prefill,
                block_table_c.stride(0),
                block_table_c.stride(1),
                token_indices_c.stride(0),
                token_indices_c.stride(1),
                out.stride(0),
                out.stride(1),
            )
        elif use_batched:
            # Measured single-wave K<=128 specialization; retain the default
            # on larger workloads and on partial 16-token tiles.
            if block_n <= 128 and 4 * self.num_cores <= num_tokens <= 16 * self.num_cores and num_tokens % 16 == 0:
                tpi = 16
            else:
                tpi = 8 if block_n <= 512 and num_tokens >= 16 * self.num_cores else 4
            num_progs = min(triton.cdiv(num_tokens, tpi), self.num_cores)
            grid = (num_progs,)
            _convert_req_index_kernel_batched[grid](
                *common,
                num_progs,
                tpi,
                has_prefill,
                block_table_c.stride(0),
                block_table_c.stride(1),
                token_indices_c.stride(0),
                token_indices_c.stride(1),
                out.stride(0),
                out.stride(1),
            )
        else:
            _convert_req_index_kernel_per_token[grid](
                *common,
                num_progs,
                has_prefill,
                block_table_c.stride(0),
                block_table_c.stride(1),
                token_indices_c.stride(0),
                token_indices_c.stride(1),
                out.stride(0),
                out.stride(1),
            )
        return out


def convert_req_index_to_global_index_fwd(
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
