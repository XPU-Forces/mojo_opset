import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training

# Real DSA workloads: packed offline batches plus decode (shared bounds) and
# 8192-token prefill rows over a 128K packed column span.
# Columns: label, rows, width, block_size, topk_blocks, row_repeat, dtype,
#          stride_col, ends_mode
CASES = [
    ("packed", 6, 32768, 8, 2048, 1, torch.bfloat16, 1, "tail"),
    ("decode-n1", 256, 131072, 8, 2048, 1, torch.bfloat16, 1, "tail"),
    ("decode-n6", 1536, 131072, 8, 2048, 6, torch.bfloat16, 1, "tail"),
    ("prefill", 8192, 131072, 8, 2048, 1, torch.bfloat16, 1, "tail"),
    # Small-batch decode: 8 rows of 16K blocks, i.e. the row count no longer
    # supplies the parallelism and the TILE/split ladder has to.
    ("decode-b8", 8, 131072, 8, 2048, 1, torch.bfloat16, 1, "tail"),
    # Narrow decode: 4K blocks lands exactly on the narrow-row tier boundary
    # (nblocks <= 4096), which a 128K/256K column span never reaches.
    ("decode-32k", 256, 32768, 8, 2048, 1, torch.bfloat16, 1, "tail"),
    # fp32 rows keep 32 B per block, so the narrow-row tier and the paired
    # bf16 load must both stay off; guards the dtype-dependent launch rules.
    ("packed-fp32", 6, 32768, 8, 2048, 1, torch.float32, 1, "tail"),
    ("decode-n1-fp32", 256, 131072, 8, 2048, 1, torch.float32, 1, "tail"),
    # Non-unit column stride, i.e. a view into a wider buffer: the kernels
    # take the strided branch instead of the affine tile path.
    ("decode-n1-strided", 256, 131072, 8, 2048, 1, torch.bfloat16, 2, "tail"),
    # Bounds drawn from [0, width] instead of [width/2, width]: most rows are
    # short or empty, which exercises the dead/edge tile paths.
    ("decode-n1-mixed", 256, 131072, 8, 2048, 1, torch.bfloat16, 1, "mixed"),
]


def _bounds(rows, width, row_repeat, device, ends_mode="tail"):
    packed = rows // row_repeat
    lo = width // 2 if ends_mode == "tail" else 0
    ends = torch.randint(lo, width + 1, (packed,), device=device).to(torch.int32)
    starts = torch.zeros(packed, device=device, dtype=torch.int32) if row_repeat == 1 else None
    return starts, ends


def _logits(rows, width, dtype, stride_col, device):
    if stride_col == 1:
        return torch.randn(rows, width, device=device, dtype=dtype)
    base = torch.randn(rows, width * stride_col, device=device, dtype=dtype)
    return base[:, ::stride_col]


def _dtype_name(dtype):
    return str(dtype).split(".")[-1]


@pytest.mark.api("functions.select_candidate_blocks")
@pytest.mark.parametrize("label,rows,width,block_size,topk_blocks,row_repeat,dtype,stride_col,ends_mode", CASES)
def test_select_candidate_blocks(
    benchmark, perf_environment, label, rows, width, block_size, topk_blocks, row_repeat, dtype, stride_col, ends_mode
):
    _, device, _, implementation = perf_environment

    def factory():
        logits = _logits(rows, width, dtype, stride_col, device)
        row_ks, row_ke = _bounds(rows, width, row_repeat, device, ends_mode)
        out = torch.empty(rows, topk_blocks, device=device, dtype=torch.int32)
        call = lambda: functions.select_candidate_blocks(
            logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat, implementation=implementation
        )
        return training(call, (), "forward")

    benchmark(
        factory=factory,
        op="select_candidate_blocks",
        shape=[rows, width],
        dtype=_dtype_name(dtype),
        phase="forward",
        block_size=block_size,
        topk_blocks=topk_blocks,
        row_repeat=row_repeat,
        stride_col=stride_col,
        ends_mode=ends_mode,
        label=label,
    )


@pytest.mark.api("functions.apply_candidate_mask")
@pytest.mark.parametrize("label,rows,width,block_size,topk_blocks,row_repeat,dtype,stride_col,ends_mode", CASES)
def test_apply_candidate_mask(
    benchmark, perf_environment, label, rows, width, block_size, topk_blocks, row_repeat, dtype, stride_col, ends_mode
):
    _, device, _, implementation = perf_environment

    def factory():
        logits = _logits(rows, width, dtype, stride_col, device)
        row_ks, row_ke = _bounds(rows, width, row_repeat, device, ends_mode)
        nblocks = -(-width // block_size)
        candidates = torch.randint(0, nblocks, (rows, topk_blocks), device=device, dtype=torch.int32)
        call = lambda: functions.apply_candidate_mask(
            logits, row_ks, row_ke, candidates, block_size, row_repeat, implementation=implementation
        )
        return training(call, (), "forward")

    benchmark(
        factory=factory,
        op="apply_candidate_mask",
        shape=[rows, width],
        dtype=_dtype_name(dtype),
        phase="forward",
        block_size=block_size,
        topk_blocks=topk_blocks,
        row_repeat=row_repeat,
        stride_col=stride_col,
        ends_mode=ends_mode,
        label=label,
    )
