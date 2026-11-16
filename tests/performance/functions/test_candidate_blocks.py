import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training

# Real DSA workloads: packed offline batches plus decode (shared bounds) and
# 8192-token prefill rows over a 128K packed column span.
CASES = [
    ("packed", 6, 32768, 8, 2048, 1),
    ("decode-n1", 256, 131072, 8, 2048, 1),
    ("decode-n6", 1536, 131072, 8, 2048, 6),
    ("prefill", 8192, 131072, 8, 2048, 1),
]


def _bounds(rows, width, row_repeat, device):
    packed = rows // row_repeat
    ends = torch.randint(width // 2, width + 1, (packed,), device=device).to(torch.int32)
    starts = torch.zeros(packed, device=device, dtype=torch.int32) if row_repeat == 1 else None
    return starts, ends


@pytest.mark.api("functions.select_candidate_blocks")
@pytest.mark.parametrize("label,rows,width,block_size,topk_blocks,row_repeat", CASES)
def test_select_candidate_blocks(benchmark, perf_environment, label, rows, width, block_size, topk_blocks, row_repeat):
    _, device, _, implementation = perf_environment

    def factory():
        logits = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
        row_ks, row_ke = _bounds(rows, width, row_repeat, device)
        out = torch.empty(rows, topk_blocks, device=device, dtype=torch.int32)
        call = lambda: functions.select_candidate_blocks(
            logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat, implementation=implementation
        )
        return training(call, (), "forward")

    benchmark(
        factory=factory,
        op="select_candidate_blocks",
        shape=[rows, width],
        dtype="bfloat16",
        phase="forward",
        block_size=block_size,
        topk_blocks=topk_blocks,
        row_repeat=row_repeat,
        label=label,
    )


@pytest.mark.api("functions.apply_candidate_mask")
@pytest.mark.parametrize("label,rows,width,block_size,topk_blocks,row_repeat", CASES)
def test_apply_candidate_mask(benchmark, perf_environment, label, rows, width, block_size, topk_blocks, row_repeat):
    _, device, _, implementation = perf_environment

    def factory():
        logits = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
        row_ks, row_ke = _bounds(rows, width, row_repeat, device)
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
        dtype="bfloat16",
        phase="forward",
        block_size=block_size,
        topk_blocks=topk_blocks,
        row_repeat=row_repeat,
        label=label,
    )
