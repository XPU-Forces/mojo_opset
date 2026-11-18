"""get_compressed_slot_mapping performance cases (merged / split / decode shapes)."""

import pytest
import torch

from mojo_opset import functions

# (name, query_lens, seq_upper, block_size, compress_ratio)
CASES = [
    ("decode_64x1_cr2", [1] * 64, 4096, 64, 2),
    ("decode_256x1_cr2", [1] * 256, 8192, 64, 2),
    ("decode_2048x1_cr2", [1] * 2048, 8192, 64, 2),
    ("decode_256x4_cr2", [4] * 256, 8192, 64, 2),
    ("prefill_1x1024_cr2", [1024], 4096, 64, 2),
    ("mixed_2x2048_cr2", [2048] * 2, 8192, 64, 2),
    ("prefill_1x8192_cr2", [8192], 16384, 64, 2),
    ("mixed_4x2048_cr4", [2048] * 4, 16384, 64, 4),
]


@pytest.mark.api("functions.get_compressed_slot_mapping")
@pytest.mark.parametrize("name,query_lens,seq_upper,block_size,compress_ratio", CASES, ids=[c[0] for c in CASES])
def test_get_compressed_slot_mapping(
    benchmark, perf_environment, name, query_lens, seq_upper, block_size, compress_ratio
):
    _, device, _, implementation = perf_environment
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.cumsum(
        torch.tensor(query_lens, dtype=torch.int64, device=device), 0
    ).to(torch.int32)
    seq_lens = torch.full((num_reqs,), seq_upper, dtype=torch.int32, device=device)
    width = (seq_upper // compress_ratio + block_size) // block_size
    block_table = torch.randint(0, 4096, (num_reqs, width), dtype=torch.int32, device=device)
    slot_mapping = torch.randint(0, 1 << 30, (num_tokens,), dtype=torch.int64, device=device)

    def factory():
        return lambda: functions.get_compressed_slot_mapping(
            num_tokens,
            slot_mapping,
            query_start_loc,
            seq_lens,
            block_table,
            block_size,
            compress_ratio,
            None,
            1,
            1,
            max(query_lens),
            implementation=implementation,
        )

    benchmark(
        factory=factory,
        op="get_compressed_slot_mapping",
        case=name,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        block_size=block_size,
        compress_ratio=compress_ratio,
        dtype="int64",
        phase="forward",
    )
