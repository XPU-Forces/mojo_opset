import pytest
import torch

from mojo_opset import MojoSALSIndexer
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


HEAD_DIM = 128
SPARSE_BLOCK_SIZE = 16
DEFAULT_SPARSE_RATIO = 0.25
DEFAULT_FIXED_TAIL = 32

MODEL_SPECS = [
    ("new_model_1", 2),
    ("new_model_2", 4),
    ("new_model_3", 4),
    ("new_model_4", 8),
    ("new_model_5", 8),
    ("new_model_6", 16),
    ("M9-23B", 8),
    ("M8-14B", 8),
]

# (q_seqlen, share_len, kv_seqlen)
PREFILL_SCENARIOS = [
    (8192,  128, 1024),
    (8192,  256, 2048),
    (16384, 256, 1024),
    (32768, 256, 2048),
]


def _generate_sals_indexer_data(G, seq_lengths, kv_heads):
    device = "npu" if torch.npu.is_available() else "cpu"
    sbs = SPARSE_BLOCK_SIZE
    max_seqlen = max(seq_lengths) if seq_lengths else 0
    max_bpg = max((max_seqlen + sbs - 1) // sbs, 1)
    num_phys = max_bpg * G + 4

    max_count = max((max_seqlen + sbs - 1) // sbs, 0)
    fixed_tail_count = min(DEFAULT_FIXED_TAIL, max_count)
    sparse_count = max(1, min(
        int((max_count - fixed_tail_count) * DEFAULT_SPARSE_RATIO + 0.5) + fixed_tail_count,
        max_count,
    ))

    query = torch.randn(G, kv_heads, HEAD_DIM, dtype=torch.float16, device=device)
    key = torch.randn(num_phys, sbs, kv_heads, HEAD_DIM, dtype=torch.float16, device=device)

    block_table = torch.zeros(G, max_bpg, dtype=torch.int32, device=device)
    for g in range(G):
        n_blks = min(max_bpg, num_phys)
        perm = torch.randperm(num_phys, device=device)[:n_blks].to(torch.int32)
        block_table[g, :n_blks] = perm
        if n_blks < max_bpg:
            block_table[g, n_blks:] = perm[-1]

    actual_seq_lengths_key = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    act_n_counts = torch.tensor(
        [(max(s, 0) + sbs - 1) // sbs for s in seq_lengths],
        dtype=torch.int32, device=device,
    )

    return (
        query, key, block_table, actual_seq_lengths_key, act_n_counts,
        sbs, DEFAULT_SPARSE_RATIO, fixed_tail_count, sparse_count,
        "lse", max_seqlen,
    )


@pytest.mark.parametrize("model_name,kv_heads", MODEL_SPECS)
@pytest.mark.parametrize("q_seqlen,share_len,kv_seqlen", PREFILL_SCENARIOS)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_indexer_perf(model_name, kv_heads, q_seqlen, share_len, kv_seqlen):
    G = q_seqlen // share_len
    (query, key, block_table, actual_seq_lengths_key, act_n_counts,
     sbs, sparse_ratio, fixed_tail_count, sparse_count,
     score_mode, max_seqlen_key) = _generate_sals_indexer_data(G, [kv_seqlen] * G, kv_heads)
    indexer = MojoSALSIndexer()
    perf(lambda: indexer(query, key, block_table, actual_seq_lengths_key, act_n_counts,
                         sbs, sparse_ratio, fixed_tail_count, sparse_count,
                         score_mode, max_seqlen_key))
