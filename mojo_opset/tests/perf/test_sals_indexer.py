import pytest
import torch

from mojo_opset import MojoSALSIndexer
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


HEAD_DIM = 128
SPARSE_BLOCK_SIZE = 64
DEFAULT_SPARSE_RATIO = 0.25
DEFAULT_FIXED_TAIL = 4
MIN_SPARSE_LEN = 512

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

# (q_seqlen, share_len, kv_seqlen, cache_block_size)
PREFILL_SCENARIOS = [
    (8192,   128, 1024, 64),
    (8192,   256, 2048, 256),
    (10240,  256, 1024, 256),
    (16384,  256, 1024, 256),
    (24576,  256, 1024, 256),
    (32768,  256, 2048, 256),
]

# 64k/80k/128k only with lightweight model to avoid OOM
LARGE_PREFILL_SCENARIOS = [
    (65536,  256, 512,  64),
    (81920,  256, 1024, 256),
    (131072, 256, 256,  64),
]

# Custom test cases as specified:
# q_len = 4096, kv_len = 16384/32768/65536, q_head_num = 3/4, kv_head_num = 1, share_len=256, cache_block_size = 64, fixed_tail_blocks = 4
CUSTOM_SCENARIOS = [
    (4096, 3, 1, 256,16384, 64, 4),
    (4096, 3, 1, 256,32768, 64, 4),
    (4096, 3, 1, 256,65536, 64, 4),
    (4096, 4, 1, 256,16384, 64, 4),
    (4096, 4, 1, 256,32768, 64, 4),
    (4096, 4, 1, 256,65536, 64, 4),
]

_CUSTOM_PARAMS = []
for _s in CUSTOM_SCENARIOS:
    q_seqlen, q_head_num, kv_head_num, share_len, kv_seqlen, cache_block_size, fixed_tail = _s
    _CUSTOM_PARAMS.append(pytest.param(
        q_seqlen, q_head_num, kv_head_num, share_len,kv_seqlen, cache_block_size, fixed_tail,
        id=f"q{q_seqlen}-qhead{q_head_num}-kvhead{kv_head_num}-kv{kv_seqlen}-cache_block_size{cache_block_size}-fixed_tail{fixed_tail}",
    ))

_SMALL_MODEL = MODEL_SPECS[0]  # new_model_1


def _generate_sals_indexer_data(G, seq_lengths, kv_heads, *, cache_block_size=64):
    device = "npu" if torch.npu.is_available() else "cpu"
    sbs = SPARSE_BLOCK_SIZE
    max_seqlen = max(seq_lengths) if seq_lengths else 0
    max_bpg = max((max_seqlen + cache_block_size - 1) // cache_block_size, 1)
    num_phys = max_bpg * G + 4

    max_count = max((max_seqlen + sbs - 1) // sbs, 0)
    fixed_tail_count = min(DEFAULT_FIXED_TAIL, max_count)
    sparse_count = max(1, min(
        int((max_count - fixed_tail_count) * DEFAULT_SPARSE_RATIO + 0.5) + fixed_tail_count,
        max_count,
    ))

    query = torch.randn(G, kv_heads, HEAD_DIM, dtype=torch.float16, device=device)
    key = torch.randn(num_phys, cache_block_size, kv_heads, HEAD_DIM, dtype=torch.float16, device=device)

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
@pytest.mark.parametrize("q_seqlen,share_len,kv_seqlen,cache_block_size", PREFILL_SCENARIOS)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_indexer_perf(model_name, kv_heads, q_seqlen, share_len, kv_seqlen, cache_block_size):
    G = (q_seqlen - MIN_SPARSE_LEN) // share_len
    (query, key, block_table, actual_seq_lengths_key, act_n_counts,
     sbs, sparse_ratio, fixed_tail_count, sparse_count,
     score_mode, max_seqlen_key) = _generate_sals_indexer_data(G, [kv_seqlen] * G, kv_heads, cache_block_size=cache_block_size)
    indexer = MojoSALSIndexer()
    perf(lambda: indexer(query, key, block_table, actual_seq_lengths_key, act_n_counts,
                         sbs, sparse_ratio, fixed_tail_count, sparse_count,
                         score_mode, max_seqlen_key))


@pytest.mark.parametrize("q_seqlen,share_len,kv_seqlen,cache_block_size", LARGE_PREFILL_SCENARIOS)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_indexer_large_perf(q_seqlen, share_len, kv_seqlen, cache_block_size):
    _model_name, kv_heads = _SMALL_MODEL
    G = (q_seqlen - MIN_SPARSE_LEN) // share_len
    (query, key, block_table, actual_seq_lengths_key, act_n_counts,
     sbs, sparse_ratio, fixed_tail_count, sparse_count,
     score_mode, max_seqlen_key) = _generate_sals_indexer_data(G, [kv_seqlen] * G, kv_heads, cache_block_size=cache_block_size)
    indexer = MojoSALSIndexer()
    perf(lambda: indexer(query, key, block_table, actual_seq_lengths_key, act_n_counts,
                         sbs, sparse_ratio, fixed_tail_count, sparse_count,
                         score_mode, max_seqlen_key))


@pytest.mark.parametrize(
    "q_seqlen,q_head_num,kv_head_num,share_len,kv_seqlen,cache_block_size,fixed_tail",
    _CUSTOM_PARAMS,
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_indexer_custom_perf(q_seqlen, q_head_num,kv_head_num, share_len, kv_seqlen, cache_block_size, fixed_tail):
    G = (q_seqlen - MIN_SPARSE_LEN) // share_len
    (query, key, block_table, actual_seq_lengths_key, act_n_counts,
     sbs, sparse_ratio, fixed_tail_count, sparse_count,
     score_mode, max_seqlen_key) = _generate_sals_indexer_data(G, [kv_seqlen] * G, kv_head_num, cache_block_size=cache_block_size)
    indexer = MojoSALSIndexer()
    perf(lambda: indexer(query, key, block_table, actual_seq_lengths_key, act_n_counts,
                         sbs, sparse_ratio, fixed_tail_count, sparse_count,
                         "lse", max_seqlen_key))
