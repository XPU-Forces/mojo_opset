import math

import pytest
import torch

from mojo_opset.experimental import mojo_dllm_attention
from mojo_opset.experimental import mojo_dllm_attention_up
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented


class DllmAttentionPerf:
    def __init__(self, scale: float, block_size: int):
        self.scale = scale
        self.block_size = block_size

    def __call__(self, query, key, value, cu_seqlen):
        return mojo_dllm_attention(query, key, value, cu_seqlen, self.scale, self.block_size)


class DllmAttentionUpPerf:
    def __init__(self, scale: float, block_size: int):
        self.scale = scale
        self.block_size = block_size

    def __call__(self, query, key, value, cu_seqlen):
        return mojo_dllm_attention_up(query, key, value, cu_seqlen, self.scale, self.block_size)


DLLM_UP_CONFIGS = [
    {
        "id": "5seq_32768",
        "seqlens": [28951, 3542, 99, 128, 48],
        "q_head_num": 5,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 8,
    },
    {
        "id": "6seq_32762",
        "seqlens": [32466, 69, 90, 41, 37, 59],
        "q_head_num": 5,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 8,
    },
    {
        "id": "20seq_32761",
        "seqlens": [
            931, 745, 1608, 2149, 433, 16814, 268, 207, 2193, 4278,
            606, 254, 128, 192, 254, 1255, 182, 177, 61, 26,
        ],
        "q_head_num": 5,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 8,
    },
    {
        "id": "58seq_32768",
        "seqlens": [
            843, 50, 118, 71, 805, 325, 578, 199, 151, 74, 478, 275,
            101, 193, 89, 82, 340, 110, 118, 1010, 1463, 1226, 491, 638,
            603, 157, 3754, 530, 1233, 608, 797, 2788, 265, 486, 472, 1482,
            573, 84, 489, 1381, 148, 207, 777, 258, 1094, 133, 676, 137,
            88, 508, 455, 365, 234, 155, 1024, 97, 792, 58,
        ],
        "q_head_num": 5,
        "kv_head_num": 1,
        "head_dim": 128,
        "block_size": 8,
    },
]


def generate_dllm_up_test_data(config: dict, device: str = "npu"):
    seqlens = config["seqlens"]
    q_head_num = config["q_head_num"]
    kv_head_num = config["kv_head_num"]
    head_dim = config["head_dim"]
    block_size = config["block_size"]

    total_seq = sum(seqlens)
    cu_seqlen = torch.tensor(
        [sum(seqlens[: i + 1]) for i in range(len(seqlens))],
        dtype=torch.int32,
        device=device,
    )

    query = torch.randn(total_seq * 2, q_head_num, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(total_seq * 2, kv_head_num, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(total_seq * 2, kv_head_num, head_dim, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    return query, key, value, cu_seqlen, scale, block_size


@pytest.mark.parametrize(
    "query, key, value, cu_seqlen, scale, block_size",
    [
        pytest.param(
            *generate_dllm_up_test_data(cfg),
            id=cfg["id"],
        )
        for cfg in DLLM_UP_CONFIGS
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_dllm_attention_up(query, key, value, cu_seqlen, scale, block_size):
    op = DllmAttentionUpPerf(scale, block_size)
    perf(lambda: op(query, key, value, cu_seqlen))  # noqa: F821


@pytest.mark.parametrize(
    "query, key, value, cu_seqlen, scale, block_size",
    [
        pytest.param(
            *generate_dllm_up_test_data(cfg),
            id=cfg["id"],
        )
        for cfg in DLLM_UP_CONFIGS
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_dllm_attention(query, key, value, cu_seqlen, scale, block_size):
    op = DllmAttentionPerf(scale, block_size)
    perf(lambda: op(query, key, value, cu_seqlen))  # noqa: F821
