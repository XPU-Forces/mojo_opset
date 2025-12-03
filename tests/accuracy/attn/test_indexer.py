import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoLightningIndexer


@pytest.mark.parametrize(
    "batch, q_seq_len, k_seq_len, q_head_num, k_head_num, head_dim, dummy_tensor",
    [
        (
            batch,
            q_seq_len,
            8192,  #
            q_head_num,
            1,
            128,
            torch.randn(1),
        )
        for batch in [1, 2, 8, 32, 128]
        for q_seq_len in [1, 128, 1024, 4096, 8192]
        for q_head_num in [128, 64]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_lightning_indexer(batch, q_seq_len, k_seq_len, q_head_num, k_head_num, head_dim, dummy_tensor):
    device = dummy_tensor.device
    query = torch.randn(
        batch,
        q_seq_len,
        q_head_num,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    query_scale = torch.randn(batch, q_seq_len, q_head_num, device=device, dtype=torch.float32)
    key = torch.randn(
        batch,
        k_seq_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    mask = torch.full((q_seq_len, q_seq_len), float("-inf"), device=device).triu_(1) if q_seq_len > 1 else None

    op = MojoLightningIndexer()
    op.forward_diff(query, query_scale, key, None, mask)
