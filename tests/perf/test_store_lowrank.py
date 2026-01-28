import pytest
import torch

from mojo_opset.experimental import MojoStoreLowrank
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

kv_lens = [1024, 2048, 4096, 8192, 13312]
slot_mappings = [torch.randperm(kv_len) for kv_len in kv_lens]

shapes_label_cache = [
    (256, 1, 512, 128),
    (256, 8, 512, 128),
]

shapes_key_lr = [
    (1, 128),
    (8, 128),
]


@pytest.mark.parametrize(
    "label_cache, key_lr, block_idxs, token_idxs, token_num",
    [
        (
            torch.zeros(size=shape0, dtype=torch.bfloat16),
            torch.randn(size=(slot_mapping.shape[0], *shape1), dtype=torch.bfloat16),
            slot_mapping // 512,
            slot_mapping % 512,
            slot_mapping.shape[0],
        )
        for shape0, shape1 in zip(shapes_label_cache, shapes_key_lr)
        for slot_mapping in slot_mappings
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_store_lowrank(label_cache, key_lr, block_idxs, token_idxs, token_num):
    store_lowrank = MojoStoreLowrank()

    perf(lambda: store_lowrank(label_cache, key_lr, block_idxs, token_idxs, token_num))
