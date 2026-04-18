import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset.experimental import MojoStoreLowrank

kv_lens = [1, 24, 1024, 2048, 4096, 8192, 13312]
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
    "shape_label_cache, shape_key_lr",
    [
        ((256, 1, 512, 128), (1, 128)),
        ((256, 8, 512, 128), (8, 128)),
    ],
)
@pytest.mark.parametrize(
    "kv_len",
    [1024, 2048, 4096, 8192, 13312],
)
@bypass_not_implemented
def test_store_lowrank(shape_label_cache, shape_key_lr, kv_len):
    slot_mapping = torch.randperm(kv_len)
    label_cache = torch.zeros(size=shape_label_cache, dtype=torch.bfloat16)
    key_lr = torch.randn(size=(slot_mapping.shape[0], *shape_key_lr), dtype=torch.bfloat16)
    block_idxs = slot_mapping // 512
    token_idxs = slot_mapping % 512
    token_num = slot_mapping.shape[0]

    store_lowrank = MojoStoreLowrank()

    store_lowrank_ref = MojoStoreLowrank._registry.get("torch")()

    store_lowrank.forward_diff_with(store_lowrank_ref, label_cache, key_lr, block_idxs, token_idxs, token_num)
