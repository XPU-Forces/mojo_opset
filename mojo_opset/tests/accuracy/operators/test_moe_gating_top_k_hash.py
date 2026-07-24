import pytest
import torch

from mojo_opset import MojoMoEGatingTopKHash
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


@pytest.mark.parametrize(
    "rows, experts, k, vocab_size, norm_type",
    [
        (16, 256, 6, 100, 1),
        (32, 256, 6, 100, 1),
        (4, 64, 4, 20, 1),
        (1, 128, 8, 50, 1),
        (32, 64, 64, 100, 1),
        (8, 128, 1, 50, 1),
        (16, 384, 6, 150, 1),
        (16, 512, 8, 200, 1),
        (16, 256, 6, 100, 0),
        (16, 256, 6, 100, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@bypass_not_implemented
def test_moe_gating_top_k_hash(rows, experts, k, vocab_size, norm_type, dtype):
    device = get_torch_device()
    torch.manual_seed(42)
    op = MojoMoEGatingTopKHash(
        k=k,
        routed_scaling_factor=1.0,
        eps=1e-6,
        norm_type=norm_type,
        out_flag=False,
        device=device,
        dtype=dtype,
    )
    op_ref = MojoMoEGatingTopKHash._registry.get("torch")(
        k=k,
        routed_scaling_factor=1.0,
        eps=1e-6,
        norm_type=norm_type,
        out_flag=False,
        device=device,
        dtype=dtype,
    )

    x = torch.empty(rows, experts).uniform_(-2, 2).to(dtype).to(device)
    input_ids = torch.randint(0, vocab_size, (rows,), dtype=torch.int64).to(device)
    tid2eid = torch.stack(
        [torch.randperm(experts, dtype=torch.int32)[:k] for _ in range(vocab_size)]
    ).to(device)

    op.forward_diff_with(op_ref, x, input_ids, tid2eid, mixed_tol=True)


@pytest.mark.parametrize(
    "rows, experts, k, vocab_size",
    [(16, 256, 6, 100), (32, 384, 6, 150)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@bypass_not_implemented
def test_moe_gating_top_k_hash_out_flag(rows, experts, k, vocab_size, dtype):
    device = get_torch_device()
    torch.manual_seed(42)

    for norm_type in (0, 1, 2):
        op = MojoMoEGatingTopKHash(
            k=k,
            norm_type=norm_type,
            out_flag=True,
            device=device,
            dtype=dtype,
        )
        op_ref = MojoMoEGatingTopKHash._registry.get("torch")(
            k=k,
            norm_type=norm_type,
            out_flag=True,
            device=device,
            dtype=dtype,
        )
        x = torch.empty(rows, experts, device=device, dtype=dtype).uniform_(-2, 2)
        input_ids = torch.randint(
            0,
            vocab_size,
            (rows,),
            dtype=torch.int64,
            device=device,
        )
        tid2eid = torch.stack(
            [torch.randperm(experts, dtype=torch.int32)[:k] for _ in range(vocab_size)]
        ).to(device)

        op.forward_diff_with(op_ref, x, input_ids, tid2eid, mixed_tol=True)


@pytest.mark.parametrize(
    "rows, experts, k, vocab_size, scaling_factor",
    [
        (16, 256, 6, 100, 1.0),
        (16, 256, 6, 100, 2.5),
        (16, 256, 6, 100, 0.5),
    ],
)
@bypass_not_implemented
def test_moe_gating_top_k_hash_scaling(rows, experts, k, vocab_size, scaling_factor):
    device = get_torch_device()
    torch.manual_seed(42)
    dtype = torch.bfloat16
    op = MojoMoEGatingTopKHash(
        k=k,
        routed_scaling_factor=scaling_factor,
        device=device,
        dtype=dtype,
    )
    op_ref = MojoMoEGatingTopKHash._registry.get("torch")(
        k=k,
        routed_scaling_factor=scaling_factor,
        device=device,
        dtype=dtype,
    )
    x = torch.empty(rows, experts, device=device, dtype=dtype).uniform_(-2, 2)
    input_ids = torch.randint(0, vocab_size, (rows,), dtype=torch.int64, device=device)
    tid2eid = torch.stack(
        [torch.randperm(experts, dtype=torch.int32)[:k] for _ in range(vocab_size)]
    ).to(device)

    op.forward_diff_with(op_ref, x, input_ids, tid2eid, mixed_tol=True)
