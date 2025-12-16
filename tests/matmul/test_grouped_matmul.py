import pytest
import torch
from mojo_opset.backends.torch_ops.matmul import TorchGroupedMatmul


@pytest.mark.npu
def test_grouped_matmul_accuracy():
    """
    Tests the accuracy of the TorchGroupedMatmul operator by comparing its
    output with a reference implementation.
    """
    grouped_matmul_op = TorchGroupedMatmul()

    # We will simulate two groups of matrix multiplications
    # Group 1: [1, 16, 32] @ [1, 32, 64]
    # Group 2: [1, 8, 16] @ [1, 16, 32]
    input_group1 = torch.randn(1, 16, 32, device="npu")
    other_group1 = torch.randn(1, 32, 64, device="npu")
    input_group2 = torch.randn(1, 8, 16, device="npu")
    other_group2 = torch.randn(1, 16, 32, device="npu")

    # The npu_grouped_matmul expects concatenated tensors
    input_cat = torch.cat([input_group1, input_group2], dim=0)
    other_cat = torch.cat([other_group1, other_group2], dim=0)
    group_info = [[1, 16, 32, 64], [1, 8, 16, 32]]

    try:
        output_std = grouped_matmul_op.forward_std(
            input_cat, other_cat, group_info, trans_input=False, trans_other=False
        )
    except Exception as e:
        pytest.fail(
            f"The torch_npu.npu_grouped_matmul operator failed. "
            f"This might be due to an incorrect 'group_info' format. Error: {e}"
        )

    # For the reference implementation, we'll manually slice and compute
    ref_output1 = torch.matmul(input_group1, other_group1)
    ref_output2 = torch.matmul(input_group2, other_group2)
    output_ref = torch.cat([ref_output1, ref_output2], dim=0)

    assert torch.allclose(
        output_std, output_ref, atol=1e-5, rtol=1e-3
    ), "The output of TorchGroupedMatmul does not match the reference implementation."