import pytest
import torch

from mojo_opset import MojoGroupedMatmul
from tests.utils import auto_switch_platform, bypass_not_implemented, get_platform

# Test cases: (inputs, weights, bias, group_type, dtype, split_item, group_list, output_dtype)
#  - inputs: List[Tensor], 每个 Tensor 形状为 [M_i, K_i]
#  - weights: List[Tensor], 每个 Tensor 形状为 [K_i, N_i]
#  - bias: List[Tensor] 或 None，对应每个输出通道的偏置
#  - group_type / group_list / split_item: 仅在需要分组或 split 行为时设置
_test_cases = [
    # Case 1: float32 单-单-单 基础场景
    #   x: [16,32], [8,16]
    #   w: [32,64], [16,32]
    #   bias: None
    #   group_type=None, group_list=None, split_item=0
    (
        [torch.randn(16, 32), torch.randn(8, 16)],
        [torch.randn(32, 64), torch.randn(16, 32)],
        None,
        None,  # Use kernel default (backend will map to -1)
        torch.float32,
        0,
        None,
        torch.float32,
    ),
    # Case 2: float16 多-多-多 场景
    #   x: [3,4], [5,4]
    #   w: [4,6], [4,6]
    #   bias: [6], [6]
    #   group_type=-1, group_list=None, split_item=0
    (
        [torch.randn(3, 4, dtype=torch.float16), torch.randn(5, 4, dtype=torch.float16)],
        [torch.randn(4, 6, dtype=torch.float16), torch.randn(4, 6, dtype=torch.float16)],
        [torch.randn(6, dtype=torch.float16), torch.randn(6, dtype=torch.float16)],
        -1,
        torch.float16,
        0,
        None,
        torch.float16,
    ),
    # Case 3: bfloat16 单-多-多 场景，使用 group_list 分组
    #   x: [10,4]
    #   w: [4,6], [4,6]
    #   bias: [6], [6]
    #   group_type=0, group_list=[5, 10] 表示按 M 维 5/5 分组，split_item=0
    (
        [torch.randn(10, 4, dtype=torch.bfloat16)],
        [torch.randn(4, 6, dtype=torch.bfloat16), torch.randn(4, 6, dtype=torch.bfloat16)],
        [torch.randn(6, dtype=torch.float32), torch.randn(6, dtype=torch.float32)],
        0,
        torch.bfloat16,
        0,
        [5, 10],
        torch.bfloat16,
    ),
]


@pytest.mark.parametrize("inputs, weights, bias, group_type, dtype, split_item, group_list, output_dtype", _test_cases)
@auto_switch_platform()
@bypass_not_implemented
def test_grouped_matmul_accuracy(inputs, weights, bias, group_type, dtype, split_item, group_list, output_dtype):
    """
    Tests the accuracy of the MojoGroupedMatmul operator.
    """
    grouped_matmul_op = MojoGroupedMatmul()

    device = get_platform()

    input_tensors = [t.to(device=device) for t in inputs]
    weight_tensors = [t.to(device=device) for t in weights]
    bias_tensors = [b.to(device=device) for b in bias] if bias else None

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    elif dtype == torch.float16:
        atol, rtol = 3e-2, 6e-3
    else: 
        atol, rtol = 0, 0

    # 当使用 group_list（分组语义）时，参考实现尚未实现相同的切分逻辑，
    # 只校验 NPU kernel 是否能正常运行，不做 forward_ref 对比。
    if group_list is not None:
        grouped_matmul_op.forward_std(
            input_tensors,
            weight_tensors,
            group_list=group_list,
            bias=bias_tensors,
            group_type=group_type,
            split_item=split_item,
            output_dtype=output_dtype,
        )
    else:
        grouped_matmul_op.forward_diff(
            input_tensors,
            weight_tensors,
            bias=bias_tensors,
            group_type=group_type,
            split_item=split_item,
            atol=atol,
            rtol=rtol,
            group_list=group_list,
            output_dtype=output_dtype,
        )