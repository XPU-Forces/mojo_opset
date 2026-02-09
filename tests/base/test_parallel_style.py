import torch

from torch import nn
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.device_mesh import init_device_mesh

from mojo_opset.distributed.parallel import (
    MojoColwiseParallel,
    MojoRowwiseParallel,
    MojoExpertParallel
)

from mojo_opset import MojoGroupGemm

def test_colwise_parallel():
    class ColwiseGemm(torch.nn.Module):
        def __init__(self, in_feature, out_feature):
            super().__init__()
            self.linear = nn.Linear(in_features=in_feature, out_features=out_feature)

        @torch.inference_mode
        def forward(self, hidden_states):
            return self.linear(hidden_states)
    
    in_feature = 4096
    out_feature = 16384

    test_colwise_gemm = ColwiseGemm(in_feature, out_feature)

    device_mesh = init_device_mesh("npu", (4,))
    test_colwise_gemm = parallelize_module(
        test_colwise_gemm, 
        device_mesh=device_mesh,
        parallelize_plan={"linear": MojoColwiseParallel()}
    )

    hidden_states = torch.zeros(16,4096)
    hidden_states = test_colwise_gemm(hidden_states)
    assert hidden_states.shape[-1] == 16384 // 4

def test_rowwise_parallel():
    class RowwiseGemm(torch.nn.Module):
        def __init__(self, in_feature, out_feature):
            super().__init__()
            self.linear = nn.Linear(in_features=in_feature, out_features=out_feature)
        
        @torch.inference_mode
        def forward(self, hidden_states):
            return self.linear(hidden_states)
    
    in_feature = 16384
    out_feature = 4096

    test_rowwise_gemm = RowwiseGemm(in_feature, out_feature)
    device_mesh = init_device_mesh("npu", (4,))

    test_rowwise_gemm = parallelize_module(
        test_rowwise_gemm, 
        device_mesh=device_mesh,
        parallelize_plan={"linear": MojoRowwiseParallel()}
    )

    hidden_states = torch.zeros(16,in_feature // 4)
    assert hidden_states.shape[-1] == test_rowwise_gemm.linear.weight.to_local().shape[-1]
    hidden_states = test_rowwise_gemm(hidden_states)

def test_moe_parallel():
    class GroupGemmEp(torch.nn.Module):
        def __init__(self, group_num, in_feature, out_feature):
            super().__init__()
            self.group_gemm = MojoGroupGemm(group_num=group_num, in_feature=in_feature, out_feature=out_feature)

        @torch.inference_mode
        def forward(self, hidden_states, group_list):
            return self.group_gemm(hidden_states, group_list=group_list)

    in_feature = 4096
    out_feature = 1024 * 2
    num_experts = 384
    select_expert_num = 8

    test_ep_group_gemm = GroupGemmEp(group_num=num_experts, in_feature=in_feature, out_feature=out_feature)

    device_mesh = init_device_mesh("npu", (4,))

    test_ep_group_gemm = parallelize_module(
        test_ep_group_gemm, 
        device_mesh=device_mesh,
        parallelize_plan={"group_gemm": MojoExpertParallel()}
    )

    hidden_states = torch.zeros((16 * select_expert_num, in_feature))
    group_list = [0 for i in range(0, num_experts // 4)]
    group_list[0] = 2 * select_expert_num
    group_list[4] = 2 * select_expert_num
    group_list[8] = 2 * select_expert_num
    group_list[16] = 2 * select_expert_num
    group_list[32] = 2 * select_expert_num
    group_list[52] = 2 * select_expert_num
    group_list[64] = 2 * select_expert_num
    group_list[95] = 2 * select_expert_num
    
    group_list = torch.tensor(group_list).npu()
    hidden_states = test_ep_group_gemm(hidden_states, group_list)

if __name__ == "__main__":
    #test_colwise_parallel()
    #test_rowwise_parallel()
    test_moe_parallel()