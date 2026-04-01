import random

import pytest
import torch

from tests.utils import bypass_not_implemented
from tests.utils import get_platform
from tests.utils import auto_switch_platform
from tests.utils import assert_close

from mojo_opset import MojoMoE

moe_configs = [
    (1024, 64, 8, 1024, 4096, "swiglu", torch.bfloat16),
]

def generate_moe_weights_and_inputs(max_num_tokens, num_experts, hidden_size, intermediate_size, activation, dtype):
    gating_weight = torch.randn(size=(hidden_size, num_experts), dtype=torch.float32) / (hidden_size ** 0.5)
    if activation == "swiglu":
        up_proj_weight = torch.randn(size=(num_experts, intermediate_size*2, hidden_size), dtype=dtype)
    else:
        up_proj_weight = torch.randn(size=(num_experts, intermediate_size, hidden_size), dtype=dtype)
    up_proj_weight = up_proj_weight / (hidden_size ** 0.5)

    down_proj_weight = torch.randn(size=(num_experts, hidden_size, intermediate_size), dtype=dtype) / (hidden_size ** 0.5)
    num_tokens = random.randint((max_num_tokens + 1)// 2, max_num_tokens)
    input_hidden = torch.randn(size=(num_tokens, hidden_size), dtype=dtype)
    return input_hidden, gating_weight, up_proj_weight, down_proj_weight

@pytest.mark.parametrize(
    "input_hidden, gating_weight, up_proj_weight, down_proj_weight, num_experts, top_k, hidden_size, intermediate_size, activation",
    [
        pytest.param(
            *generate_moe_weights_and_inputs(max_num_tokens=max_num_tokens, num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size, activation=activation, dtype=dtype),
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            activation,
        )
        for max_num_tokens, num_experts, top_k, hidden_size, intermediate_size, activation, dtype in moe_configs
    ]
)
@auto_switch_platform()
def test_moe(input_hidden, gating_weight, up_proj_weight, down_proj_weight, num_experts, top_k, hidden_size, intermediate_size, activation):
    mojo_moe = MojoMoE(
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        activation,
    )
    assert mojo_moe.gating.gate_weight.dtype == torch.float32
    mojo_moe.load_state_dict({
        "gating.gate_weight": gating_weight,
        "experts.up_proj_weight": up_proj_weight,
        "experts.down_proj_weight": down_proj_weight,
    })
    assert mojo_moe.gating.gate_weight.dtype == torch.float32

    mojo_output = mojo_moe(input_hidden)

    # placeholder for comparison with other backends
    def naive_moe(input_hidden):
        router_logits = input_hidden.float() @ gating_weight.float()
        router_scores = torch.softmax(router_logits, dim=-1) # [num_tokens, num_experts]
        router_weights, router_experts = router_scores.topk(top_k, dim=-1) # [num_tokens, topk]
        router_weights = router_weights / torch.sum(router_weights, dim=-1, keepdim=True)
        expert_mask = torch.nn.functional.one_hot(router_experts, num_experts) # [num_tokens, topk, num_experts]
        expert_mask = expert_mask.permute(2, 1, 0) # [num_experts, topk, num_tokens]
        
        final_hidden_states = torch.zeros_like(input_hidden, dtype=torch.float32)
        for expert_idx in range(num_experts):
            top_x, idx = torch.where(expert_mask[expert_idx])
            current_state = input_hidden[idx]
            current_intermediate = torch.nn.functional.linear(current_state, up_proj_weight[expert_idx].float())
            if activation == "swiglu":
                current_intermediate = torch.nn.functional.silu(current_intermediate[:, :intermediate_size]) * current_intermediate[:, intermediate_size:]
            else:
                assert False, "not supported yet"
            current_output = torch.nn.functional.linear(current_intermediate, down_proj_weight[expert_idx].float())
            current_output = current_output * router_weights[idx, top_x, None]
            final_hidden_states.index_add_(0, idx, current_output)
        return final_hidden_states.to(input_hidden.dtype)
           
    naive_output = naive_moe(input_hidden)

    assert_close(mojo_output, naive_output)
