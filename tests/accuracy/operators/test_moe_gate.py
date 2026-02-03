import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoMoEGating

@pytest.mark.parametrize(
    "top_k,num_experts, batch_size, seq_len, hidden_dim, dtype",
    [
        (
            2,
            16,
            2,
            4096,
            1024,
            dtype  
        )
        for dtype in [ "float32", "bfloat16"]
    ],
)

@auto_switch_platform()
@bypass_not_implemented
def test_group_moe_gate(top_k, num_experts, batch_size, seq_len, hidden_dim, dtype):
    torch.manual_seed(1234)
    device = torch.device("npu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim,device=device, dtype=dtype)  # [2,4,16]
    gate_weight = torch.randn(hidden_dim, num_experts, device=device, dtype=dtype)
    gate_weight_param = torch.nn.Parameter(gate_weight)
    gate = MojoMoEGating(
        hidden_size = hidden_dim,
        num_experts = num_experts,
        top_k = top_k,
        )
    gate.gate_weight=gate_weight_param
    ref_indices,ref_gate_weights = gate.forward_ref(hidden_states)
    triton_indices, triton_gate_weights = gate.forward(hidden_states)
    torch.testing.assert_close(triton_indices, ref_indices, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(triton_gate_weights, ref_gate_weights, atol=1e-6, rtol=1e-6)