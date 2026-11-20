import torch
import torch.nn.functional as F


def shared_expert_gate_fwd(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_out: torch.Tensor,
) -> torch.Tensor:
    """``sigmoid(hidden_states @ gate_weight.T) * shared_out`` in fp32."""
    compute_dtype = (
        torch.float32
        if hidden_states.dtype in (torch.float16, torch.bfloat16)
        else hidden_states.dtype
    )
    gate = torch.sigmoid(F.linear(hidden_states.to(compute_dtype),
                                  gate_weight.to(compute_dtype)))
    return (gate * shared_out.to(compute_dtype)).to(shared_out.dtype)
