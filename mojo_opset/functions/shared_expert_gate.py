from typing import Optional

import torch

from ._dispatch import load_impl


def shared_expert_gate(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_out: torch.Tensor,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Fused ``sigmoid(hidden_states @ gate_weight.T) * shared_out``.

    ``gate_weight`` is the bias-free ``[1, hidden]`` weight of the model's
    ``shared_expert_gate`` Linear.  ``hidden_states`` and ``shared_out`` share
    their trailing dimension and may be 2-D ``[tokens, hidden]`` or 3-D
    ``[batch, seq, hidden]``.
    """

    forward, _ = load_impl("shared_expert_gate", implementation)
    return forward(hidden_states, gate_weight, shared_out)
