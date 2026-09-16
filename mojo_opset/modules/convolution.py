from typing import Optional

import torch

from mojo_opset import functions


class CausalConv1dUpdateStateInfer(torch.nn.Module):
    """Inference convolution with caller-owned, in-place rolling state."""

    def __init__(self, *, implementation: Optional[str] = None):
        super().__init__()
        self.implementation = implementation

    def forward(self, hidden_states, conv_state, weight, bias=None, activation=None):
        return functions.causal_conv1d_update_state_infer(
            hidden_states, conv_state, weight, bias, activation, implementation=self.implementation,
        )
