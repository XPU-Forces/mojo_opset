from typing import Optional

import torch
import torch.nn.functional as F

from ..operator import MojoOperator

class MojoCausalConv1dUpdateState(MojoOperator):
    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,         
        bias: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Causal Convolution-1D forward.
        
        Args:
            hidden_states: Hidden states with shape of (batch, dim, seq_len)
            conv_state: Initial state to be convoluted with hidden_states, with shape of (batch, dim, state_len)
            weight: Weight of Conv1d operator, with shape of (dim, window_size) 
            bias:  Bias of Conv1d, with shape of (dim,)
            activation: Flag for making silu activation on output

        Returns: Causal Conv1d output with shape of (batch, dim, seq_len  + state_len - window_size + 1)
        
        Notes:
            - After forward this function conv_state will be update to final state.
        """
        _, hidden_size, seq_len = hidden_states.shape
        state_len = conv_state.shape[-1]
        hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        conv_state.copy_(hidden_states_new[:, :, -state_len:])
        out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
        out = out[:, :, -seq_len:]
        if activation in ["silu", "swish"]:
            out = F.silu(out)
        out = out.to(hidden_states.dtype)
        return out