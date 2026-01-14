from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels.npu import causal_conv1d_update_bdt_fwd
from mojo_opset.core.operators.convolution import MojoCausalConv1dUpdateState

class TTXCausalConv1dUpdateState(MojoCausalConv1dUpdateState):
    supported_platforms_list = ["npu"]

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

        Returns: Causal Conv1d output with shape of (batch, dim, seq_len)
        
        Notes:
            - After forward this function conv_state will be update to final state.
        """
        return causal_conv1d_update_bdt_fwd(hidden_states, conv_state, weight, 
                                            bias=bias, activation=activation)