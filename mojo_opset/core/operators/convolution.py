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


class MojoConv1d(MojoOperator):
    """Functional ``conv1d`` core op over ``[batch, channels, seq_len]`` inputs.

    This operator follows ``torch.nn.functional.conv1d`` semantics as closely
    as practical for a core op:
        - ``hidden_states`` uses ``[batch, in_channels, seq_len]``.
        - ``weight`` uses the native ``conv1d`` layout
          ``[out_channels, in_channels / groups, kernel_size]``.
        - ``bias`` is optional with shape ``[out_channels]``.
        - ``stride``, ``padding``, ``dilation``, and ``groups`` are operator
          hyper-parameters carried on the module instance.

    Defaults are chosen to match the current USM Conformer depthwise module:
        - ``stride=1``
        - ``padding="same"``
        - ``dilation=1``
        - ``groups=None`` which means "resolve to ``in_channels`` at runtime",
          i.e. depthwise by default for the current use case.

    Notes about the intentionally chosen boundary:
        - This op is still functional: weights and bias are passed to
          ``forward`` instead of being owned as ``nn.Parameter``.
        - ``padding_mode`` is not modeled; this op always follows the zero-pad
          behavior of ``F.conv1d``.
        - ``transposed`` convolution is out of scope.
        - Fused activation / residual / causal state-update behavior remains in
          separate operators.
        - Backend-specific packed layouts are out of scope for the core op.
    """

    def __init__(
        self,
        stride: int = 1,
        padding: int | str = "same",
        dilation: int = 1,
        groups: Optional[int] = None,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply ``conv1d`` using native PyTorch weight layout.

        Args:
            hidden_states: Input tensor in ``[batch, in_channels, seq_len]``.
            weight: Native conv1d weight tensor in
                ``[out_channels, in_channels / groups, kernel_size]``.
            bias: Optional bias tensor in ``[out_channels]``.

        Returns:
            Tensor in ``[batch, out_channels, output_seq_len]``.

        Raises:
            ValueError: If inputs violate the conv1d contract.
        """

        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be [B, C_in, T], got {hidden_states.shape}")
        if weight.ndim != 3:
            raise ValueError(f"weight must be [C_out, C_in/groups, K], got {weight.shape}")

        _, in_channels, _ = hidden_states.shape
        out_channels, weight_in_channels, _ = weight.shape
        groups = in_channels if self.groups is None else self.groups

        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if in_channels % groups != 0:
            raise ValueError(f"in_channels={in_channels} must be divisible by groups={groups}")
        expected_weight_in_channels = in_channels // groups
        if weight_in_channels != expected_weight_in_channels:
            raise ValueError(
                "weight second dimension must equal in_channels / groups, got "
                f"{weight_in_channels} vs expected {expected_weight_in_channels}"
            )
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != out_channels:
                raise ValueError(f"bias must be [C_out] with C_out={out_channels}, got {bias.shape}")

        output = F.conv1d(
            hidden_states.to(weight.dtype),
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups,
        )
        return output.to(hidden_states.dtype)

    def extra_repr(self) -> str:
        return (
            f"stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}"
        )
