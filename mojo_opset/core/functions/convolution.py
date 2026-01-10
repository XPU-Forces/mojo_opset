from typing import Optional

import torch

from ..function import MojoFunction


class MojoCausalConv1dFunction(MojoFunction):
    """
    MojoCausalConv1dFunction implements the causal 1D convolution for text diffusion.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        residual: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        activation: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the causal 1D convolution.

        Todo:
            impl reference impl for causal conv1d.

        Args:
            x (torch.Tensor):
                Input tensor of shape [***].
            weight (torch.Tensor):
                Weight tensor of shape [***].
            bias (Optional[torch.Tensor]):
                Bias tensor of shape [***]. Default: `None`.
            initial_state (Optional[torch.Tensor]):
                Initial state tensor of shape [***],
                where `N` is the number of sequences in the batch and `W` is the kernel size.
                If provided, the initial state is used to initialize the cache. Default: `None`.
            output_final_state (bool):
                Whether to output the final state of shape [***]. Default: `False`.
            final_states_out (Optional[torch.Tensor]):
                Output tensor for final states of shape [***]. Default: `None`.
            activation (Optional[str]):
        """
        raise NotImplementedError("MojoCausalConv1dFunction forward is not implemented.")

    @staticmethod
    def backward(
        ctx,
        dy: torch.Tensor,
        dht: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        Backward pass of the causal 1D convolution.

        Todo:
            impl reference impl for causal conv1d.
        Args:
            grad_outputs (torch.Tensor):
                Gradient tensor of shape [***].

        Returns:
            Tuple of (dx, dweight, dbias, dresidual, dinitial_state, None, None, None, None).
        """
        raise NotImplementedError("MojoCausalConv1dFunction backward is not implemented.")


def mojo_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [***].
        weight (torch.Tensor):
            Weight tensor of shape [***]. Default: `None`.
        bias (torch.Tensor):
            Bias tensor of shape [***]. Default: `None`.
        residual (torch.Tensor):
            Residual tensor of shape [***]. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state tensor of shape [N, D, W],
            where `N` is the number of sequences in the batch and `W` is the kernel size.
            If provided, the initial state is used to initialize the cache. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape [N, D, W]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (Optional[str]):
            Specifies the backend to use for the convolution operation. Supported values are `'cuda'` and `'triton'`.
            Default: `'triton'`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths (optional)

    Returns:
        Tuple of (output, final_state).
        If `output_final_state` is `False`, the final state is `None`.
    """

    y, final_state = MojoCausalConv1dFunction.apply(
        x,
        weight,
        bias,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
    )
    return y, final_state
