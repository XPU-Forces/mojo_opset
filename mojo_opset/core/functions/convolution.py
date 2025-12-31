from typing import Optional

import torch

from ..function import MojoFunction


class MojoCausalConv1dFunction(MojoFunction):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        initial_state=None,
        output_final_state=False,
        final_states_out=None,
        activation=None,
        cu_seqlens=None,
    ):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass


def mojo_causal_conv1d(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    backend: str = "triton",
    cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, T, D].
        weight (Optional[torch.Tensor]):
            Weight tensor of shape [D, W]. Default: `None`.
        bias (Optional[torch.Tensor]):
            Bias tensor of shape [D]. Default: `None`.
        residual (Optional[torch.Tensor]):
            Residual tensor of shape [B, T, D]. Default: `None`.
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
