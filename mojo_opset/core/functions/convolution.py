from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

from einops import rearrange

from ..function import MojoFunction


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: str = None,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    x = rearrange(x, "b t d -> b d t")

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    dtype_in = x.dtype
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32) if bias is not None else None

    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_state is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        initial_state = initial_state.to(x.dtype)
        x_padded = torch.cat([initial_state, x], dim=-1)
        out = F.conv1d(x_padded, weight.unsqueeze(1), bias, padding=0, groups=dim)

    out = out[..., :seqlen]

    final_states = None
    if output_final_state:
        start_idx = x.shape[-1] - (width - 1)
        if start_idx < 0:
            final_states = F.pad(x, (width - 1 - x.shape[-1], 0))
        else:
            final_states = x[..., start_idx:]

        final_states = final_states.to(dtype_in)

        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states

    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    out = rearrange(out, "b d t -> b t d")

    if residual is not None:
        out = out + residual

    return out, final_states_out


def _ref_forward_impl(
    x,
    weight,
    bias,
    residual,
    initial_state,
    output_final_state,
    activation,
    cu_seqlens,
):
    if cu_seqlens is None:
        out, final_state = causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            initial_state=initial_state,
            output_final_state=output_final_state,
            final_states_out=None,
            activation=activation,
            residual=residual,
        )
        return out, final_state
    else:
        # NOTE(@wenshuo.zhao): under varlen setting, device computing leads to incorrect results,
        # so we use cpu computing results as golden.
        w_cpu = weight.cpu().float() if weight is not None else None
        b_cpu = bias.cpu().float() if bias is not None else None

        device = x.device
        dtype = x.dtype
        x_cpu = x.cpu()

        res_cpu = residual.cpu() if residual is not None else None

        s_cpu = initial_state.cpu() if initial_state is not None else None

        out_list = []
        state_list = []

        for batch_idx, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            chunk_x = x_cpu[:, bos:eos]
            chunk_res = res_cpu[:, bos:eos] if res_cpu is not None else None

            chunk_state = s_cpu[batch_idx : batch_idx + 1] if s_cpu is not None else None

            curr_out, curr_state = causal_conv1d(
                x=chunk_x,
                weight=w_cpu,
                bias=b_cpu,
                initial_state=chunk_state,
                output_final_state=output_final_state,
                final_states_out=None,
                activation=activation,
                residual=chunk_res,
            )

            out_list.append(curr_out)
            if output_final_state:
                state_list.append(curr_state)

        out = torch.cat(out_list, dim=1).to(device=device, dtype=dtype)

        final_state = None
        if output_final_state and state_list:
            final_state = torch.cat(state_list, dim=0).to(device=device, dtype=dtype)

        return out, final_state


class MojoCausalConv1dFunction(MojoFunction):
    """
    MojoCausalConv1dFunction implements the causal 1D convolution.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a causal 1D convolution, optionally with state passing for recurrent inference.

        Args:
            ctx: The context object for saving information for the backward pass.
            x (torch.Tensor): The input tensor.
                - For batched mode: Shape `[B, T, D]`, where B is batch size,
                  T is sequence length, and D is the hidden dimension.
                - For varlen mode: Shape `[1, Total_T, D]`, where Total_T is the
                  sum of all sequence lengths in the batch.
            weight (torch.Tensor): The convolution kernel weights.
                Shape `[D, W]`, where W is the kernel width.
            bias (Optional[torch.Tensor]): An optional bias tensor added after convolution.
                Shape `[D]`. Defaults to None.
            residual (Optional[torch.Tensor]): An optional residual tensor to be added
                to the output. Its shape must match `x`. Defaults to None.
            initial_state (Optional[torch.Tensor]): The initial state for the convolution,
                used for recurrent/streaming computations.
                - For batched mode: Shape `[B, D, W-1]`.
                - For varlen mode: Shape `[N, D, W-1]`, where N is the number of
                  sequences in the batch.
                Defaults to None, implying a zero-padded initial state.
            output_final_state (bool): If True, the function will compute and return
                the final state of the convolution, suitable for use as `initial_state`
                in a subsequent call. Defaults to False.
            activation (str, optional): The activation function to apply to the output.
                Supported values: "silu", "swish", or None. Defaults to None.
            cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths, used to
                specify boundaries for sequences in varlen mode. Shape `[N+1]`,
                e.g., `[0, len_seq1, len_seq1+len_seq2, ...]`. If provided, the
                function operates in varlen mode. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - out (torch.Tensor): The output tensor of the convolution, with the same
              shape as the input `x`.
            - final_state (Optional[torch.Tensor]): The final state of the convolution,
              with shape `[B or N, D, W-1]`. Returned only if `output_final_state`
              is True, otherwise None.
        """
        ctx.save_for_backward(x, weight, bias, residual, initial_state, cu_seqlens)
        ctx.output_final_state = output_final_state
        ctx.activation = activation

        out, final_state = _ref_forward_impl(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )
        return out, final_state

    @staticmethod
    def backward(
        ctx,
        dy: torch.Tensor,
        dht: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        Backward pass of the causal 1D convolution.

        Args:
            ctx: The context object containing tensors saved from the forward pass.
            dy (torch.Tensor): The gradient of the loss with respect to the output tensor `out`.
                Its shape must match the `out` tensor from the forward pass:
                - For batched mode: `[B, T, D]`
                - For varlen mode: `[1, Total_T, D]`
            dht (Optional[torch.Tensor]): The gradient of the loss with respect to the
                `final_state` tensor. This is only provided if `output_final_state`
                was True in the forward pass. Its shape must match `final_state`:
                - For batched mode: `[B, D, W-1]`
                - For varlen mode: `[N, D, W-1]`
                Defaults to None.

        Returns:
            Tuple of (dx, dweight, dbias, dresidual, dinitial_state, None, None, None, None).
        """
        x, weight, bias, residual, initial_state, cu_seqlens = ctx.saved_tensors
        output_final_state = ctx.output_final_state
        activation = ctx.activation

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            tensors_to_grad = [x]

            if weight is not None:
                weight = weight.detach().requires_grad_(True)
                tensors_to_grad.append(weight)

            if bias is not None:
                bias = bias.detach().requires_grad_(True)
                tensors_to_grad.append(bias)

            if residual is not None:
                residual = residual.detach().requires_grad_(True)
                tensors_to_grad.append(residual)

            if initial_state is not None:
                initial_state = initial_state.detach().requires_grad_(True)
                tensors_to_grad.append(initial_state)

            out, final_state = _ref_forward_impl(
                x=x,
                weight=weight if weight is not None else None,
                bias=bias if bias is not None else None,
                residual=residual if residual is not None else None,
                initial_state=initial_state if initial_state is not None else None,
                output_final_state=output_final_state,
                activation=activation,
                cu_seqlens=cu_seqlens,
            )

            outputs_with_grad = []
            grads_from_upstream = []

            outputs_with_grad.append(out)
            grads_from_upstream.append(dy)

            if output_final_state and final_state is not None and dht is not None:
                outputs_with_grad.append(final_state)
                grads_from_upstream.append(dht)

            computed_grads = torch.autograd.grad(
                outputs_with_grad, tensors_to_grad, grads_from_upstream, allow_unused=True
            )

        grad_idx = 0

        dx = computed_grads[grad_idx]
        grad_idx += 1

        dw = None
        if weight is not None:
            dw = computed_grads[grad_idx]
            grad_idx += 1

        db = None
        if bias is not None:
            db = computed_grads[grad_idx]
            grad_idx += 1

        dr = None
        if residual is not None:
            dr = computed_grads[grad_idx]
            grad_idx += 1

        d_init = None
        if initial_state is not None:
            d_init = computed_grads[grad_idx]
            grad_idx += 1

        return dx, dw, db, dr, d_init, None, None, None


def mojo_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        ctx: The context object for saving information for the backward pass.
        x (torch.Tensor): The input tensor.
            - For batched mode: Shape `[B, T, D]`, where B is batch size,
                T is sequence length, and D is the hidden dimension.
            - For varlen mode: Shape `[1, Total_T, D]`, where Total_T is the
                sum of all sequence lengths in the batch.
        weight (torch.Tensor): The convolution kernel weights.
            Shape `[D, W]`, where W is the kernel width.
        bias (Optional[torch.Tensor]): An optional bias tensor added after convolution.
            Shape `[D]`. Defaults to None.
        residual (Optional[torch.Tensor]): An optional residual tensor to be added
            to the output. Its shape must match `x`. Defaults to None.
        initial_state (Optional[torch.Tensor]): The initial state for the convolution,
            used for recurrent/streaming computations.
            - For batched mode: Shape `[B, D, W-1]`.
            - For varlen mode: Shape `[N, D, W-1]`, where N is the number of
                sequences in the batch.
            Defaults to None, implying a zero-padded initial state.
        output_final_state (bool): If True, the function will compute and return
            the final state of the convolution, suitable for use as `initial_state`
            in a subsequent call. Defaults to False.
        activation (str, optional): The activation function to apply to the output.
            Supported values: "silu", "swish", or None. Defaults to None.
        cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths, used to
            specify boundaries for sequences in varlen mode. Shape `[N+1]`,
            e.g., `[0, len_seq1, len_seq1+len_seq2, ...]`. If provided, the
            function operates in varlen mode. Defaults to None.


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
