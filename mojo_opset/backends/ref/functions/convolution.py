from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

from einops import rearrange

from mojo_opset.core import MojoCausalConv1dFunction


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


class RefCausalConv1dFunction(MojoCausalConv1dFunction):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
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
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
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
