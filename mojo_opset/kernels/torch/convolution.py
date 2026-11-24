from typing import Optional

import torch
import torch.nn.functional as torch_f


def causal_conv1d_update_state_infer_fwd(x, conv_state, weight, bias, activation):
    if x.shape[-1] == 0:
        return torch.empty_like(x)
    combined = torch.cat((conv_state, x), dim=-1).to(weight.dtype)
    if conv_state.shape[-1]:
        conv_state.copy_(combined[..., -conv_state.shape[-1]:])
    output = torch.nn.functional.conv1d(combined, weight[:, None], bias, groups=x.shape[1])
    output = output[..., -x.shape[-1]:]
    if activation in ("silu", "swish"):
        output = torch.nn.functional.silu(output)
    return output.to(x.dtype).contiguous()


def _causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    activation: Optional[str],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_dtype = x.dtype
    x_channels_first = x.transpose(1, 2).float()
    width = weight.shape[1]
    if initial_state is None:
        convolution_input = x_channels_first
        padding = width - 1
    else:
        convolution_input = torch.cat((initial_state.float(), x_channels_first), dim=-1)
        padding = 0

    output = torch_f.conv1d(
        convolution_input,
        weight.float().unsqueeze(1),
        None if bias is None else bias.float(),
        padding=padding,
        groups=weight.shape[0],
    )[..., : x.shape[1]]
    if activation is not None:
        output = torch_f.silu(output)
    output = output.to(input_dtype).transpose(1, 2)
    if residual is not None:
        output = output + residual

    final_state = None
    if output_final_state:
        state_source = x_channels_first.to(input_dtype)
        if initial_state is not None:
            state_source = torch.cat((initial_state, state_source), dim=-1)
        state_length = width - 1
        final_state = torch_f.pad(state_source, (max(state_length - state_source.shape[-1], 0), 0))
        final_state = final_state[..., -state_length:] if state_length else final_state[..., :0]
    return output, final_state


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    activation: Optional[str],
    cu_seqlens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if cu_seqlens is None:
        return _causal_conv1d(x, weight, bias, residual, initial_state, output_final_state, activation)

    offsets = cu_seqlens.detach().cpu().tolist()
    outputs = []
    final_states = []
    for batch, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = None if initial_state is None else initial_state[batch : batch + 1]
        chunk_residual = None if residual is None else residual[:, start:end]
        output, final_state = _causal_conv1d(
            x[:, start:end],
            weight,
            bias,
            chunk_residual,
            state,
            output_final_state,
            activation,
        )
        outputs.append(output)
        if final_state is not None:
            final_states.append(final_state)
    packed_output = torch.cat(outputs, dim=1)
    packed_final_state = torch.cat(final_states) if final_states else None
    return packed_output, packed_final_state


def causal_conv1d_bwd(
    grad_output: torch.Tensor,
    grad_final_state: Optional[torch.Tensor],
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    activation: Optional[str],
    cu_seqlens: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    originals = (x, weight, bias, residual, initial_state)
    differentiable = tuple(tensor.detach().requires_grad_(True) for tensor in originals if tensor is not None)
    iterator = iter(differentiable)
    recompute = tuple(next(iterator) if tensor is not None else None for tensor in originals)
    with torch.enable_grad():
        output, final_state = causal_conv1d_fwd(*recompute, output_final_state, activation, cu_seqlens)
        outputs = [output]
        grad_outputs = [grad_output]
        if final_state is not None and grad_final_state is not None:
            outputs.append(final_state)
            grad_outputs.append(grad_final_state)
        gradients = torch.autograd.grad(outputs, differentiable, grad_outputs)

    gradient_iterator = iter(gradients)
    return tuple(next(gradient_iterator).to(tensor.dtype) if tensor is not None else None for tensor in originals)
