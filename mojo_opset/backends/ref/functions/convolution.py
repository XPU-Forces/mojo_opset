from typing import Optional

import torch
import torch.nn.functional as F

from einops import rearrange

from mojo_opset.core import MojoCausalConv1dFunction


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_state=None,
    output_final_state=False,
    final_states_out=None,
    activation=None,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x, weight = x.to(torch.float32), weight.to(torch.float32)
    bias = bias.to(torch.float32) if bias is not None else None
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_state is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_state, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if output_final_state:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in,
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not output_final_state else (out, final_states_out)


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
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.save_for_backward(x, weight, bias, residual, initial_state)

        if cu_seqlens is None:
            ref = causal_conv1d_ref(
                x=rearrange(x, "b t d -> b d t"),
                weight=weight,
                bias=bias,
                initial_state=initial_state,
                output_final_state=output_final_state,
                final_states_out=None,
                activation=activation,
            )
        else:
            ref = torch.cat(
                [
                    rearrange(
                        causal_conv1d_ref(
                            x=rearrange(x[:, bos:eos].cpu().contiguous(), "b t d -> b d t"),
                            weight=weight.cpu(),
                            bias=bias.cpu() if bias else None,
                            initial_state=initial_state,
                            output_final_state=output_final_state,
                            final_states_out=None,
                            activation=activation,
                        ),
                        "b t d -> b d t",
                    )
                    + (residual[:, bos:eos].cpu() if residual else torch.zeros_like(x[:, bos:eos].cpu()))
                    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
                ],
                1,
            )
        return ref

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor]):
        pass


class RefSdpaFunction(MojoSdpaFunction):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale=1.0, enable_gqa=False):
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        ctx.save_for_backward(query, key, value, mask)
        return output

    @staticmethod
    def backward(ctx, do):
        query, key, value, attn_mask = ctx.saved_tensors

        with torch.enable_grad():
            query = query.detach().requires_grad_(True)
            key = key.detach().requires_grad_(True)
            value = value.detach().requires_grad_(True)

            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                scale=ctx.scale,
                enable_gqa=ctx.enable_gqa,
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                output, (query, key, value), do, retain_graph=False, allow_unused=False
            )

        return grad_query, grad_key, grad_value, None, None, None
