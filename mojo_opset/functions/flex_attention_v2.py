"""FlexAttention v2 interface and autograd state, including prepared dropout."""

from typing import NamedTuple

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl
from ._flex_attention import _restore_block_mask
from ._flex_attention import _save_block_mask
from ._flex_attention import _validate


class _SavedDropoutLayout(NamedTuple):
    block_table: torch.Tensor
    q_len: int
    kv_len: int
    num_active: int
    dkv_schedule: torch.Tensor | None
    dkv_ranges: torch.Tensor | None
    dkv_num_tasks: int


class _SavedDropout(NamedTuple):
    dropout_p: float
    threshold: int
    scale: float
    bitpacked: bool
    batch: int
    heads: int
    tiles: torch.Tensor | None
    layout: _SavedDropoutLayout


class FlexAttentionV2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        block_mask,
        scale,
        implementation,
        enable_gqa=False,
        score_mod=None,
        return_lse=False,
        dropout=None,
    ):
        if score_mod is not None or return_lse:
            raise NotImplementedError("FlexAttention v2 does not support score_mod or return_lse")
        _validate(q, k, v, block_mask)
        if not enable_gqa and q.shape[1] != k.shape[1]:
            raise ValueError("enable_gqa=False requires equal Q/K/V head counts")
        forward, backward = load_impl("flex_attention_v2", implementation, require_backward=True)
        ctx.has_dropout_state = dropout is not None
        dropout_tensors = ()
        if dropout is None:
            output, lse = forward(q, k, v, block_mask, scale)
            saved_output = output
        else:
            if not getattr(forward, "supports_dropout", False):
                raise NotImplementedError("Prepared dropout is supported by FlexAttention v2 A5 Triton and Torch")
            layout = dropout.layout
            if layout.block_mask is not block_mask:
                raise ValueError("Dropout state must be prepared for this BlockMask")
            if q.shape[:3] != (dropout.batch, dropout.heads, layout.q_len) or k.shape[2] != layout.kv_len:
                raise ValueError("Q/K shapes do not match the prepared dropout state")
            output, lse, saved_output = forward(q, k, v, block_mask, scale, dropout=dropout)
            dropout_tensors = (
                dropout.tiles,
                layout.block_table,
                layout.dkv_schedule,
                layout.dkv_ranges,
            )
            ctx.dropout_metadata = (
                dropout.dropout_p,
                dropout.threshold,
                dropout.scale,
                dropout.bitpacked,
                dropout.batch,
                dropout.heads,
                layout.q_len,
                layout.kv_len,
                layout.num_active,
                layout.dkv_num_tasks,
            )
        mask_tensors = _save_block_mask(ctx, block_mask)
        ctx.save_for_backward(q, k, v, saved_output, lse, *dropout_tensors, *mask_tensors)
        ctx.scale, ctx.backward_kernel = scale, backward
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output, lse, *mask_tensors = ctx.saved_tensors
        dropout = None
        if ctx.has_dropout_state:
            random_values, block_table, schedule, ranges, *mask_tensors = mask_tensors
            p, threshold, drop_scale, packed, batch, heads, q_len, kv_len, active, tasks = ctx.dropout_metadata
            dropout = _SavedDropout(
                dropout_p=p,
                threshold=threshold,
                scale=drop_scale,
                bitpacked=packed,
                batch=batch,
                heads=heads,
                tiles=random_values,
                layout=_SavedDropoutLayout(
                    block_table=block_table,
                    q_len=q_len,
                    kv_len=kv_len,
                    num_active=active,
                    dkv_schedule=schedule,
                    dkv_ranges=ranges,
                    dkv_num_tasks=tasks,
                ),
            )
        block_mask = _restore_block_mask(ctx, mask_tensors)
        kwargs = {} if dropout is None else {"dropout": dropout}
        dq, dk, dv = ctx.backward_kernel(
            grad_output.contiguous(), q, k, v, output, lse, block_mask, ctx.scale, **kwargs
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flex_attention_v2(
    q,
    k,
    v,
    score_mod=None,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    return_lse=False,
    *,
    implementation=None,
    dropout=None,
    **kwargs,
):
    """Grouped attention with optional precomputed sparse dropout.

    Prepare ``dropout`` with ``utils.attention_dropout.create_flex_dropout_mask``
    outside capture, and keep it unchanged until backward completes. A5 Triton
    and Torch support dropout; score_mod and return_lse remain unsupported.
    """
    # Specialize p=0 exactly like the original no-dropout call.
    if dropout is not None and dropout.dropout_p == 0:
        dropout = None
    return FlexAttentionV2Function.apply(
        q, k, v, block_mask, scale, implementation, enable_gqa, score_mod, return_lse, dropout
    )
