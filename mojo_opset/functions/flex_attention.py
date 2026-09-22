"""Original Mojo FlexAttention interface."""

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl

# Fields consumed by the kernel launchers and the reference. Keep additions
# explicit: a new kernel-side tensor must participate in saved-tensor hooks and
# version checks, rather than being retained invisibly in an arbitrary object.
_MASK_TENSORS = (
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
    "dense_mask",
    "packed_partial_mask",
    "partial_mask_offsets",
    "partial_block_table",
    "task_list_work_items",
    "task_list_offsets",
    "task_list_split_bases",
)
_MASK_METADATA = (
    "BLOCK_SIZE",
    "seq_lengths",
    "mask_mod",
    "task_list_max_sub",
    "task_list_sparse_kv_block_size",
    "task_list_num_kv_heads",
)


class FlexAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_mask, scale, implementation, op_id):
        forward, backward = load_impl(op_id, implementation, require_backward=True)
        output, lse = forward(q, k, v, block_mask, scale)
        # Snapshot only fields used by these kernels, without generic object
        # reflection/TreeSpec operations that Dynamo cannot capture.
        ctx.mask_type = type(block_mask)
        ctx.mask_tensor_names = tuple(name for name in _MASK_TENSORS if hasattr(block_mask, name))
        ctx.mask_metadata = tuple(
            (name, getattr(block_mask, name)) for name in _MASK_METADATA if hasattr(block_mask, name)
        )
        ctx.save_for_backward(q, k, v, output, lse, *(getattr(block_mask, name) for name in ctx.mask_tensor_names))
        ctx.scale, ctx.backward_kernel = scale, backward
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output, lse, *mask_tensors = ctx.saved_tensors
        # Restore the same BlockMask type without rerunning its constructor or
        # cloning any tensor. Forward's metadata snapshot is independent of
        # later attribute reassignment on the caller's mask object.
        block_mask = object.__new__(ctx.mask_type)
        for name, value in ctx.mask_metadata:
            setattr(block_mask, name, value)
        for name, tensor in zip(ctx.mask_tensor_names, mask_tensors):
            setattr(block_mask, name, tensor)
        dq, dk, dv = ctx.backward_kernel(grad_output.contiguous(), q, k, v, output, lse, block_mask, ctx.scale)
        return dq, dk, dv, None, None, None, None


def _validate(q, k, v, block_mask):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("FlexAttention expects q/k/v [B,H,S,D]")
    if k.shape[:3] != v.shape[:3] or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("FlexAttention q/k/v batch, KV length, and QK head dimensions must agree")
    if k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if block_mask is None:
        raise ValueError("FlexAttention requires a prebuilt BlockMask")


def flex_attention(q, k, v, block_mask=None, sm_scale=None, *, implementation=None):
    """Mojo interface: grouped [B,H,S,D] attention with a prebuilt BlockMask."""
    _validate(q, k, v, block_mask)
    return FlexAttentionFunction.apply(q, k, v, block_mask, sm_scale, implementation, "flex_attention")
