"""UC backend for ``MojoGroupRMSNorm``.

Group RMSNorm is the per-group RMSNorm with a shared trailing dimension
``norm_size``.  The dedicated kernel ``mojo_group_rmsnorm_{bf16,fp16}`` accepts
a stacked ``(G, M, N)`` tensor with ``(G, N)`` weight and produces a ``(G, M, N)``
output.  This wrapper is responsible for:

* validating that the kernel fast path is applicable (uniform dtype, uniform
  per-group shape, supported dtype, sane ``num_groups`` / ``norm_size``);
* flattening every group's leading dimensions into a 2-D ``(M, N)`` view,
  stacking them into a contiguous ``(G, M, N)`` tensor for the kernel call;
* unpacking the output back into a list of per-group tensors that match the
  original input shapes.

Anything outside the fast path defers to ``MojoGroupRMSNorm.forward``.
"""

from typing import List, Sequence

import torch

from mojo_opset.core import MojoGroupRMSNorm

from ._utils import _typed_api
from ._utils import _uc_kernels


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _can_use_kernel(
    op: MojoGroupRMSNorm,
    input_groups: Sequence[torch.Tensor],
) -> bool:
    if not isinstance(input_groups, (list, tuple)):
        return False
    if len(input_groups) != op.num_groups:
        return False
    if op.num_groups <= 0 or op.norm_size <= 0:
        return False
    if not op.elementwise_affine or op.weight is None:
        return False
    if op.weight.shape != (op.num_groups, op.norm_size):
        return False

    first = input_groups[0]
    if not isinstance(first, torch.Tensor):
        return False
    if first.dtype not in _SUPPORTED_DTYPES:
        return False
    if first.shape[-1] != op.norm_size:
        return False

    target_dtype = first.dtype
    target_shape = first.shape
    target_device = first.device
    for tensor in input_groups:
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.dtype != target_dtype:
            return False
        if tensor.shape != target_shape:
            return False
        if tensor.device != target_device:
            return False

    if op.weight.device != target_device:
        return False

    return True


class UCGroupRMSNorm(MojoGroupRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, input_groups) -> List[torch.Tensor]:
        if not _can_use_kernel(self, input_groups):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        dtype = input_groups[0].dtype
        original_shapes = [tensor.shape for tensor in input_groups]
        N = self.norm_size
        G = self.num_groups

        try:
            api = _typed_api("mojo_group_rmsnorm", dtype)
        except NotImplementedError:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # Flatten each group to (M, N).  When the input is itself 1-D we
        # treat it as ``(1, N)``.
        flat_groups = []
        first_M = None
        for tensor in input_groups:
            contig = tensor.contiguous()
            if contig.dim() == 0:
                # rms_norm over an empty trailing dim is not meaningful; fall
                # back to the reference implementation.
                raise NotImplementedError(
                    "UC backend cannot service this call (shape/dtype/contract not "
                    "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                    "(2026-06-08), this wrapper does not silently fall back to torch — "
                    "use TTX / torch_npu / torch_native backend for unsupported inputs."
                )
            if contig.dim() == 1:
                flat = contig.reshape(1, contig.numel())
            else:
                flat = contig.reshape(-1, N)
            if first_M is None:
                first_M = flat.shape[0]
            elif flat.shape[0] != first_M:
                # Per-group ``M`` varies – kernel requires uniform shape.
                raise NotImplementedError(
                    "UC backend cannot service this call (shape/dtype/contract not "
                    "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                    "(2026-06-08), this wrapper does not silently fall back to torch — "
                    "use TTX / torch_npu / torch_native backend for unsupported inputs."
                )
            flat_groups.append(flat)

        if first_M is None or first_M == 0:
            # No work to do (e.g. empty group). Defer to reference for
            # correctness of bookkeeping (empty output preserved).
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        stacked_x = torch.stack(flat_groups, dim=0).contiguous()
        weight = self.weight
        if weight.dtype != dtype:
            weight = weight.to(dtype)
        weight = weight.contiguous()

        stacked_y = torch.empty_like(stacked_x)
        eps = float(self.variance_epsilon)

        _uc_kernels()[api](stacked_x, weight, stacked_y, G, first_M, N, eps)

        output_groups: List[torch.Tensor] = []
        for g_idx, original_shape in enumerate(original_shapes):
            output_groups.append(stacked_y[g_idx].reshape(original_shape))
        return output_groups
