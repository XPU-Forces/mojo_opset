import math
from typing import Optional

import torch

from mojo_opset.core import MojoSdpa

from ._utils import _typed_api
from ._utils import _uc_kernels


_SUPPORTED_DTYPES = (torch.bfloat16,)
_SUPPORTED_HEAD_DIM = 128

_KERNEL_BY_SHAPE = {
    (1, 5, 1, 4096, 128): "mojo_sdpa_b1_qh5_kvh1_s4096_d128",
}


def _is_default_scale(scale: Optional[float], head_dim: int) -> bool:
    if scale is None:
        return True
    default_scale = 1.0 / math.sqrt(head_dim)
    return math.isclose(float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-12)


def _static_kernel_api(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    scale: Optional[float],
    enable_gqa: bool,
) -> Optional[str]:
    if attn_mask is not None:
        return None
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return None
    if query.dtype not in _SUPPORTED_DTYPES:
        return None
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return None
    if key.device != query.device or value.device != query.device:
        return None

    batch, q_heads, q_seq, head_dim = query.shape
    k_batch, kv_heads, k_seq, k_head_dim = key.shape
    v_batch, v_heads, v_seq, v_head_dim = value.shape

    if batch != k_batch or batch != v_batch:
        return None
    if kv_heads != v_heads:
        return None
    if q_seq != k_seq or q_seq != v_seq:
        return None
    if head_dim != k_head_dim or head_dim != v_head_dim:
        return None
    if head_dim != _SUPPORTED_HEAD_DIM:
        return None
    if q_heads != kv_heads and (not enable_gqa or q_heads % kv_heads != 0):
        return None
    if not _is_default_scale(scale, head_dim):
        return None

    return _KERNEL_BY_SHAPE.get((batch, q_heads, kv_heads, q_seq, head_dim))


class UCSdpa(MojoSdpa):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if query.numel() == 0:
            return torch.empty_like(query)

        api = _static_kernel_api(query, key, value, attn_mask, self.scale, self.enable_gqa)
        if api is None:
            return super().forward(query, key, value, attn_mask)

        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        out = torch.empty_like(q)

        try:
            kernel = _uc_kernels()[_typed_api(api, q.dtype)]
        except NotImplementedError:
            return super().forward(query, key, value, attn_mask)
        kernel(q, k, v, out)
        return out.reshape(query.shape)
