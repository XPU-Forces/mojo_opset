import math
from typing import Optional

import torch

from mojo_opset.core import MojoSdpa
from mojo_opset.utils.logging import get_logger

from ._utils import _typed_api
from ._utils import _uc_kernels


logger = get_logger(__name__)

_SUPPORTED_DTYPES = (torch.bfloat16,)
_KERNEL_BY_SHAPE = {
    (1, 5, 1, 4096, 128): "mojo_sdpa_b1_qh5_kvh1_s4096_d128",
}


def _is_default_scale(scale: Optional[float], head_dim: int) -> bool:
    if scale is None:
        return True
    default_scale = 1.0 / math.sqrt(head_dim)
    return math.isclose(float(scale), default_scale, rel_tol=1e-6, abs_tol=1e-12)


def _assert_sdpa_contract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    enable_gqa: bool,
) -> None:
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4
    assert query.shape[0] == key.shape[0] == value.shape[0]
    assert query.shape[-1] == key.shape[-1] and key.shape[-1] == value.shape[-1]
    head_dim = query.shape[-1]
    assert head_dim in {64, 128}
    assert query.shape[-2] == key.shape[-2] and key.shape[-2] == value.shape[-2]

    if not enable_gqa:
        assert query.shape[1] == key.shape[1] and key.shape[1] == value.shape[1]
    else:
        assert key.shape[1] == value.shape[1] and query.shape[1] % key.shape[1] == 0

    assert query.dtype == key.dtype == value.dtype


def _assert_sdpa_mask(attn_mask: torch.Tensor, seq_length: int) -> None:
    assert len(attn_mask.shape) == 2 and attn_mask.shape[0] == seq_length and attn_mask.shape[1] == seq_length
    assert attn_mask.dtype == torch.bool


def _assert_uc_static_kernel_contract(query: torch.Tensor, scale: Optional[float]) -> None:
    head_dim = query.shape[-1]
    assert query.dtype in _SUPPORTED_DTYPES
    assert _is_default_scale(scale, head_dim)


def _static_kernel_api(query: torch.Tensor, key: torch.Tensor) -> str:
    batch, q_heads, q_seq, head_dim = query.shape
    kv_heads = key.shape[1]
    api = _KERNEL_BY_SHAPE.get((batch, q_heads, kv_heads, q_seq, head_dim))
    if api is None:
        raise NotImplementedError(
            "UC SDPA only supports static shape "
            "(batch=1, q_heads=5, kv_heads=1, seq=4096, head_dim=128) for now, "
            f"got {(batch, q_heads, kv_heads, q_seq, head_dim)}."
        )
    return api


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

        _assert_sdpa_contract(query, key, value, self.enable_gqa)
        if attn_mask is not None:
            _assert_sdpa_mask(attn_mask, query.shape[-2])
            logger.warning_once(
                "UC SDPA does not support attention masks yet; falling back to torch implementation."
            )
            return super().forward(query, key, value, attn_mask)

        _assert_uc_static_kernel_contract(query, self.scale)
        api = _static_kernel_api(query, key)

        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        out = torch.empty_like(q)

        kernel = _uc_kernels()[_typed_api(api, q.dtype)]
        kernel(q, k, v, out)
        return out.reshape(query.shape)
