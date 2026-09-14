from typing import Optional
from typing import Tuple

import torch

from mojo_opset.core import MojoApplyRoPE
from mojo_opset.core import MojoRotaryEmbedding
from mojo_opset.utils.logging import get_logger

from ._utils import run_kernel


logger = get_logger(__name__)


class UCRotaryEmbedding(MojoRotaryEmbedding):
    supported_platforms_list = ["npu"]

    def __init__(
        self,
        rope_theta,
        rope_dim,
        attention_scaling: float = 1.0,
        init_max_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(rope_theta, rope_dim, attention_scaling, init_max_length, **kwargs)
        if init_max_length is None:
            raise ValueError("init_max_length must be provided for UCRotaryEmbedding")

    def forward(
        self,
        x: torch.Tensor,
        cu_q_lens: Optional[torch.Tensor] = None,
        total_seq_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cu_q_lens is not None:
            assert cu_q_lens.dtype == torch.int32
        if total_seq_lens is not None:
            assert total_seq_lens.dtype == torch.int32
        if position_ids is not None:
            assert position_ids.dtype == torch.int32
        assert position_ids is None or cu_q_lens is None, "At most one of cu_q_lens or position_ids should be provided"

        if cu_q_lens is not None:
            logger.warning_once(
                "UC rotary embedding does not support varlen mode yet; falling back to torch implementation."
            )
            return super().forward(x, cu_q_lens, total_seq_lens, position_ids)
        elif position_ids is not None:
            assert position_ids.shape == x.shape[:-1], "position_ids must have the same shape as x except the hidden dimension"
            position_ids = position_ids.contiguous()
        else:
            return self._arange_cache(x)

        return self._position_ids_cache(position_ids)

    def _arange_cache(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() < 2:
            raise AssertionError("x must have at least two dimensions for padded prefill rotary embedding")

        seq_len = x.shape[1]
        rope_dim = self.cos.shape[-1]
        if seq_len > self.cos.shape[0]:
            raise ValueError(f"seq_len {seq_len} exceeds rotary cache length {self.cos.shape[0]}")

        cos_out = torch.empty((seq_len, rope_dim), device=self.cos.device, dtype=self.cos.dtype)
        sin_out = torch.empty((seq_len, rope_dim), device=self.sin.device, dtype=self.sin.dtype)
        if seq_len == 0 or rope_dim == 0:
            return cos_out, sin_out

        run_kernel(
            "mojo_rotary_embedding_arange",
            self.cos.dtype,
            self.cos.contiguous(),
            self.sin.contiguous(),
            cos_out,
            sin_out,
            self.cos.shape[0],
            rope_dim,
            seq_len,
        )
        return cos_out, sin_out

    def _position_ids_cache(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_dim = self.cos.shape[-1]
        rows = position_ids.numel()
        output_shape = tuple(position_ids.shape) + (rope_dim,)
        cos_out = torch.empty(output_shape, device=self.cos.device, dtype=self.cos.dtype)
        sin_out = torch.empty(output_shape, device=self.sin.device, dtype=self.sin.dtype)
        if rows == 0 or rope_dim == 0:
            return cos_out, sin_out

        run_kernel(
            "mojo_rotary_embedding_position_ids",
            self.cos.dtype,
            self.cos.contiguous(),
            self.sin.contiguous(),
            position_ids.reshape(-1).contiguous(),
            cos_out.reshape(rows, rope_dim),
            sin_out.reshape(rows, rope_dim),
            self.cos.shape[0],
            rope_dim,
            rows,
        )
        return cos_out, sin_out


class UCApplyRoPE(MojoApplyRoPE):
    supported_platforms_list = ["npu"]
    _STATIC_APPLY_ROPE_KERNELS = frozenset(
        {
            (96, 96, torch.float16),
            (96, 32, torch.bfloat16),
            (88, 88, torch.bfloat16),
            (128, 48, torch.float16),
            (128, 128, torch.float16),
        }
    )

    @staticmethod
    def _normalize_to_token_head(
        q: torch.Tensor,
        k: torch.Tensor,
        head_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[int]]:
        if q.ndim == 3:
            if head_first:
                q = q.transpose(0, 1).contiguous()
                k = k.transpose(0, 1).contiguous()
            else:
                q = q.contiguous()
                k = k.contiguous()
            return q, k, None, None

        if head_first:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
        else:
            q = q.contiguous()
            k = k.contiguous()

        batch_size, seq_len = q.shape[0], q.shape[1]
        q = q.reshape(batch_size * seq_len, q.shape[2], q.shape[3]).contiguous()
        k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3]).contiguous()
        return q, k, batch_size, seq_len

    @staticmethod
    def _restore_from_token_head(
        x: torch.Tensor,
        original_shape: torch.Size,
        head_first: bool,
        batch_size: Optional[int],
        seq_len: Optional[int],
    ) -> torch.Tensor:
        if len(original_shape) == 3:
            if head_first:
                x = x.transpose(0, 1).contiguous()
            return x.reshape(original_shape)

        x = x.reshape(batch_size, seq_len, x.shape[1], x.shape[2])
        if head_first:
            x = x.transpose(1, 2).contiguous()
        return x.reshape(original_shape)

    @staticmethod
    def _run_static_token_head_kernel(
        api: str,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *shape_args: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_out = q.clone(memory_format=torch.contiguous_format)
        k_out = k.clone(memory_format=torch.contiguous_format)
        run_kernel(api, q.dtype, q, k, cos, sin, q_out, k_out, *shape_args)
        return q_out, k_out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_first: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q.ndim == k.ndim, "q and k must have the same dimension"
        assert q.ndim == 3 or q.ndim == 4, "q and k must be 3D or 4D"
        assert cos.shape == sin.shape, "cos and sin must have the same shape"
        if q.ndim == 3:
            assert cos.ndim == 2, "3D q/k inputs expect 2D cos/sin"
        elif cos.ndim not in (2, 3):
            raise ValueError("4D q/k inputs expect 2D or 3D cos/sin")

        if q.dtype != k.dtype:
            raise ValueError(f"q and k must have the same dtype, got {q.dtype} and {k.dtype}.")
        if cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise NotImplementedError("UC backend MojoApplyRoPE expects float32 cos/sin tensors.")

        if q.numel() == 0 or k.numel() == 0:
            return torch.empty_like(q), torch.empty_like(k)

        rope_dim = cos.shape[-1]
        q_norm, k_norm, batch_size, seq_len = self._normalize_to_token_head(q, k, head_first)
        rows, q_heads, head_dim = q_norm.shape
        k_rows, k_heads, k_head_dim = k_norm.shape
        if rows != k_rows or head_dim != k_head_dim:
            raise ValueError("q and k must have matching token count and head dimension")

        config_key = (head_dim, rope_dim, q_norm.dtype)
        if config_key not in self._STATIC_APPLY_ROPE_KERNELS:
            raise NotImplementedError(
                "UC backend MojoApplyRoPE does not provide an aligned static kernel for "
                f"head_dim={head_dim}, rope_dim={rope_dim}, dtype={q_norm.dtype}."
            )

        if cos.ndim == 2:
            cos_kind = "cos2d"
            cos_kernel = cos.contiguous()
            sin_kernel = sin.contiguous()
            shape_args = (rows, q_heads, k_heads, cos.shape[0])
        else:
            cos_kind = "costoken"
            cos_kernel = cos.reshape(rows, rope_dim).contiguous()
            sin_kernel = sin.reshape(rows, rope_dim).contiguous()
            shape_args = (rows, q_heads, k_heads)

        api = f"mojo_apply_rope_tnh_d{head_dim}_r{rope_dim}_{cos_kind}"
        q_out, k_out = self._run_static_token_head_kernel(
            api, q_norm, k_norm, cos_kernel, sin_kernel, *shape_args
        )
        return (
            self._restore_from_token_head(q_out, q.shape, head_first, batch_size, seq_len),
            self._restore_from_token_head(k_out, k.shape, head_first, batch_size, seq_len),
        )
