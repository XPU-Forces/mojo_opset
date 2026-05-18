from typing import Optional
from typing import Tuple

import torch

from mojo_opset.core import MojoApplyRoPE
from mojo_opset.core import MojoRotaryEmbedding

from ._utils import run_kernel


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

        if cu_q_lens is not None or position_ids is not None:
            # TODO(tilelang-uc): replace with a UC gather kernel once BufferLoad-backed DRAM indices are supported.
            return super().forward(x, cu_q_lens, total_seq_lens, position_ids)
        return self._arange_cache(x)

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
            seq_len,
            rope_dim,
        )
        return cos_out, sin_out


class UCApplyRoPE(MojoApplyRoPE):
    supported_platforms_list = ["npu"]

    @staticmethod
    def _normalize_to_token_head(
        q: torch.Tensor,
        k: torch.Tensor,
        head_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[int]]:
        if q.ndim == 3:
            if head_first:
                q_norm = q.transpose(0, 1).contiguous()
                k_norm = k.transpose(0, 1).contiguous()
            else:
                q_norm = q.contiguous()
                k_norm = k.contiguous()
            return q_norm, k_norm, None, None

        if head_first:
            q_norm = q.transpose(1, 2).contiguous()
            k_norm = k.transpose(1, 2).contiguous()
        else:
            q_norm = q.contiguous()
            k_norm = k.contiguous()

        batch_size, seq_len, q_heads, head_dim = q_norm.shape
        k_heads = k_norm.shape[2]
        q_norm = q_norm.reshape(batch_size * seq_len, q_heads, head_dim).contiguous()
        k_norm = k_norm.reshape(batch_size * seq_len, k_heads, head_dim).contiguous()
        return q_norm, k_norm, batch_size, seq_len

    @staticmethod
    def _restore_layout(
        x: torch.Tensor,
        original_shape: torch.Size,
        head_first: bool,
        batch_size: Optional[int],
        seq_len: Optional[int],
    ) -> torch.Tensor:
        if len(original_shape) == 3:
            if head_first:
                return x.transpose(0, 1).contiguous().reshape(original_shape)
            return x.reshape(original_shape)

        assert batch_size is not None and seq_len is not None
        x = x.reshape(batch_size, seq_len, x.shape[1], x.shape[2])
        if head_first:
            x = x.transpose(1, 2).contiguous()
        return x.reshape(original_shape)

    @staticmethod
    def _flatten_cos_sin(
        cos: torch.Tensor,
        sin: torch.Tensor,
        rows: int,
        batch_size: Optional[int],
        seq_len: Optional[int],
        half_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise NotImplementedError("UC backend MojoApplyRoPE expects float32 cos/sin tensors.")

        if cos.ndim == 2:
            if cos.shape[0] == rows:
                cos_flat = cos
                sin_flat = sin
            elif batch_size is not None and seq_len is not None and cos.shape[0] == seq_len:
                cos_flat = cos.unsqueeze(0).expand(batch_size, seq_len, cos.shape[1]).reshape(rows, cos.shape[1])
                sin_flat = sin.unsqueeze(0).expand(batch_size, seq_len, sin.shape[1]).reshape(rows, sin.shape[1])
            else:
                raise ValueError(
                    f"cos/sin first dimension must be {rows} or padded seq_len {seq_len}, got {cos.shape}."
                )
        elif cos.ndim == 3:
            if batch_size is None or seq_len is None:
                raise ValueError("3D cos/sin are only valid for 4D q/k inputs.")
            if cos.shape[0] != batch_size or cos.shape[1] != seq_len:
                raise ValueError(f"3D cos/sin must have shape [B, S, D], got {cos.shape}.")
            cos_flat = cos.reshape(rows, cos.shape[-1])
            sin_flat = sin.reshape(rows, sin.shape[-1])
        else:
            raise ValueError(f"cos/sin must be 2D or 3D, got {cos.ndim}D.")

        return cos_flat[:, :half_dim].contiguous(), sin_flat[:, :half_dim].contiguous()

    @staticmethod
    def _apply_one(
        x: torch.Tensor,
        cos_half: torch.Tensor,
        sin_half: torch.Tensor,
        rope_dim: int,
    ) -> torch.Tensor:
        rows, heads, head_dim = x.shape
        half_dim = rope_dim // 2
        nope_dim = head_dim - rope_dim

        if half_dim == 0:
            return x.clone(memory_format=torch.contiguous_format)

        x_rope = x[..., nope_dim:]
        x1 = x_rope[..., :half_dim].contiguous()
        x2 = x_rope[..., half_dim:].contiguous()
        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)

        run_kernel(
            "mojo_apply_rope",
            x.dtype,
            x1,
            x2,
            cos_half,
            sin_half,
            y1,
            y2,
            rows,
            heads,
            half_dim,
        )

        if nope_dim > 0:
            return torch.cat((x[..., :nope_dim], y1, y2), dim=-1)
        return torch.cat((y1, y2), dim=-1)

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
        assert q.shape[-1] == k.shape[-1], "q and k must have the same head dimension"

        if q.numel() == 0 or k.numel() == 0:
            return torch.empty_like(q), torch.empty_like(k)

        rope_dim = cos.shape[-1]
        head_dim = q.shape[-1]
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}.")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim {rope_dim} exceeds head_dim {head_dim}.")

        q_norm, k_norm, batch_size, seq_len = self._normalize_to_token_head(q, k, head_first)
        if q_norm.shape[0] != k_norm.shape[0]:
            raise ValueError(f"q and k token counts must match, got {q_norm.shape[0]} and {k_norm.shape[0]}.")

        cos_half, sin_half = self._flatten_cos_sin(
            cos,
            sin,
            q_norm.shape[0],
            batch_size,
            seq_len,
            rope_dim // 2,
        )

        q_out = self._apply_one(q_norm, cos_half, sin_half, rope_dim)
        k_out = self._apply_one(k_norm, cos_half, sin_half, rope_dim)

        return (
            self._restore_layout(q_out, q.shape, head_first, batch_size, seq_len),
            self._restore_layout(k_out, k.shape, head_first, batch_size, seq_len),
        )
