import os
import torch
from typing import Optional

from ..mojo_operator import MojoOperator
from .. import VALID_KV_LAYOUTS


class MojoStoreKVCache(MojoOperator):
    pass


class MojoStorePagedKVCache(MojoOperator):
    def __init__(
        self,
        kv_layout: str = VALID_KV_LAYOUTS[0],
        kv_dim: int = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        Common parameter definitions for StoreKVPaged.

        Init parameters:
        - kv_layout (str): KV computation layout, values defined by VALID_KV_LAYOUTS, default VALID_KV_LAYOUTS[0].
        - kv_dim (int): KV hidden dimension D_kv, used in scenarios like MLA to distinguish KV compression dimension from K dimension; positive integer.
        - is_varlen (bool): When True, prioritize TND continuous token perspective; when False, use BNSD; default True.
        - op_name (str): Operator name placeholder.

        Scope and description:
        - Only covers common parameters; does not involve paging strategy details and quantization parameters (QuantMode/QuantParam).
        """
        super().__init__(op_name)
        if kv_layout not in VALID_KV_LAYOUTS:
            raise ValueError(f"kv_layout must be one of {VALID_KV_LAYOUTS}, got {kv_layout}")
        if kv_dim is None or not isinstance(kv_dim, int) or kv_dim <= 0:
            raise ValueError("kv_dim must be a positive integer")
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen must be a boolean type")
        self.kv_layout = kv_layout
        self.kv_dim = kv_dim
        self.is_varlen = is_varlen

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Forward parameters (general level):
        - key: Shape [B, S_new, H_kv, D_kv], dtype float16/bfloat16.
        - value: Shape [B, S_new, H_kv, D_kv], dtype float16/bfloat16.
        - key_cache: Shape [B, S_cap, H_kv, D_kv], dtype same as key; used for writing.
        - value_cache: Shape [B, S_cap, H_kv, D_kv], dtype same as value; used for writing.
        - kv_lens: Optional, shape [B], dtype=int32, represents historical stored length.
        - block_table: Optional, placeholder [B, T] (T>=1), dtype=int32; only for paging placeholder, does not implement page management.

        Validation and constraints:
        - dtype: key/value must be float16 or bfloat16; cache must match.
        - Shape consistency: key/value batch, H_kv, D_kv must be consistent; cache H_kv, D_kv must match input; S_new and kv_lens increment logic is not handled in this method.
        - kv_lens: If provided, must be [B], int32, elements non-negative and not exceeding S_cap.
        - block_table: If provided, must be [B,T], int32, T>=1, elements non-negative.
        """
        
        # Write logic placeholder
        raise NotImplementedError("MojoStorePagedKVCache forward only performs general parameter validation, does not contain specific write logic")

    def forward_ref(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Reference implementation (golden): Write new key/value to cache according to general semantics, strictly distinguish TND/BNSD inputs.
        Input layout contract:
        - When is_varlen=True (TND): Only accept key/value/key_cache/value_cache as [T, H_kv, D_kv]
          · Otherwise raise: ValueError("Expected TND when is_varlen=True; got shape ...")
        - When is_varlen=False (BNSD): Only accept key/value as [B, S_new, H_kv, D_kv], cache as [B, S_cap, H_kv, D_kv]
          · Otherwise raise: ValueError("Expected BNSD when is_varlen=False; got shape ...")
        Formula semantics (does not involve block_table paging details, only placeholder validation):
        - TND: offset = int(kv_lens) (0 if not provided);
          * key_cache[offset:offset+T_new, :, :] = key[:T_new, :, :]
          * value_cache[offset:offset+T_new, :, :] = value[:T_new, :, :]
        - BNSD: For each batch b, offset = kv_lens[b] (0 if not provided); write to [offset, offset+S_new).
        Returns: None.
        """
        # Common dtype/dimension check
        if key.dtype != value.dtype or key_cache.dtype != value_cache.dtype or key.dtype != key_cache.dtype:
            raise TypeError("key/value and cache dtypes must be consistent")
        if key.shape[-1] != self.kv_dim or value.shape[-1] != self.kv_dim:
            raise ValueError("key/value last dimension must equal kv_dim")
        if key_cache.shape[-1] != self.kv_dim or value_cache.shape[-1] != self.kv_dim:
            raise ValueError("cache last dimension must equal kv_dim")
        if self.is_varlen:
            # Only accept TND
            if not (key.ndim == value.ndim == key_cache.ndim == value_cache.ndim == 3):
                raise ValueError(f"Expected TND when is_varlen=True; got shapes key={tuple(key.shape)}, value={tuple(value.shape)}, key_cache={tuple(key_cache.shape)}, value_cache={tuple(value_cache.shape)}")
            T_new, Hkv, Dkv = key.shape
            T_cap, Hc, Dc = key_cache.shape
            if Hc != Hkv or Dc != Dkv:
                raise ValueError("cache H/D must match input")
            if value.shape != key.shape or value_cache.shape != key_cache.shape:
                raise ValueError("value shape must match key shape, value_cache shape must match key_cache shape")
            # kv_lens can be None or 0-D/1-D single-element integer tensor
            if kv_lens is None:
                offset = 0
            else:
                if kv_lens.ndim not in (0, 1):
                    raise ValueError("kv_lens must be scalar or single-element tensor in TND mode")
                offset = int(kv_lens.item()) if kv_lens.numel() == 1 else int(kv_lens[0].item())
                if offset < 0 or offset > T_cap:
                    raise ValueError("kv_lens must be in [0, T_cap]")
            end = offset + T_new
            if end > T_cap:
                raise ValueError("Write out of bounds: offset+T_new exceeds T_cap")
            key_cache[offset:end, :, :] = key[:, :, :]
            value_cache[offset:end, :, :] = value[:, :, :]
            return None
        else:
            # Only accept BNSD
            if not (key.ndim == value.ndim == key_cache.ndim == value_cache.ndim == 4):
                raise ValueError(f"Expected BNSD when is_varlen=False; got shapes key={tuple(key.shape)}, value={tuple(value.shape)}, key_cache={tuple(key_cache.shape)}, value_cache={tuple(value_cache.shape)}")
            B, S_new, Hkv, Dkv = key.shape
            Bc, S_cap, Hc, Dc = key_cache.shape
            if Bc != B or Hc != Hkv or Dc != Dkv:
                raise ValueError("cache B/H/D must match input")
            if value.shape != key.shape or value_cache.shape != key_cache.shape:
                raise ValueError("value shape must match key shape, value_cache shape must match key_cache shape")
            if kv_lens is None:
                kv_lens = torch.zeros((B,), dtype=torch.int32, device=key.device)
            else:
                if kv_lens.ndim != 1 or kv_lens.shape[0] != B:
                    raise ValueError("kv_lens must be [B]")
                if kv_lens.dtype not in (torch.int32, torch.int64):
                    raise TypeError("kv_lens must be integer type")
                if torch.any(kv_lens < 0) or torch.any(kv_lens > S_cap):
                    raise ValueError("kv_lens elements must be in [0, S_cap]")
            for b in range(B):
                offset = int(kv_lens[b].item())
                end = offset + S_new
                if end > S_cap:
                    raise ValueError(f"Write out of bounds: batch {b} offset+S_new exceeds S_cap")
                key_cache[b, offset:end, :, :] = key[b, :, :, :]
                value_cache[b, offset:end, :, :] = value[b, :, :, :]
            return None


class MojoStoreMLAKVCache(MojoOperator):
    pass


class MojoStorePagedMLAKVCache(MojoOperator):
    pass
