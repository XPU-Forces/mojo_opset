from typing import Optional

import torch

from mojo_opset.core import MojoStorePagedKVCache


class RefStorePagedKVCache(MojoStorePagedKVCache):
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
                raise ValueError(
                    f"Expected TND when is_varlen=True; got shapes key={tuple(key.shape)}, value={tuple(value.shape)}, key_cache={tuple(key_cache.shape)}, value_cache={tuple(value_cache.shape)}"
                )
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
                raise ValueError(
                    f"Expected BNSD when is_varlen=False; got shapes key={tuple(key.shape)}, value={tuple(value.shape)}, key_cache={tuple(key_cache.shape)}, value_cache={tuple(value_cache.shape)}"
                )
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
