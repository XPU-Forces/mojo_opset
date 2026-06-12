from typing import Optional, Tuple

import torch

from ...core.operator import MojoOperator
from ...core.operators.normalization import MojoGroupRMSNorm
from ...core.operators.position_embedding import MojoApplyRoPE
from ...core.operators.quantize import MojoStaticQuant
from ...core.operators.kv_cache import MojoStorePagedKVCache

__all__ = ["MojoFusedNormRoPEQuantStore"]


class MojoFusedNormRoPEQuantStore(MojoOperator):
    """Fused QK-Norm + RoPE + KV-StaticQuant + PagedKVStore.

    Combines GroupRMSNorm (on Q and K), RoPE (on Q and K), StaticQuant (on K
    and V), and paged KV cache store into a single operator to eliminate
    intermediate tensor materializations.

    When ``update_kv=False``, only Q norm+rope is computed — the K/V quant and
    store paths are skipped entirely (useful for YOCO reuse layers).

    Sub-operations are composed from existing Mojo core operators:
      - ``MojoGroupRMSNorm``: per-head QK norm
      - ``MojoApplyRoPE``: rotary position embedding
      - ``MojoStaticQuant``: static per-channel int8 quantization
      - ``MojoStorePagedKVCache``: paged KV cache store
    """

    def __init__(
        self,
        num_heads_swa_q: int,
        num_heads_swa_k: int,
        num_heads_full_q: int,
        num_heads_full_k: int,
        head_dim: int,
        norm_eps: float = 1e-5,
        use_query_norm: bool = True,
        use_key_norm: bool = True,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads_swa_q = num_heads_swa_q
        self.num_heads_swa_k = num_heads_swa_k
        self.num_heads_full_q = num_heads_full_q
        self.num_heads_full_k = num_heads_full_k
        self.head_dim = head_dim
        self.use_query_norm = use_query_norm
        self.use_key_norm = use_key_norm

        num_norm_groups = (2 if use_query_norm else 0) + (2 if use_key_norm else 0)
        if num_norm_groups > 0:
            self.qk_norm = MojoGroupRMSNorm._registry.get(self._backend)(num_norm_groups, head_dim, eps=norm_eps)
        else:
            self.qk_norm = None

        self.apply_rope = MojoApplyRoPE._registry.get(self._backend)()

        self.full_k_quantize = MojoStaticQuant._registry.get(self._backend)((num_heads_full_k, head_dim), quant_dtype=quant_dtype)
        self.full_v_quantize = MojoStaticQuant._registry.get(self._backend)((num_heads_full_k, head_dim), quant_dtype=quant_dtype)
        self.swa_k_quantize = MojoStaticQuant._registry.get(self._backend)((num_heads_swa_k, head_dim), quant_dtype=quant_dtype)
        self.swa_v_quantize = MojoStaticQuant._registry.get(self._backend)((num_heads_swa_k, head_dim), quant_dtype=quant_dtype)

        self.store_full_kvcache = MojoStorePagedKVCache._registry.get(self._backend)()
        self.store_swa_kvcache = MojoStorePagedKVCache._registry.get(self._backend)()

    def forward(
        self,
        swa_query: torch.Tensor,
        swa_key: torch.Tensor,
        swa_value: torch.Tensor,
        full_query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        full_key_cache: torch.Tensor,
        full_value_cache: torch.Tensor,
        swa_key_cache: torch.Tensor,
        swa_value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        cu_q_lens: Optional[torch.Tensor],
        context_kv_lens: torch.Tensor,
        block_tables_sparse: torch.Tensor,
        cu_q_lens_sparse: Optional[torch.Tensor],
        context_kv_lens_sparse: torch.Tensor,
        update_kv: bool = True,
    ) -> Tuple[
        torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
    ]:
        """
        Args:
            swa_query: [T, swa_nh_q, head_dim] pre-norm SWA query
            swa_key: [T, swa_nkv, head_dim] pre-norm SWA key
            swa_value: [T, swa_nkv, head_dim] SWA value (no norm/rope)
            full_query: [T, full_nh_q, head_dim] pre-norm full query
            full_key: [T, full_nkv, head_dim] pre-norm full key
            full_value: [T, full_nkv, head_dim] full value (no norm/rope)
            cos: [T, rope_dim] RoPE cosine
            sin: [T, rope_dim] RoPE sine
            full_key_cache: paged KV cache for full attention keys
            full_value_cache: paged KV cache for full attention values
            swa_key_cache: paged KV cache for SWA keys
            swa_value_cache: paged KV cache for SWA values
            block_tables: [B, max_blocks] block table for full attn
            cu_q_lens: [B+1] cumulative query lengths (None for decode)
            context_kv_lens: [B] existing KV lengths before store
            block_tables_sparse: [B, max_blocks] block table for SWA
            cu_q_lens_sparse: [B+1] cumulative query lengths for SWA
            context_kv_lens_sparse: [B] existing KV lengths for SWA
            update_kv: if True, run full pipeline (norm+rope+quant+store);
                       if False, only compute Q norm+rope (skip K/V entirely)

        Returns:
            Tuple of:
              - swa_query_out: [T, swa_nh_q, head_dim] post-norm+rope
              - full_query_out: [T, full_nh_q, head_dim] post-norm+rope
              - full_key_out: [T, full_nkv, head_dim] int8 (None if update_kv=False)
              - full_k_scale: [full_nkv, head_dim] (None if update_kv=False)
              - swa_key_out: [T, swa_nkv, head_dim] int8 (None if update_kv=False)
              - swa_k_scale: [swa_nkv, head_dim] (None if update_kv=False)
              - full_value_out: [T, full_nkv, head_dim] int8 (None if update_kv=False)
              - full_v_scale: [full_nkv, head_dim] (None if update_kv=False)
              - swa_value_out: [T, swa_nkv, head_dim] int8 (None if update_kv=False)
              - swa_v_scale: [swa_nkv, head_dim] (None if update_kv=False)
        """
        # --- 1. GroupRMSNorm on Q and/or K ---
        if self.use_query_norm and self.use_key_norm:
            swa_query, swa_key, full_query, full_key = self.qk_norm(
                [swa_query, swa_key, full_query, full_key]
            )
        elif self.use_query_norm:
            swa_query, full_query = self.qk_norm([swa_query, full_query])
        elif self.use_key_norm:
            swa_key, full_key = self.qk_norm([swa_key, full_key])

        # --- 2. RoPE on Q (always) and K (only if update_kv) ---
        if update_kv:
            swa_query, swa_key = self.apply_rope(swa_query, swa_key, cos, sin, head_first=False)
            full_query, full_key = self.apply_rope(full_query, full_key, cos, sin, head_first=False)
        else:
            # Only Q needs rope; pass Q as both q and k, discard second output
            swa_query, _ = self.apply_rope(swa_query, swa_query, cos, sin, head_first=False)
            full_query, _ = self.apply_rope(full_query, full_query, cos, sin, head_first=False)
            return (
                swa_query, full_query,
                None, None, None, None, None, None, None, None,
            )

        # --- 3. StaticQuant on K and V ---
        full_key_q, full_k_scale = self.full_k_quantize(full_key)
        full_val_q, full_v_scale = self.full_v_quantize(full_value)
        swa_key_q, swa_k_scale = self.swa_k_quantize(swa_key)
        swa_val_q, swa_v_scale = self.swa_v_quantize(swa_value)

        # --- 4. Store to paged KV cache ---
        self.store_full_kvcache(
            full_key_q, full_val_q,
            full_key_cache, full_value_cache,
            block_tables, cu_q_lens, context_kv_lens,
        )
        self.store_swa_kvcache(
            swa_key_q, swa_val_q,
            swa_key_cache, swa_value_cache,
            block_tables_sparse, cu_q_lens_sparse, context_kv_lens_sparse,
        )

        return (
            swa_query, full_query,
            full_key_q, full_k_scale,
            swa_key_q, swa_k_scale,
            full_val_q, full_v_scale,
            swa_val_q, swa_v_scale,
        )

    def extra_repr(self) -> str:
        return (
            f"num_heads_swa_q={self.num_heads_swa_q}, "
            f"num_heads_swa_k={self.num_heads_swa_k}, "
            f"num_heads_full_q={self.num_heads_full_q}, "
            f"num_heads_full_k={self.num_heads_full_k}, "
            f"head_dim={self.head_dim}, "
            f"use_query_norm={self.use_query_norm}, "
            f"use_key_norm={self.use_key_norm}"
        )
