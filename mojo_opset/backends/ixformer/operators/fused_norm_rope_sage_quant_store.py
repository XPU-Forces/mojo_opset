import torch

from ixformer import functions as ixf_f
from mojo_opset.experimental import MojoFusedNormRoPESageQuantStore


class IxformerFusedNormRoPESageQuantStore(MojoFusedNormRoPESageQuantStore):
    """Ixformer backend for SAGE-aware fused QK-Norm + RoPE + KV-Quant + Store.

      - SWA stream: ``rms_norm_qk_rotary_embedding_quant_kv_and_store``.

      - Full stream: when SAGE is off, the same
        ``rms_norm_qk_rotary_embedding_quant_kv_and_store``; when SAGE is on, the
        ``rms_norm_sage_qk_rotary_embedding`` kernel, which performs the exact
        same norm + rope + static K/V quant + paged store and additionally
        produces the per-token dynamic-int8 view of the key and its per-token
        scale, storing both into their own paged caches
        (``sage_full_k_pt_cache`` / ``sage_full_k_pt_scale_cache``) inline,
        exactly like ``key_cache`` / ``value_cache`` — all in one fused kernel —
        and also returns ``(output_q, key_quant, value_quant, key_pt_int8,
        key_pt_scale)``.
    """

    supported_platforms_list = ["ilu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.head_dim != 128:
            raise NotImplementedError(
                f"ixformer fused kernel only supports head_dim=128, got {self.head_dim}"
            )
        if not (self.use_query_norm and self.use_key_norm):
            raise NotImplementedError(
                f"ixformer fused kernel only supports use_query_norm and use_key_norm, got {self.use_query_norm} and {self.use_key_norm}"
            )

        self.register_load_state_dict_post_hook(self._convert_norm_weight_to_bf16)

    @staticmethod
    def _convert_norm_weight_to_bf16(module, incompatible_keys):
        if module.qk_norm is not None:
            module.qk_norm.weight = torch.nn.Parameter(
                module.qk_norm.weight.data.to(torch.bfloat16)
            )
        

    def _run_stream_update_kv(
        self, query, key, value, weight_q, weight_k,
        scale_k, scale_v, cos, sin, rotary_dim,
        key_cache, value_cache, block_table, cu_q_lens, context_kv_lens, eps,
    ):
        head_size = self.head_dim
        if cu_q_lens is None:
            batch_size = context_kv_lens.shape[0]
            cu_q_lens = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query.device
            )
        output_q, key_quant, value_quant = (
            ixf_f.rms_norm_qk_rotary_embedding_quant_kv_and_store(
                cos, sin,
                query, key, value,
                weight_q, weight_k,
                head_size,
                scale_k, scale_v,
                key_cache, value_cache,
                block_table, cu_q_lens, context_kv_lens,
                eps=eps, is_neox_style=True, rotary_dim=rotary_dim,
            )
        )
        return output_q, key_quant, value_quant

    def _run_sage_stream_update_kv(
        self, query, key, value, weight_q, weight_k,
        scale_k, scale_v, cos, sin, rotary_dim,
        key_cache, value_cache, block_table, cu_q_lens, context_kv_lens, eps,
        key_pt_cache, key_pt_scale_cache,
    ):
        """Full + SAGE: same all-in-one fused norm + rope + static int8 K/V quant
        + paged store as ``_run_stream_update_kv``, additionally producing the
        per-token dynamic-int8 key and its per-token scale and storing both into
        their own paged "Bypass" caches (``key_pt_cache`` / ``key_pt_scale_cache``),
        exactly like ``key_cache`` / ``value_cache`` — all in a single fused kernel.

        Returns ``(output_q, key_quant, value_quant, key_pt_int8, key_pt_scale)``:
          - ``output_q``    : [T, nh_q, head_dim] post-norm+rope query (bf16)
          - ``key_quant``   : [T, nkv, head_dim] static per-channel int8 key
          - ``value_quant`` : [T, nkv, head_dim] static per-channel int8 value
          - ``key_pt_int8`` : [T, nkv, head_dim] per-token int8 key
          - ``key_pt_scale``: [T, nkv, 1] per-token float32 scale
        """
        head_size = self.head_dim
        if cu_q_lens is None:
            batch_size = context_kv_lens.shape[0]
            cu_q_lens = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query.device
            )
        output_q, key_quant, value_quant, key_pt_int8, key_pt_scale = (
            ixf_f.rms_norm_sage_qk_rotary_embedding(
                cos, sin,
                query, key, value,
                weight_q, weight_k,
                head_size,
                scale_k, scale_v,
                key_cache, value_cache,
                key_pt_cache, key_pt_scale_cache,
                block_table, cu_q_lens, context_kv_lens,
                eps=eps, is_neox_style=True, rotary_dim=rotary_dim,
                quant_dtype=torch.int8,
            )
        )
        return output_q, key_quant, value_quant, key_pt_int8, key_pt_scale

    def _run_stream_no_update(
        self, query, key, weight_q, weight_k,
        cos, sin, rotary_dim, eps,
    ):
        head_size = self.head_dim
        output_q, _ = ixf_f.rms_norm_qk_rotary_embedding(
            cos, sin,
            query, key,
            weight_q, weight_k,
            head_size,
            eps=eps, is_neox_style=True, rotary_dim=rotary_dim,
        )
        return output_q

    def forward(
        self,
        swa_query, swa_key, swa_value,
        full_query, full_key, full_value,
        cos, sin,
        full_key_cache, full_value_cache,
        swa_key_cache, swa_value_cache,
        block_tables, cu_q_lens, context_kv_lens,
        block_tables_sparse, cu_q_lens_sparse, context_kv_lens_sparse,
        sage_full_k_pt_cache=None,
        sage_full_k_pt_scale_cache=None,
        update_kv=True,
    ):

        eps = self.qk_norm.variance_epsilon if self.qk_norm is not None else 1e-5
        rotary_dim = cos.shape[-1]

        w = self.qk_norm.weight
        swa_wq = w[0]
        swa_wk = w[1]
        full_wq = w[2]
        full_wk = w[3]

        swa_ks = self.swa_k_quantize.scale
        swa_vs = self.swa_v_quantize.scale
        full_ks = self.full_k_quantize.scale
        full_vs = self.full_v_quantize.scale

        # --- YOCO reuse layer: Q norm+rope only, no K/V touch ---
        if not update_kv:
            swa_q_out = self._run_stream_no_update(
                swa_query, swa_key, swa_wq, swa_wk,
                cos, sin, rotary_dim, eps,
            )
            full_q_out = self._run_stream_no_update(
                full_query, full_key, full_wq, full_wk,
                cos, sin, rotary_dim, eps,
            )
            return (
                swa_q_out, full_q_out,
                None, None, None, None, None, None, None, None,
                None, None,
            )

        # --- SWA stream: all-in-one fused norm+rope+static-quant+store ---
        swa_q_out, swa_key_q, swa_val_q = self._run_stream_update_kv(
            swa_query, swa_key, swa_value, swa_wq, swa_wk,
            swa_ks, swa_vs,
            cos, sin, rotary_dim,
            swa_key_cache, swa_value_cache,
            block_tables_sparse, cu_q_lens_sparse, context_kv_lens_sparse, eps,
        )

        # --- Full stream: all-in-one fused kernel (+ per-token int8 K for SAGE) ---
        if self.enable_sage:
            (full_q_out, full_key_q, full_val_q,
             full_key_pt_int8, full_key_pt_scale) = self._run_sage_stream_update_kv(
                full_query, full_key, full_value, full_wq, full_wk,
                full_ks, full_vs,
                cos, sin, rotary_dim,
                full_key_cache, full_value_cache,
                block_tables, cu_q_lens, context_kv_lens, eps,
                sage_full_k_pt_cache, sage_full_k_pt_scale_cache,
            )
        else:
            full_q_out, full_key_q, full_val_q = self._run_stream_update_kv(
                full_query, full_key, full_value, full_wq, full_wk,
                full_ks, full_vs,
                cos, sin, rotary_dim,
                full_key_cache, full_value_cache,
                block_tables, cu_q_lens, context_kv_lens, eps,
            )
            full_key_pt_int8, full_key_pt_scale = None, None

        return (
            swa_q_out, full_q_out,
            full_key_q, full_ks,
            swa_key_q, swa_ks,
            full_val_q, full_vs,
            swa_val_q, swa_vs,
            full_key_pt_int8, full_key_pt_scale,
        )
