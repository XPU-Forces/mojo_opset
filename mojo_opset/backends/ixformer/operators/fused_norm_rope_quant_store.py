from typing import Optional, Tuple

import torch

from ixformer import functions as ixf_f
from mojo_opset.experimental import MojoFusedNormRoPEQuantStore


class IxformerFusedNormRoPEQuantStore(MojoFusedNormRoPEQuantStore):
    """Ixformer backend for fused QK-Norm + RoPE + KV-StaticQuant + PagedKVStore.

    Calls ``rms_norm_qk_rotary_embedding_quant_kv_store`` (for update_kv=True)
    or ``rms_norm_qk_rotary_embedding`` (for update_kv=False) twice — once per
    attention stream (SWA and full).

    The ixformer kernel accepts cos/sin tensors of shape ``(T, rope_dim)``
    directly (no positions / cos_sin_cache indirection).
    """

    supported_platforms_list = ["ilu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.head_dim != 128:
            raise NotImplementedError(
                f"ixformer fused kernel only supports head_dim=128, got {self.head_dim}"
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
        output_q, key_quant, key_scale, value_quant, value_scale = (
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
        return output_q, key_quant, key_scale, value_quant, value_scale

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
            )

        swa_q_out, swa_key_q, swa_k_scale, swa_val_q, swa_v_scale = self._run_stream_update_kv(
            swa_query, swa_key, swa_value, swa_wq, swa_wk,
            swa_ks, swa_vs,
            cos, sin, rotary_dim,
            swa_key_cache, swa_value_cache,
            block_tables_sparse, cu_q_lens_sparse, context_kv_lens_sparse, eps,
        )

        full_q_out, full_key_q, full_k_scale, full_val_q, full_v_scale = self._run_stream_update_kv(
            full_query, full_key, full_value, full_wq, full_wk,
            full_ks, full_vs,
            cos, sin, rotary_dim,
            full_key_cache, full_value_cache,
            block_tables, cu_q_lens, context_kv_lens, eps,
        )

        return (
            swa_q_out, full_q_out,
            full_key_q, full_k_scale,
            swa_key_q, swa_k_scale,
            full_val_q, full_v_scale,
            swa_val_q, swa_v_scale,
        )
