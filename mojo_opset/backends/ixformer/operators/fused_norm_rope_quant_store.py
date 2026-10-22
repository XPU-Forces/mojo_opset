from typing import Optional, Tuple

import torch

from ixformer import functions as ixf_f
from mojo_opset.core.operators.quantize import MojoStaticQuant
from mojo_opset.experimental import MojoFusedNormRoPEFp8KVStore
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

        swa_q_out, swa_key_q, swa_val_q = self._run_stream_update_kv(
            swa_query, swa_key, swa_value, swa_wq, swa_wk,
            swa_ks, swa_vs,
            cos, sin, rotary_dim,
            swa_key_cache, swa_value_cache,
            block_tables_sparse, cu_q_lens_sparse, context_kv_lens_sparse, eps,
        )

        full_q_out, full_key_q, full_val_q = self._run_stream_update_kv(
            full_query, full_key, full_value, full_wq, full_wk,
            full_ks, full_vs,
            cos, sin, rotary_dim,
            full_key_cache, full_value_cache,
            block_tables, cu_q_lens, context_kv_lens, eps,
        )

        return (
            swa_q_out, full_q_out,
            full_key_q, full_ks,
            swa_key_q, swa_ks,
            full_val_q, full_vs,
            swa_val_q, swa_vs,
        )


class IxformerFusedNormRoPEFp8KVStore(MojoFusedNormRoPEFp8KVStore):
    """Ixformer fp8 backend for fused QK-Norm + RoPE + KV-StaticQuant + PagedKVStore.

    Calls ``rms_norm_qk_rotary_embedding_fp8_kv_store`` (for update_kv=True) or
    ``rms_norm_qk_rotary_embedding`` (for update_kv=False) twice — once per
    attention stream (SWA and full).

    Two things separate this from the int8 backend. The kernel writes the
    quantized K/V straight into the paged cache without handing them back, so
    only the Q outputs are returned. And it reads the scales per (batch, head)
    rather than per-channel, which is the layout ixattn's fp8 paged FMHA takes
    too, so the scales pass from here straight into the attention call. That
    batch axis is a real dimension of the parameter, so the operator is built
    for one ``batch_size`` and the calls have to match it.
    """

    supported_platforms_list = ["ilu"]

    def __init__(self, *args, batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        if self.head_dim != 128:
            raise NotImplementedError(
                f"ixformer fused kernel only supports head_dim=128, got {self.head_dim}"
            )
        if not (self.use_query_norm and self.use_key_norm):
            raise NotImplementedError(
                f"ixformer fused kernel only supports use_query_norm and use_key_norm, got {self.use_query_norm} and {self.use_key_norm}"
            )
        if not hasattr(ixf_f, "rms_norm_qk_rotary_embedding_fp8_kv_store"):
            raise NotImplementedError(
                "the installed ixformer has no rms_norm_qk_rotary_embedding_fp8_kv_store"
            )

        # The base builds per-channel (head_num_k, head_dim) scales for the int8
        # cache. Rebuild them in the (batch_size, head_num_k) layout the fp8
        # kernel reads, so forward hands the parameters over untouched.
        self.batch_size = batch_size
        quant_cls = MojoStaticQuant._registry.get(self._backend)
        self.full_k_quantize = quant_cls((batch_size, self.num_heads_full_k), quant_dtype=torch.float8_e4m3fn)
        self.full_v_quantize = quant_cls((batch_size, self.num_heads_full_k), quant_dtype=torch.float8_e4m3fn)
        self.swa_k_quantize = quant_cls((batch_size, self.num_heads_swa_k), quant_dtype=torch.float8_e4m3fn)
        self.swa_v_quantize = quant_cls((batch_size, self.num_heads_swa_k), quant_dtype=torch.float8_e4m3fn)

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
        output_q = ixf_f.rms_norm_qk_rotary_embedding_fp8_kv_store(
            cos, sin,
            query, key, value,
            weight_q, weight_k,
            head_size,
            scale_k, scale_v,
            key_cache, value_cache,
            block_table, cu_q_lens, context_kv_lens,
            eps=eps, is_neox_style=True, rotary_dim=rotary_dim,
        )
        return output_q

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

        swa_ks = self.swa_k_quantize.scale
        swa_vs = self.swa_v_quantize.scale
        full_ks = self.full_k_quantize.scale
        full_vs = self.full_v_quantize.scale

        swa_q_out = self._run_stream_update_kv(
            swa_query, swa_key, swa_value, swa_wq, swa_wk,
            swa_ks, swa_vs,
            cos, sin, rotary_dim,
            swa_key_cache, swa_value_cache,
            block_tables_sparse, cu_q_lens_sparse, context_kv_lens_sparse, eps,
        )

        full_q_out = self._run_stream_update_kv(
            full_query, full_key, full_value, full_wq, full_wk,
            full_ks, full_vs,
            cos, sin, rotary_dim,
            full_key_cache, full_value_cache,
            block_tables, cu_q_lens, context_kv_lens, eps,
        )

        # The kernel keeps the quantized K/V in the paged cache only.
        return (
            swa_q_out, full_q_out,
            None, full_ks,
            None, swa_ks,
            None, full_vs,
            None, swa_vs,
        )
