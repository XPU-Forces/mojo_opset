# Operator support

Implementations by architecture; `—` means no enabled accelerator implementation.
All functions also support `torch_reference`. See function docstrings for
supported shapes, dtypes and options.

| Function | `npu.a2` | `npu.a5` |
| --- | --- | --- |
| `apply_penalties_temperature` | triton | triton |
| `causal_conv1d` | triton | triton |
| `causal_conv1d_update_state_infer` | triton | triton |
| `dequant` | triton | triton |
| `dequant_swiglu_quant` | torch_npu | torch_npu |
| `diffusion_attention` | triton | triton |
| `dynamic_quant` | triton, torch_npu | triton, torch_npu |
| `embedding_nf4_dequant` | triton | triton |
| `flash_attention` | triton | triton |
| `flash_attention_infer` | triton | triton |
| `flex_attention` | triton | triton |
| `flex_attention_v2` | triton | triton |
| `gelu` | triton, torch_npu | triton, torch_npu |
| `group_gemm` | triton, torch_npu | triton, torch_npu |
| `group_rms_norm_infer` | triton, torch_npu | triton, torch_npu |
| `indexer` | triton components | triton components |
| `join_prob_reject_sampling` | triton | triton |
| `layer_norm_infer` | triton | triton |
| `layer_norm_quant` | torch_npu | torch_npu |
| `lightning_indexer` | triton | triton |
| `moe_dynamic_quant` | triton, torch_npu | triton, torch_npu |
| `multimodal_rope_infer` | triton | triton |
| `n_gram_decode` | triton | triton |
| `n_gram_prefill` | triton | triton |
| `native_swa_infer` | — | native |
| `over_encoding_decode` | triton | triton |
| `paged_decode_gqa_infer` | triton, torch_npu | triton, torch_npu |
| `paged_decode_swa_infer` | triton, torch_npu | triton, torch_npu |
| `paged_prefill_gqa_infer` | triton, torch_npu | triton, torch_npu |
| `paged_prefill_swa_infer` | triton, torch_npu | triton, torch_npu |
| `prefill_gqa_infer` | torch_npu | torch_npu |
| `quant_batch_gemm_reduce_sum` | — | — |
| `quant_gemm` | triton, torch_npu | triton, torch_npu |
| `reject_sampling` | triton | triton |
| `residual_add_layer_norm_infer` | triton | triton |
| `residual_add_layer_norm_quant` | torch_npu | torch_npu |
| `residual_add_rms_norm_infer` | triton, torch_npu | triton, torch_npu |
| `residual_add_rms_norm_quant` | torch_npu | torch_npu |
| `rms_norm` | triton | triton |
| `rms_norm_infer` | triton, torch_npu | triton, torch_npu |
| `rms_norm_quant` | torch_npu | torch_npu |
| `rope` | triton | triton |
| `rope_cos_sin` | triton | triton |
| `rope_infer` | triton, torch_npu | triton, torch_npu |
| `rotate_activation` | torch_reference | torch_reference |
| `sdpa_infer` | triton | triton |
| `silu` | triton, torch_npu | triton, torch_npu |
| `static_quant` | triton | triton |
| `store_lowrank` | triton | triton |
| `store_paged_kv_cache` | triton | triton |
| `swa` | triton | triton |
| `swa_infer` | triton | triton |
| `swiglu` | triton, torch_npu | triton, torch_npu |
| `top_k_sampling` | triton | triton |
| `top_p_filter` | triton | triton |
| `top_p_sampling` | triton | triton |
| `varlen_fa_infer` | native | — |
| `vision_rope_2d_infer` | triton, torch_npu | triton, torch_npu |
| `vision_rope_cos_sin_2d` | triton | triton |
