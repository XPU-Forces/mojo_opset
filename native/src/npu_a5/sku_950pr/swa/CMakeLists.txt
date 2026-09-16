# Keep specialization bounds identical to the checked C++ launch contract.
function(add_swa_kernel suffix q_heads kv_heads window interleave)
    mojo_native_kernel(swa_v5_mojo_${suffix}_bf16_kernel kernel/entry.cpp DEFINITIONS
        SWA_V5_ENTRY_KERNEL=swa_v5_mojo_${suffix}_bf16
        SWA_V5_ENTRY_NUM_Q_HEADS=${q_heads}
        SWA_V5_ENTRY_NUM_KV_HEADS=${kv_heads}
        SWA_V5_ENTRY_MAX_Q_LEN=1024
        SWA_V5_ENTRY_MAX_KV_LEN=9216
        SWA_V5_ENTRY_DIM=128
        SWA_V5_ENTRY_WINDOW_LEFT=${window}
        SWA_V5_ENTRY_WINDOW_RIGHT=0
        SWA_V5_ENTRY_GLOBAL_WINDOW_SIZE=4
        SWA_V5_ENTRY_IS_FORWARD=0
        SWA_V5_ENTRY_IS_CAUSAL=1
        SWA_V5_ENTRY_GQA_LAYOUT_ABAB=${interleave}
        SWA_V5_ENTRY_OUTPUT_FP16=0
        SWA_V5_ENTRY_INPUT_BF16=1
        SWA_V5_ENTRY_OUTPUT_BF16=1
        SWA_V5_ENTRY_NUM_AIC_CORES=28
    )
endfunction()

add_swa_kernel(h16_kv4_aabb 16 4 1023 0)
add_swa_kernel(h16_kv4_abab 16 4 255 1)
add_swa_kernel(h8_kv1_aabb 8 1 1023 0)
add_swa_kernel(h8_kv1_abab 8 1 255 1)
mojo_native_operator(swa
    swa_v5_mojo_h16_kv4_aabb_bf16_kernel
    swa_v5_mojo_h16_kv4_abab_bf16_kernel
    swa_v5_mojo_h8_kv1_aabb_bf16_kernel
    swa_v5_mojo_h8_kv1_abab_bf16_kernel
)
