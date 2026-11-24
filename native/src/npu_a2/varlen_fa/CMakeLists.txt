# Original launch variants: BF16/FP16 x causal/full. Numerical kernels are unchanged.
foreach(dtype IN ITEMS bf16 fp16)
    if(dtype STREQUAL "bf16")
        set(_type bfloat16_t)
        set(_suffix "")
    else()
        set(_type half)
        set(_suffix "_fp16")
    endif()
    foreach(mode IN ITEMS causal full)
        if(mode STREQUAL "causal")
            set(_key 2)
            set(_mode_suffix "")
        else()
            set(_key 9)
            set(_mode_suffix "_full")
        endif()
        set(_target "varlen_fa_standalone${_mode_suffix}${_suffix}")
        mojo_native_kernel(${_target} kernel/varlen_fa.cpp CATLASS DEFINITIONS
            TILING_KEY_VAR=${_key} DTYPE_QUERY=${_type} DTYPE_MASK=uint8_t
            CUSTOM_FA_ENABLE_SOFTMAX_FLASH_V2=0
            VARLEN_FA_KERNEL_NAME=varlen_fa${_mode_suffix}${_suffix}
        )
        list(APPEND _kernels ${_target})
    endforeach()
endforeach()
mojo_native_operator(varlen_fa ${_kernels})
