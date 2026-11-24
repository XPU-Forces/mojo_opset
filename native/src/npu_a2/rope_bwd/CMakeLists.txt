include("${CMAKE_CURRENT_LIST_DIR}/../common/kernel_types.cmake")
foreach(dtype IN ITEMS bf16 fp16 fp32)
    set(_target "mojo_a2_rope_backward_${dtype}")
    mojo_native_kernel(${_target} kernel.cpp DEFINITIONS
        DTYPE_INPUT=${_kernel_type_${dtype}} DTYPE_SIN_EMBEDS=${_kernel_type_${dtype}} MOJO_ROPE_KERNEL=${_target})
    list(APPEND _kernels ${_target})
    if(NOT dtype STREQUAL "fp32")
        set(_target "mojo_a2_rope_backward_${dtype}_fp32_table")
        mojo_native_kernel(${_target} kernel.cpp DEFINITIONS
            DTYPE_INPUT=${_kernel_type_${dtype}} DTYPE_SIN_EMBEDS=float MOJO_ROPE_KERNEL=${_target})
        list(APPEND _kernels ${_target})
    endif()
endforeach()
mojo_native_operator(rope_bwd ${_kernels})
