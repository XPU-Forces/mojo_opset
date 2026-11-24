include("${CMAKE_CURRENT_LIST_DIR}/../common/kernel_types.cmake")
foreach(dtype IN ITEMS bf16 fp16 fp32)
    set(_target "mojo_a2_swiglu_forward_${dtype}")
    mojo_native_kernel(${_target} kernel.cpp DEFINITIONS
        DTYPE_INPUT=${_kernel_type_${dtype}} MOJO_SWIGLU_KERNEL=${_target})
    list(APPEND _kernels ${_target})
endforeach()
mojo_native_operator(swiglu_fwd ${_kernels})
