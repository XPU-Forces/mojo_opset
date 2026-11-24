include("${CMAKE_CURRENT_LIST_DIR}/../common/kernel_types.cmake")
foreach(dtype IN ITEMS bf16 fp16)
    set(_target mojo_a2_fa_bwd_${dtype})
    mojo_native_kernel(${_target} kernel.cpp DEFINITIONS
        DTYPE_INPUT=${_kernel_type_${dtype}} MOJO_FA_KERNEL=${_target})
    list(APPEND _kernels ${_target})
endforeach()
mojo_native_operator(fa_bwd ${_kernels})
target_link_libraries(_native PRIVATE tiling_api platform)
