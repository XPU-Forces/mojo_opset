include("${CMAKE_CURRENT_LIST_DIR}/../common/kernel_types.cmake")
foreach(dtype IN ITEMS bf16 fp16)
    set(_target "mojo_a2_group_wgrad_${dtype}")
    mojo_native_kernel(${_target} kernel.cpp DEFINITIONS
        DTYPE_INPUT=${_kernel_type_${dtype}} MOJO_GROUP_WEIGHT_NZ=0 MOJO_GROUP_TRANS_A=true
        MOJO_GROUP_TRANS_B=false MOJO_GROUP_KERNEL=${_target})
    list(APPEND _kernels ${_target})
endforeach()
mojo_native_operator(group_gemm_bwd ${_kernels})
target_link_libraries(_native PRIVATE tiling_api platform)
