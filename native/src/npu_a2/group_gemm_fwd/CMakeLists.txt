include("${CMAKE_CURRENT_LIST_DIR}/../common/kernel_types.cmake")
foreach(dtype IN ITEMS bf16 fp16)
    set(_target "mojo_a2_group_forward_${dtype}")
    mojo_native_kernel(${_target} kernel.cpp DEFINITIONS
        DTYPE_INPUT=${_kernel_type_${dtype}} MOJO_GROUP_WEIGHT_NZ=0 MOJO_GROUP_TRANS_A=false
        MOJO_GROUP_TRANS_B=true MOJO_GROUP_KERNEL=${_target})
    list(APPEND _kernels ${_target})
    set(_target "mojo_a2_group_dgrad_${dtype}")
    mojo_native_kernel(${_target} kernel.cpp DEFINITIONS
        DTYPE_INPUT=${_kernel_type_${dtype}} MOJO_GROUP_WEIGHT_NZ=0 MOJO_GROUP_TRANS_A=false
        MOJO_GROUP_TRANS_B=false MOJO_GROUP_KERNEL=${_target})
    list(APPEND _kernels ${_target})
endforeach()
mojo_native_operator(group_gemm_fwd ${_kernels})
target_link_libraries(_native PRIVATE tiling_api platform)
