set(CMAKE_ASC_ARCHITECTURES "dav-3510")
find_package(ASC REQUIRED HINTS "${_cann_arch_dir}/lib64/cmake")
enable_language(ASC)

add_library(mojo_topk SHARED kernel/deepselect_bf16.asc kernel/topk_rescan_fp32.asc)
target_include_directories(mojo_topk PRIVATE ${MOJO_ASCEND_INCLUDE_DIRS})
target_compile_options(mojo_topk PRIVATE -O2 -std=c++20
    -U__cpp_concepts -U__cpp_impl_three_way_comparison
    -mllvm -cce-aicore-dcpreload-args=false)
target_link_libraries(mojo_topk PRIVATE ascendcl "${_cann_arch_dir}/lib64/libascendc_runtime.a")
target_link_options(mojo_topk PRIVATE -Wl,-Bsymbolic-functions)
set_target_properties(mojo_topk PROPERTIES
    LINKER_LANGUAGE ASC
    LIBRARY_OUTPUT_DIRECTORY "${MOJO_NATIVE_OUTPUT_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/lib64")
set_property(GLOBAL APPEND PROPERTY MOJO_NATIVE_LIBRARIES "$<TARGET_FILE_NAME:mojo_topk>")

mojo_native_operator(topk mojo_topk)
