set(CMAKE_ASC_ARCHITECTURES "dav-3510")
find_package(ASC REQUIRED HINTS "${_cann_arch_dir}/lib64/cmake")
enable_language(ASC)

set(_swa_source_dir "${CMAKE_CURRENT_LIST_DIR}")
add_library(mojo_swa SHARED "${_swa_source_dir}/host.asc")
target_include_directories(mojo_swa PRIVATE
    "${_swa_source_dir}/../.." "${_swa_source_dir}"
    ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_ROOT}/include" ${MOJO_ASCEND_INCLUDE_DIRS}
)
target_compile_options(mojo_swa PRIVATE ${_torch_flags})
target_link_libraries(mojo_swa PRIVATE ${TORCH_LIBRARIES} torch_npu ascendcl)
set_target_properties(mojo_swa PROPERTIES
    LINKER_LANGUAGE ASC
    LIBRARY_OUTPUT_DIRECTORY "${MOJO_NATIVE_OUTPUT_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${TORCH_LIB_DIR};${TORCH_NPU_LIB_DIR};${ASCEND_CANN_PACKAGE_PATH}/lib64"
)
set_property(GLOBAL APPEND PROPERTY MOJO_NATIVE_LIBRARIES "$<TARGET_FILE_NAME:mojo_swa>")
target_compile_definitions(_native PRIVATE MOJO_NATIVE_WITH_SWA=1)
target_link_libraries(_native PRIVATE mojo_swa)
