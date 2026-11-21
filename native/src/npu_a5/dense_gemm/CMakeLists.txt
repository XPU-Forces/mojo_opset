set(CMAKE_ASC_ARCHITECTURES "dav-3510")
find_package(ASC REQUIRED HINTS "${_cann_arch_dir}/lib64/cmake")
enable_language(ASC)

# Both the host table and device instances come from this build-time catalogue.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate.py" "${CMAKE_CURRENT_SOURCE_DIR}/tuning.json" "${CMAKE_CURRENT_SOURCE_DIR}/specializations.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/bf16_gemm.h" "${CMAKE_CURRENT_SOURCE_DIR}/fmt/format.h")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"
    --output "${CMAKE_CURRENT_BINARY_DIR}/generated" COMMAND_ERROR_IS_FATAL ANY)
include("${CMAKE_CURRENT_BINARY_DIR}/generated/sources.cmake")
add_library(mojo_dense_gemm SHARED ${GEMM_AOT_SOURCES})
target_include_directories(mojo_dense_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${MOJO_ASCEND_INCLUDE_DIRS})
target_compile_options(mojo_dense_gemm PRIVATE -O2 -std=c++20 --cce-disable-vf-stack-reserved-ubuf
    "SHELL:-mllvm -enable-hiipu-vf-loop-unroll" "SHELL:-mllvm -cce-aicore-dcpreload-args=false" ${_torch_flags})
target_link_libraries(mojo_dense_gemm PRIVATE ascendcl)
set_target_properties(mojo_dense_gemm PROPERTIES LINKER_LANGUAGE ASC
    LIBRARY_OUTPUT_DIRECTORY "${MOJO_NATIVE_OUTPUT_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/lib64")
set_property(GLOBAL APPEND PROPERTY MOJO_NATIVE_LIBRARIES "$<TARGET_FILE_NAME:mojo_dense_gemm>")

mojo_native_operator(dense_gemm mojo_dense_gemm)
# Pin the host-only formatter so clean build images need no system fmt package.
target_compile_definitions(_native PRIVATE FMT_HEADER_ONLY)
target_sources(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/aot_dispatch.cpp")
target_compile_features(_native PRIVATE cxx_std_20)
target_compile_options(_native PRIVATE -fvisibility=hidden)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
foreach(notice DEEPGEMM FMT)
    configure_file("${CMAKE_CURRENT_BINARY_DIR}/generated/${notice}_LICENSE.txt" "${MOJO_NATIVE_OUTPUT_DIR}/${notice}_LICENSE.txt" COPYONLY)
    set_property(GLOBAL APPEND PROPERTY MOJO_NATIVE_NOTICES "${notice}_LICENSE.txt")
endforeach()

# Remove the obsolete notice when upgrading an existing build output directory.
file(REMOVE "${MOJO_NATIVE_OUTPUT_DIR}/MAGIC_ENUM_LICENSE.txt")
