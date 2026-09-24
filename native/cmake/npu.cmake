if(MOJO_NATIVE_PROVIDER STREQUAL "npu_a2")
    set(_default_soc Ascend910B1)
elseif(MOJO_NATIVE_PROVIDER STREQUAL "npu_a5")
    # Architecture-wide ASC kernels select dav-3510, with runtime core counts.
    set(_default_soc "")
elseif(MOJO_NATIVE_PROVIDER STREQUAL "npu_a5/sku_950pr")
    set(_default_soc Ascend950PR_9579)
else()
    message(FATAL_ERROR "Unsupported NPU provider: '${MOJO_NATIVE_PROVIDER}'")
endif()
set(SOC_VERSION "${_default_soc}" CACHE STRING "CANN SoC name")
if(MOJO_NATIVE_PROVIDER STREQUAL "npu_a5/sku_950pr" AND NOT SOC_VERSION MATCHES "^Ascend950PR(_|$)")
    message(FATAL_ERROR "sku_950pr requires an Ascend950PR SoC, got '${SOC_VERSION}'")
endif()
set(CATLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/catlass" CACHE PATH "External CATLASS checkout")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit")
if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('torch_npu').origin).parent)"
    RESULT_VARIABLE _query_status
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    ERROR_VARIABLE _query_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _query_status EQUAL 0)
    message(FATAL_ERROR "Cannot locate torch_npu with ${Python3_EXECUTABLE}: ${_query_error}")
endif()
set(TORCH_NPU_LIB_DIR "${TORCH_NPU_ROOT}/lib")
if(NOT MOJO_NATIVE_PROVIDER STREQUAL "npu_a5")
    include("${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
endif()
set(_cann_arch_dir "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")
set(MOJO_ASCEND_INCLUDE_DIRS
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${_cann_arch_dir}/include"
    "${_cann_arch_dir}/include/ascendc"
    "${_cann_arch_dir}/include/ascendc/basic_api"
    "${_cann_arch_dir}/asc"
    "${_cann_arch_dir}/asc/include"
    "${_cann_arch_dir}/asc/include/basic_api"
    "${_cann_arch_dir}/asc/include/interface"
    "${_cann_arch_dir}/ascendc/include"
    "${_cann_arch_dir}/ascendc/include/highlevel_api"
    "${_cann_arch_dir}/ascendc/include/basic_api"
    "${_cann_arch_dir}/ascendc/include/highlevel_api/impl"
    "${_cann_arch_dir}/ascendc/include/basic_api/impl"
    "${_cann_arch_dir}/ascendc/include/basic_api/interface"
)
set(_npu_link_dirs
    "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "${_cann_arch_dir}/lib64"
    "${_cann_arch_dir}/devlib"
    "${ASCEND_CANN_PACKAGE_PATH}/fwkacllib/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/driver/lib64"
    "/usr/local/Ascend/driver/lib64"
    "/usr/local/Ascend/driver/lib64/driver"
    "${TORCH_LIB_DIR}"
    "${TORCH_NPU_LIB_DIR}"
)
# Also inherited by CANN's generated host-stub projects.
link_directories(${_npu_link_dirs})
target_link_directories(_native PRIVATE ${_npu_link_dirs})
target_include_directories(_native PRIVATE ${MOJO_ASCEND_INCLUDE_DIRS} "${TORCH_NPU_ROOT}/include")
target_link_libraries(_native PRIVATE torch_npu ascendcl dl)
set_property(TARGET _native APPEND PROPERTY INSTALL_RPATH
    "${TORCH_NPU_LIB_DIR}" "${ASCEND_CANN_PACKAGE_PATH}/lib64")

function(mojo_native_kernel target source)
    cmake_parse_arguments(KERNEL "CATLASS" "" "DEFINITIONS" ${ARGN})
    set(_includes "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/kernel"
        "${CMAKE_CURRENT_BINARY_DIR}" ${MOJO_ASCEND_INCLUDE_DIRS})
    if(KERNEL_CATLASS)
        if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
            message(FATAL_ERROR "CATLASS headers missing. Run git submodule update --init native/third_party/catlass or set CATLASS_ROOT.")
        endif()
        list(APPEND _includes "${CATLASS_ROOT}/include")
        configure_file("${CATLASS_ROOT}/LICENSE" "${MOJO_NATIVE_OUTPUT_DIR}/CATLASS_LICENSE.txt" COPYONLY)
        set_property(GLOBAL APPEND PROPERTY MOJO_NATIVE_NOTICES "CATLASS_LICENSE.txt")
    endif()
    ascendc_library(${target} SHARED "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    set_property(GLOBAL APPEND PROPERTY MOJO_NATIVE_LIBRARIES "$<TARGET_FILE_NAME:${target}>")
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${MOJO_NATIVE_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${MOJO_NATIVE_OUTPUT_DIR}"
    )
    target_include_directories(${target} PRIVATE ${_includes})
    ascendc_include_directories(${target} PRIVATE ${_includes})
    target_compile_definitions(${target} PRIVATE ${KERNEL_DEFINITIONS})
    ascendc_compile_definitions(${target} PRIVATE ${KERNEL_DEFINITIONS})
    target_include_directories(_native PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/${target}")
endfunction()
