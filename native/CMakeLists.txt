cmake_minimum_required(VERSION 3.18)
project(mojo_opset_native LANGUAGES CXX)

set(MOJO_NATIVE_PROVIDER "" CACHE STRING "Native provider directory, for example npu_a2")
set(MOJO_NATIVE_OP "all" CACHE STRING "Operator to build, or all")
if(NOT MOJO_NATIVE_PROVIDER MATCHES "^([a-z0-9]+)(_[a-z0-9_]+)?(/sku_[a-z0-9_]+)?$")
    message(FATAL_ERROR "Invalid native provider: '${MOJO_NATIVE_PROVIDER}'")
endif()
set(_platform "${CMAKE_MATCH_1}")
set(_provider_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/${MOJO_NATIVE_PROVIDER}")
set(_platform_cmake "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${_platform}.cmake")
if(NOT EXISTS "${_provider_dir}/bindings.cpp" OR NOT EXISTS "${_platform_cmake}")
    message(FATAL_ERROR "Unsupported native provider: '${MOJO_NATIVE_PROVIDER}'")
endif()
set(MOJO_NATIVE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mojo_opset_lib/${MOJO_NATIVE_PROVIDER}"
    CACHE PATH "Runtime artifact directory")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
get_filename_component(MOJO_NATIVE_OUTPUT_DIR "${MOJO_NATIVE_OUTPUT_DIR}" ABSOLUTE)
file(MAKE_DIRECTORY "${MOJO_NATIVE_OUTPUT_DIR}")
# An interrupted rebuild must not retain an earlier successful-build receipt.
file(REMOVE "${MOJO_NATIVE_OUTPUT_DIR}/_build_info.json")

# Discover recipes, not arbitrary C++ sources (which may contain multiple entries).
if(MOJO_NATIVE_OP STREQUAL "all")
    file(GLOB _recipes CONFIGURE_DEPENDS "${_provider_dir}/*/CMakeLists.txt")
elseif(MOJO_NATIVE_OP MATCHES "^[a-z0-9_]+$" AND EXISTS "${_provider_dir}/${MOJO_NATIVE_OP}/CMakeLists.txt")
    set(_recipes "${_provider_dir}/${MOJO_NATIVE_OP}/CMakeLists.txt")
else()
    message(FATAL_ERROR "Unsupported operator '${MOJO_NATIVE_OP}' for ${MOJO_NATIVE_PROVIDER}")
endif()
if(NOT _recipes)
    message(FATAL_ERROR "No native operators for ${MOJO_NATIVE_PROVIDER}")
endif()

# Locate Torch without importing its runtime (imports can print to stdout).
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec('torch').origin).parent)"
    RESULT_VARIABLE _query_status
    OUTPUT_VARIABLE TORCH_ROOT
    ERROR_VARIABLE _query_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _query_status EQUAL 0)
    message(FATAL_ERROR "Cannot locate torch with ${Python3_EXECUTABLE}: ${_query_error}")
endif()
set(Torch_DIR "${TORCH_ROOT}/share/cmake/Torch")
find_package(Torch CONFIG REQUIRED)
set(TORCH_LIB_DIR "${TORCH_ROOT}/lib")
separate_arguments(_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

add_custom_target(mojo_native)

# One Python module per selected architecture/SKU. Recipes add host/kernel sources.
Python3_add_library(_native MODULE WITH_SOABI "${_provider_dir}/bindings.cpp")
set_target_properties(_native PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${MOJO_NATIVE_OUTPUT_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${TORCH_LIB_DIR}"
)
target_include_directories(_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${_provider_dir}" ${TORCH_INCLUDE_DIRS}
)
target_compile_definitions(_native PRIVATE TORCH_EXTENSION_NAME=_native)
target_compile_options(_native PRIVATE -O2 ${_torch_flags} -Wall)
target_link_directories(_native PRIVATE "${TORCH_LIB_DIR}")
target_link_libraries(_native PRIVATE ${TORCH_LIBRARIES} torch_python)
add_dependencies(mojo_native _native)

# Platform files configure their toolchain/dependencies and kernel build helper.
include("${_platform_cmake}")

function(mojo_native_operator name)
    string(TOUPPER "${name}" _operator_define)
    target_compile_definitions(_native PRIVATE "MOJO_NATIVE_WITH_${_operator_define}=1")
    target_sources(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/host.cpp")
    target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/kernel")
    target_link_libraries(_native PRIVATE ${ARGN})
endfunction()

foreach(recipe IN LISTS _recipes)
    get_filename_component(_operator_dir "${recipe}" DIRECTORY)
    get_filename_component(_operator "${_operator_dir}" NAME)
    add_subdirectory("src/${MOJO_NATIVE_PROVIDER}/${_operator}")
endforeach()

# Preserve the extension's configured runtime paths after platform recipes run.
set(CMAKE_SKIP_RPATH FALSE)

# Publish a receipt only after the selected build succeeds. Wheel packaging
# checks it without importing Torch or executing the extension.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_config_var('SOABI')); print(sysconfig.get_platform())"
    OUTPUT_VARIABLE _python_build_info
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _build_info_status
)
if(NOT _build_info_status EQUAL 0)
    message(FATAL_ERROR "Cannot query build interpreter metadata")
endif()
string(REPLACE "\n" ";" _python_build_info "${_python_build_info}")
list(GET _python_build_info 0 _python_soabi)
list(GET _python_build_info 1 _python_platform)
get_property(_kernel_libraries GLOBAL PROPERTY MOJO_NATIVE_LIBRARIES)
set(_libraries "$<TARGET_FILE_NAME:_native>" ${_kernel_libraries})
string(JOIN "\", \"" _libraries_json ${_libraries})
get_property(_notices GLOBAL PROPERTY MOJO_NATIVE_NOTICES)
set(_notices_json "")
if(_notices)
    list(REMOVE_DUPLICATES _notices)
    string(JOIN "\", \"" _notices_json ${_notices})
    set(_notices_json "\"${_notices_json}\"")
endif()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json" CONTENT
    "{\"provider\": \"${MOJO_NATIVE_PROVIDER}\", \"operator\": \"${MOJO_NATIVE_OP}\", \"python_soabi\": \"${_python_soabi}\", \"platform\": \"${_python_platform}\", \"libraries\": [\"${_libraries_json}\"], \"notices\": [${_notices_json}]}\n")
add_custom_command(TARGET mojo_native POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json" "${MOJO_NATIVE_OUTPUT_DIR}/_build_info.json")
