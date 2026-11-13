set(CMAKE_ASC_ARCHITECTURES "dav-3510")
find_package(ASC REQUIRED HINTS "${_cann_arch_dir}/lib64/cmake")
enable_language(ASC)

add_library(mojo_attention_dropout SHARED kernel/entry.asc)
target_include_directories(mojo_attention_dropout PRIVATE ${MOJO_ASCEND_INCLUDE_DIRS})
target_link_libraries(mojo_attention_dropout PRIVATE ascendcl)
set_target_properties(mojo_attention_dropout PROPERTIES
    LINKER_LANGUAGE ASC
    LIBRARY_OUTPUT_DIRECTORY "${MOJO_NATIVE_OUTPUT_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/lib64")
set_property(GLOBAL APPEND PROPERTY MOJO_NATIVE_LIBRARIES "$<TARGET_FILE_NAME:mojo_attention_dropout>")
mojo_native_operator(attention_dropout mojo_attention_dropout)
