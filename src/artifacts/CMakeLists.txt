# SPDX-License-Identifier: MIT
# Reuse the pinned upstream header-only JSON and public-domain SHA-256 code.
set(_ffn_vendor "${PROJECT_SOURCE_DIR}/repos/llama.cpp/vendor")
if(NOT EXISTS "${_ffn_vendor}/nlohmann/json.hpp")
    message(FATAL_ERROR "Initialize the locked llama.cpp checkout for the artifact JSON dependency.")
endif()
file(READ "${PROJECT_SOURCE_DIR}/dependencies.lock.json" FFN_LOCK_JSON)
file(SHA256 "${PROJECT_SOURCE_DIR}/dependencies.lock.json" FFN_LOCK_SHA256)
set(FFN_SOURCE_HASHES "{}")
foreach(relative IN ITEMS
    crt/coralnpu_dispatch.h crt/coralnpu_main.c crt/coralnpu_dma.c
    compiler/Target/CoralNPULinkerTool.cpp
    runtime/sim/simulator_backend.c)
    file(SHA256 "${CORALNPU_SOURCE_DIR}/${relative}" digest)
    string(JSON FFN_SOURCE_HASHES SET "${FFN_SOURCE_HASHES}" "${relative}" "\"${digest}\"")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CORALNPU_SOURCE_DIR}/${relative}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/dependencies.lock.json")
configure_file(build_identity.h.in "${CMAKE_CURRENT_BINARY_DIR}/build_identity.h" @ONLY)
add_library(coral_ffn_artifact STATIC
    ffn_artifact.cc "${_ffn_vendor}/hash/sha256/sha256.c")
target_include_directories(coral_ffn_artifact PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_include_directories(coral_ffn_artifact SYSTEM PRIVATE "${_ffn_vendor}" "${_ffn_vendor}/hash")
target_include_directories(coral_ffn_artifact PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(coral_ffn_artifact PRIVATE -Wall -Wextra -fexceptions)
add_executable(coral-ffn-artifact-check ffn_artifact_check.cc)
target_link_libraries(coral-ffn-artifact-check PRIVATE coral_ffn_artifact)
target_compile_options(coral-ffn-artifact-check PRIVATE -Wall -Wextra -fexceptions)
set_target_properties(coral-ffn-artifact-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
