# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.26)
project(LlamacppCoralM1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "M1 currently supports Linux x86_64, including WSL2.")
endif()

set(CORALNPU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/repos/coralnpu-compiler"
    CACHE PATH "Pinned Coral compiler checkout")
set(CORALNPU_MPACT_SIMULATOR_LIB "" CACHE FILEPATH
    "Prebuilt MPACT .so from the pinned Coral compiler Bazel dependencies")
get_filename_component(CORALNPU_SOURCE_DIR "${CORALNPU_SOURCE_DIR}" REALPATH)
if(NOT EXISTS "${CORALNPU_SOURCE_DIR}/runtime/driver/coralnpu_device.h" OR
   NOT EXISTS "${CORALNPU_SOURCE_DIR}/third_party/iree/CMakeLists.txt")
    message(FATAL_ERROR "Initialize the pinned dependencies before configuring M1.")
endif()

# The upstream plugin otherwise launches Bazel during CMake configuration.
# Require the explicit prebuilt library to make that side effect impossible.
if(NOT CORALNPU_MPACT_SIMULATOR_LIB OR
   NOT EXISTS "${CORALNPU_MPACT_SIMULATOR_LIB}" OR
   IS_DIRECTORY "${CORALNPU_MPACT_SIMULATOR_LIB}")
    message(FATAL_ERROR
        "Set CORALNPU_MPACT_SIMULATOR_LIB to an existing pinned MPACT .so. "
        "This project will not build or download the simulator during configure.")
endif()
get_filename_component(_m1_mpact_real
    "${CORALNPU_MPACT_SIMULATOR_LIB}" REALPATH)
set(CORALNPU_MPACT_SIMULATOR_LIB "${_m1_mpact_real}" CACHE FILEPATH
    "Prebuilt MPACT .so from the pinned Coral compiler Bazel dependencies" FORCE)

# Import the runtime only, without the upstream top-level compiler/CRT build.
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_CPUINFO OFF CACHE BOOL "" FORCE)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK OFF CACHE BOOL "" FORCE)
set(IREE_EXTERNAL_HAL_DRIVERS coralnpu CACHE STRING "" FORCE)
set(IREE_CMAKE_PLUGIN_PATHS "${CORALNPU_SOURCE_DIR}/build_tools/cmake"
    CACHE STRING "" FORCE)
set(CORALNPU_ENABLE_SPIKE OFF CACHE BOOL "" FORCE)
set(CORALNPU_ENABLE_VERILATOR OFF CACHE BOOL "" FORCE)
set(CORALNPU_ENABLE_FPGA OFF CACHE BOOL "" FORCE)

# Pinned upstream runtime CMake assumes these include roots are global.
# Keep all dependency sources unchanged.
include_directories(
    "${CORALNPU_SOURCE_DIR}"
    "${CORALNPU_SOURCE_DIR}/third_party/coralnpu"
)
add_subdirectory("${CORALNPU_SOURCE_DIR}/third_party/iree"
                 "${CMAKE_BINARY_DIR}/iree" EXCLUDE_FROM_ALL)

add_library(coral_m1_runtime STATIC src/runtime/matmul_runtime.cc)
target_include_directories(coral_m1_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(coral_m1_runtime PRIVATE
    iree::async
    iree::async::util::proactor_pool
    iree::base
    iree::hal
    iree::hal::drivers::local_sync::sync_driver
    iree::hal::local
    iree::modules::hal
    iree::vm
    iree::vm::bytecode::module
    coralnpu_runtime::driver::coralnpu
    coralnpu_runtime::sim::simulation
    coralnpu_runtime::sim::simulator_mpact
    coralnpu_metrics_runtime::stats_collector
)

add_executable(coral-matmul-runner src/runtime/matmul_runner.cc)
target_link_libraries(coral-matmul-runner PRIVATE coral_m1_runtime)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(coral_m1_runtime PRIVATE -Wall -Wextra -fexceptions)
    target_compile_options(coral-matmul-runner PRIVATE -Wall -Wextra -fexceptions)
endif()

# Register tests only when the caller supplies already compiled VMFBs.
# Configuration never exports, compiles or executes a test artifact.
include(CTest)
if(BUILD_TESTING)
    foreach(m IN ITEMS 1 16 32)
        set(M1_VMFB_M${m} "" CACHE FILEPATH
            "M1 strict F32 VMFB for M=${m}, K=16, N=8")
        if(M1_VMFB_M${m})
            get_filename_component(artifact "${M1_VMFB_M${m}}" ABSOLUTE)
            if(NOT EXISTS "${artifact}")
                message(FATAL_ERROR "M1_VMFB_M${m} does not exist: ${artifact}")
            endif()
            add_test(NAME m1_matmul_m${m}
                COMMAND coral-matmul-runner
                    --module "${artifact}" --m ${m} --k 16 --n 8
                    --repeat 3 --recreate 2 --self-test)
            set_tests_properties(m1_matmul_m${m} PROPERTIES TIMEOUT 180)
        endif()
    endforeach()
endif()

option(CORAL_BUILD_M2 "Build the M2 GGML Coral integration" OFF)
if(CORAL_BUILD_M2)
    add_subdirectory(src/ggml)
endif()
