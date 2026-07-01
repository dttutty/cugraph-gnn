# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(WHOLEGRAPH)

project(
  WHOLEGRAPH
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Write the version header
rapids_cmake_write_version_file(include/wholegraph/version_config.hpp)

# ##################################################################################################
# * User Options  ------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build libwholegraph shared libraries" ON)
option(CMAKE_CUDA_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF
)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build benchmark" ON)
option(USE_NCCL_RUNTIME_WHEEL "Use the NCCL wheel at runtime instead of the system library" OFF)

# ##################################################################################################
# * Set options based on user defined one  -----------------------------------

# ##################################################################################################
# * Base rapids options ------------------------------------------------------ default build type
#   rapids_cmake_build_type(Release)
rapids_cmake_build_type(RelWithDebInfo)

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ON)

message("CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

# ##################################################################################################
# * compiler options -----------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET wholegraph-exports
  INSTALL_EXPORT_SET wholegraph-exports
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WHOLEGRAPH_CXX_FLAGS "")
set(WHOLEGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND WHOLEGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)

message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

# Enable device code compression to reduce binary sizes
include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
rapids_cuda_enable_fatbin_compression(VARIABLE WHOLEGRAPH_CUDA_FLAGS TUNE_FOR rapids)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CMAKE_CUDA_LINEINFO)
  list(APPEND WHOLEGRAPH_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Building with debugging flags")
  list(APPEND WHOLEGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()

# ##################################################################################################
# * find CPM based dependencies  -----------------------------------------------

rapids_cpm_init()

include(cmake/thirdparty/get_raft.cmake)
include(cmake/thirdparty/get_nccl.cmake)

file(
  GLOB
  WHOLEGRAPH_SOURCES
  "src/*.cpp"
  "src/wholegraph/*.cpp"
  "src/wholegraph_tensor_ops/*.cpp"
  "src/wholegraph_tensor_ops/*.cu"
  "src/wholegraph_tensor_ops/functions/*.cu"
  "src/wholegraph_ops/*.cpp"
  "src/wholegraph_ops/*.cu"
  "src/graph_ops/*.cu"
  "src/graph_ops/*.cpp"
)

add_library(wholegraph)
add_library(wholegraph::wholegraph ALIAS wholegraph)

target_sources(wholegraph PRIVATE ${WHOLEGRAPH_SOURCES})

set_target_properties(
  wholegraph
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  wholegraph PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${WHOLEGRAPH_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${WHOLEGRAPH_CUDA_FLAGS}>"
)

# ##################################################################################################
# * include paths --------------------------------------------------------------

set(WHOLEGRAPH_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    CACHE STRING "Path to wholegraph include directory"
)

target_include_directories(
  wholegraph
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
  PUBLIC "$<BUILD_INTERFACE:${NCCL_INCLUDE_DIR}>" "$<BUILD_INTERFACE:${WHOLEGRAPH_INCLUDE_DIR}>"
         "$<INSTALL_INTERFACE:include>"
)

# ##################################################################################################
# * link libraries -------------------------------------------------------------
target_link_libraries(
  wholegraph
  PUBLIC CUDA::cuda_driver CUDA::cudart_static raft::raft
  PRIVATE NCCL::NCCL
)

if(CUDAToolkit_VERSION VERSION_GREATER "12.3")
  # Link the NVML library if CUDA version is greater than 12.3
  target_link_libraries(wholegraph PRIVATE CUDA::nvml)
endif()

# optionally build tests
if(BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
  include(CTest) # calls enable_testing()

  add_subdirectory(tests)
endif()

# optionally build benchmark
if(BUILD_BENCHMARKS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  add_subdirectory(bench)
endif()

# ##################################################################################################
# * install targets ----------------------------------------------------------

set(doc_string [=[
]=]
)
set(code_string [=[
]=]
)

set(WHOLEGRAPH_PUBLIC_HEADERS
    wholegraph/device_reference.cuh
    wholegraph/embedding.h
    wholegraph/env_func_ptrs.h
    wholegraph/global_reference.h
    wholegraph/tensor_description.h
    wholegraph/wholegraph.h
    wholegraph/wholegraph_tensor.h
    wholegraph/wholegraph_tensor_op.h
    wholegraph/wholegraph_op.h
    wholegraph/graph_op.h
)

foreach(file IN LISTS WHOLEGRAPH_PUBLIC_HEADERS)
  cmake_path(GET file PARENT_PATH file_out_)
  string(JOIN "/" out_path "include" ${file_out_})
  install(FILES "${WHOLEGRAPH_INCLUDE_DIR}/${file}" DESTINATION "${out_path}")
endforeach()

rapids_cmake_install_lib_dir(lib_dir)

set(PROGRAM_PERMISSIONS_DEFAULT OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                                WORLD_READ WORLD_EXECUTE
)

set(WHOLEGRAPH_INSTALL_LIBS wholegraph)

install(
  TARGETS ${WHOLEGRAPH_INSTALL_LIBS}
  DESTINATION ${lib_dir}
  PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT}
  EXPORT wholegraph-exports
)

rapids_export(
  INSTALL wholegraph
  EXPORT_SET wholegraph-exports
  GLOBAL_TARGETS wholegraph
  NAMESPACE wholegraph::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

# build export targets
rapids_export(
  BUILD wholegraph
  EXPORT_SET wholegraph-exports
  GLOBAL_TARGETS wholegraph
  NAMESPACE wholegraph::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)
