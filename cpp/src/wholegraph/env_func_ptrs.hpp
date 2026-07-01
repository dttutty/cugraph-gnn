/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/env_func_ptrs.h>

namespace wholegraph {

struct default_memory_context_t {
  wholegraph_tensor_description_t desc;
  wholegraph_memory_allocation_type_t allocation_type;
  void* ptr;
};

/**
 * @brief : Default environment functions for memory allocation.
 * Will use cudaMalloc/cudaFree, cudaMallocHost/cudaFreeHost, malloc/free.
 * Useful for function tests, NOT designed for performance tests.
 *
 * @return : pointers to the functions of current CUDA device
 */
wholegraph_env_func_t* get_default_env_func();

/**
 * @brief : Environment functions for memory allocation with caches.
 * Will cache allocated memory blocks, and reuse if possible.
 * Minimal block size is 256 bytes, block with size < 1G bytes is aligned to power of 2,
 * block with size >= 1G bytes is aligned to 1G bytes.
 * Useful for performance tests. Need warm up to fill caches.
 *
 * @return : pointers to the functions of current CUDA device
 */
wholegraph_env_func_t* get_cached_env_func();

/**
 * @brief : drop all caches of inside cached allocator of current CUDA device
 */
void drop_cached_env_func_cache();

}  // namespace wholegraph
