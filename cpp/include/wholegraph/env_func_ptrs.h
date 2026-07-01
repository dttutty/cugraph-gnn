/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>
#include <wholegraph/tensor_description.h>

/**
 * Function pointers for memory allocation.
 * Input tensor memory should be allocated and use void* pointer to the memory and
 * wholegraph_array_description_t or wholegraph_matrix_description_t to specify the shape Output
 * tensor with fixed size should be the same as Input tensor. Output tensor with shape determined by
 * Op should has void* memory_context input and allocated by wholegraph_malloc_func_t functions.
 */

#ifdef __cplusplus
extern "C" {
#endif

enum wholegraph_memory_allocation_type_t {
  WHOLEGRAPH_MA_NONE = 0,
  WHOLEGRAPH_MA_DEVICE,
  WHOLEGRAPH_MA_HOST,
  WHOLEGRAPH_MA_PINNED,
};

/**
 * Function pointer to create temporary memory context.
 */
typedef void (*wholegraph_create_memory_context_func_t)(void** memory_context,
                                                         void* global_context);

typedef void (*wholegraph_destroy_memory_context_func_t)(void* memory_context,
                                                          void* global_context);

typedef void* (*wholegraph_malloc_func_t)(
  wholegraph_tensor_description_t* desc,
  wholegraph_memory_allocation_type_t memory_allocation_type,
  void* memory_context,
  void* global_context);

typedef void (*wholegraph_free_func_t)(void* memory_context, void* global_context);

struct wholegraph_temp_memory_func_t {
  wholegraph_create_memory_context_func_t create_memory_context_fn;
  wholegraph_destroy_memory_context_func_t destroy_memory_context_fn;
  wholegraph_malloc_func_t malloc_fn;
  wholegraph_free_func_t free_fn;
  void* global_context;
};
struct wholegraph_output_memory_func_t {
  wholegraph_malloc_func_t malloc_fn;
  wholegraph_free_func_t free_fn;
  void* global_context;
};

struct wholegraph_env_func_t {
  wholegraph_temp_memory_func_t temporary_fns; /* function pointers to create temporary memory */
  wholegraph_output_memory_func_t output_fns;  /* function pointers to create Op output memory */
};

cudaDeviceProp* get_device_prop(int dev_id);

#ifdef __cplusplus
}
#endif
