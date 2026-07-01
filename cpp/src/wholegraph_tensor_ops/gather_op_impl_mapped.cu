/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph.h>

#include "cuda_macros.hpp"
#include "wholegraph_tensor_ops/functions/gather_scatter_func.h"
#include "wholegraph_tensor_ops/functions/sort_indices_func.h"
#include "wholegraph_tensor_ops/temp_memory_handle.hpp"
#include "wholegraph_tensor_ops/thrust_allocator.hpp"

namespace wholegraph_tensor_ops {

wholegraph_error_code_t wholegraph_gather_mapped(
  wholegraph_gref_t wholegraph_gref,
  wholegraph_matrix_description_t wholegraph_desc,
  void* indices,
  wholegraph_array_description_t indice_desc,
  void* output,
  wholegraph_matrix_description_t output_desc,
  bool gather_with_sorted_ids,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms)
{
  if (gather_with_sorted_ids) {
    wg_thrust_allocator thrust_allocator(p_env_fns);
    temp_memory_handle dev_indices_after_sort(p_env_fns);
    void* dev_indices_after_sort_ptr =
      dev_indices_after_sort.device_malloc(indice_desc.size, indice_desc.dtype);
    temp_memory_handle dev_raw_indices(p_env_fns);
    void* dev_raw_indices_ptr = dev_raw_indices.device_malloc(indice_desc.size, indice_desc.dtype);
    auto raw_indices_desc = wholegraph_create_array_desc(indice_desc.size, 0, indice_desc.dtype);
    WHOLEGRAPH_RETURN_ON_FAIL(sort_indices_func(indices,
                                                 indice_desc,
                                                 dev_indices_after_sort_ptr,
                                                 dev_raw_indices_ptr,
                                                 &thrust_allocator,
                                                 p_env_fns,
                                                 stream));
    WHOLEGRAPH_RETURN_ON_FAIL(gather_with_sorted_ids_func(wholegraph_gref,
                                                           wholegraph_desc,
                                                           dev_indices_after_sort_ptr,
                                                           indice_desc,
                                                           dev_raw_indices_ptr,
                                                           raw_indices_desc,
                                                           output,
                                                           output_desc,
                                                           stream,
                                                           gather_sms));
  } else {
    WHOLEGRAPH_RETURN_ON_FAIL(gather_func(wholegraph_gref,
                                           wholegraph_desc,
                                           indices,
                                           indice_desc,
                                           output,
                                           output_desc,
                                           stream,
                                           gather_sms));
  }
  WG_CUDA_DEBUG_SYNC_STREAM(stream);
  return WHOLEGRAPH_SUCCESS;
}

}  // namespace wholegraph_tensor_ops
