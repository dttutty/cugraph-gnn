/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph.h>

#include "cuda_macros.hpp"
#include "wholegraph_tensor_ops/functions/gather_scatter_func.h"

namespace wholegraph_tensor_ops {

wholegraph_error_code_t wholegraph_scatter_mapped(
  void* input,
  wholegraph_matrix_description_t input_desc,
  void* indices,
  wholegraph_array_description_t indices_desc,
  wholegraph_gref_t wholegraph_gref,
  wholegraph_matrix_description_t wholegraph_desc,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms)
{
  WHOLEGRAPH_RETURN_ON_FAIL(scatter_func(input,
                                          input_desc,
                                          indices,
                                          indices_desc,
                                          wholegraph_gref,
                                          wholegraph_desc,
                                          stream,
                                          scatter_sms));
  WG_CUDA_DEBUG_SYNC_STREAM(stream);
  return WHOLEGRAPH_SUCCESS;
}

}  // namespace wholegraph_tensor_ops
