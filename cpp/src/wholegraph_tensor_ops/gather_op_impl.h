/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/global_reference.h>
#include <wholegraph/wholegraph.h>

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
  int gather_sms);

wholegraph_error_code_t wholegraph_gather_nccl(wholegraph_handle_t wholegraph_handle,
                                                 wholegraph_matrix_description_t wholegraph_desc,
                                                 void* indices,
                                                 wholegraph_array_description_t indice_desc,
                                                 void* output,
                                                 wholegraph_matrix_description_t output_desc,
                                                 wholegraph_env_func_t* p_env_fns,
                                                 cudaStream_t stream,
                                                 int gather_sms);

wholegraph_error_code_t wholegraph_gather_distributed(
  wholegraph_handle_t wholegraph_handle,
  wholegraph_matrix_description_t wholegraph_desc,
  void* indices,
  wholegraph_array_description_t indice_desc,
  void* output,
  wholegraph_matrix_description_t output_desc,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms);

}  // namespace wholegraph_tensor_ops
