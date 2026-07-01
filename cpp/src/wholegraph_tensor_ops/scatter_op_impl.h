/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/global_reference.h>
#include <wholegraph/wholegraph.h>

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
  int scatter_sms);

wholegraph_error_code_t wholegraph_scatter_nccl(void* input,
                                                  wholegraph_matrix_description_t input_desc,
                                                  void* indices,
                                                  wholegraph_array_description_t indices_desc,
                                                  wholegraph_handle_t wholegraph_handle,
                                                  wholegraph_matrix_description_t wholegraph_desc,
                                                  wholegraph_env_func_t* p_env_fns,
                                                  cudaStream_t stream,
                                                  int scatter_sms);

wholegraph_error_code_t wholegraph_scatter_distributed(
  void* input,
  wholegraph_matrix_description_t input_desc,
  void* indices,
  wholegraph_array_description_t indices_desc,
  wholegraph_handle_t wholegraph_handle,
  wholegraph_matrix_description_t wholegraph_desc,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms);

}  // namespace wholegraph_tensor_ops
