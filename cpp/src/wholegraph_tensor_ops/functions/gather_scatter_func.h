/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/global_reference.h>
#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

namespace wholegraph_tensor_ops {

wholegraph_error_code_t gather_func(wholegraph_gref_t embedding_gref,
                                     wholegraph_matrix_description_t embedding_desc,
                                     void* indices,
                                     wholegraph_array_description_t indices_desc,
                                     void* output,
                                     wholegraph_matrix_description_t output_desc,
                                     cudaStream_t stream,
                                     int gather_sms = -1);

wholegraph_error_code_t gather_with_sorted_ids_func(
  wholegraph_gref_t embedding_gref,
  wholegraph_matrix_description_t embedding_desc,
  void* indices,
  wholegraph_array_description_t indices_desc,
  void* raw_indices,
  wholegraph_array_description_t raw_indices_desc,
  void* output,
  wholegraph_matrix_description_t output_desc,
  cudaStream_t stream,
  int gather_sms);

wholegraph_error_code_t scatter_func(const void* input,
                                      wholegraph_matrix_description_t input_desc,
                                      void* indices,
                                      wholegraph_array_description_t indices_desc,
                                      wholegraph_gref_t embedding_gref,
                                      wholegraph_matrix_description_t embedding_desc,
                                      cudaStream_t stream,
                                      int scatter_sms = -1);

}  // namespace wholegraph_tensor_ops
