/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuda_runtime_api.h>
#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

namespace graph_ops {
wholegraph_error_code_t csr_add_self_loop_impl(
  int* csr_row_ptr,
  wholegraph_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholegraph_array_description_t csr_col_ptr_array_desc,
  int* output_csr_row_ptr,
  wholegraph_array_description_t output_csr_row_ptr_array_desc,
  int* output_csr_col_ptr,
  wholegraph_array_description_t output_csr_col_ptr_array_desc,
  cudaStream_t stream);

}  // namespace graph_ops
