/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "csr_add_self_loop_impl.h"
#include "error.hpp"
#include "logger.hpp"
#include <wholegraph/graph_op.h>

wholegraph_error_code_t csr_add_self_loop(wholegraph_tensor_t csr_row_ptr_tensor,
                                           wholegraph_tensor_t csr_col_ptr_tensor,
                                           wholegraph_tensor_t output_csr_row_ptr_tensor,
                                           wholegraph_tensor_t output_csr_col_ptr_tensor,
                                           void* stream)
{
  auto csr_row_ptr_tensor_desc = *wholegraph_tensor_get_tensor_description(csr_row_ptr_tensor);
  if (csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEGRAPH_ERROR("Input csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (csr_row_ptr_tensor_desc.dtype != WHOLEGRAPH_DT_INT) {
    WHOLEGRAPH_ERROR("Input csr_row_ptr_tensor should be int tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto csr_col_ptr_tensor_desc = *wholegraph_tensor_get_tensor_description(csr_col_ptr_tensor);
  if (csr_col_ptr_tensor_desc.dim != 1) {
    WHOLEGRAPH_ERROR("Input csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_desc.dtype != WHOLEGRAPH_DT_INT) {
    WHOLEGRAPH_ERROR("Input csr_col_ptr_tensor should be int tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto output_csr_row_ptr_tensor_desc =
    *wholegraph_tensor_get_tensor_description(output_csr_row_ptr_tensor);
  if (output_csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEGRAPH_ERROR("Output output_csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (output_csr_row_ptr_tensor_desc.dtype != WHOLEGRAPH_DT_INT) {
    WHOLEGRAPH_ERROR("Output output_csr_row_ptr_tensor should be int tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto output_csr_col_ptr_tensor_desc =
    *wholegraph_tensor_get_tensor_description(output_csr_col_ptr_tensor);
  if (output_csr_col_ptr_tensor_desc.dim != 1) {
    WHOLEGRAPH_ERROR("Output output_csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (output_csr_col_ptr_tensor_desc.dtype != WHOLEGRAPH_DT_INT) {
    WHOLEGRAPH_ERROR("Output output_csr_col_ptr_tensor should be int tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }

  wholegraph_array_description_t csr_row_ptr_array_desc, csr_col_ptr_array_desc,
    output_csr_row_ptr_array_desc, output_csr_col_ptr_array_desc;
  if (!wholegraph_convert_tensor_desc_to_array(&csr_row_ptr_array_desc,
                                                &csr_row_ptr_tensor_desc)) {
    WHOLEGRAPH_ERROR("Input csr_row_ptr_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (!wholegraph_convert_tensor_desc_to_array(&csr_col_ptr_array_desc,
                                                &csr_col_ptr_tensor_desc)) {
    WHOLEGRAPH_ERROR("Input csr_col_ptr_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }

  if (!wholegraph_convert_tensor_desc_to_array(&output_csr_row_ptr_array_desc,
                                                &output_csr_row_ptr_tensor_desc)) {
    WHOLEGRAPH_ERROR("Output output_csr_row_ptr_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (!wholegraph_convert_tensor_desc_to_array(&output_csr_col_ptr_array_desc,
                                                &output_csr_col_ptr_tensor_desc)) {
    WHOLEGRAPH_ERROR("Output output_csr_col_ptr_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  void* csr_row_ptr        = wholegraph_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr        = wholegraph_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* output_csr_row_ptr = wholegraph_tensor_get_data_pointer(output_csr_row_ptr_tensor);
  void* output_csr_col_ptr = wholegraph_tensor_get_data_pointer(output_csr_col_ptr_tensor);
  return graph_ops::csr_add_self_loop_impl(static_cast<int*>(csr_row_ptr),
                                           csr_row_ptr_array_desc,
                                           static_cast<int*>(csr_col_ptr),
                                           csr_col_ptr_array_desc,
                                           static_cast<int*>(output_csr_row_ptr),
                                           output_csr_row_ptr_array_desc,
                                           static_cast<int*>(output_csr_col_ptr),
                                           output_csr_col_ptr_array_desc,
                                           static_cast<cudaStream_t>(stream));
}
