/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholegraph/wholegraph_op.h>

#include <wholegraph_ops/weighted_sample_without_replacement_impl.h>

#include "error.hpp"
#include "logger.hpp"

wholegraph_error_code_t wholegraph_csr_weighted_sample_without_replacement(
  wholegraph_tensor_t wg_csr_row_ptr_tensor,
  wholegraph_tensor_t wg_csr_col_ptr_tensor,
  wholegraph_tensor_t wg_csr_weight_ptr_tensor,
  wholegraph_tensor_t center_nodes_tensor,
  int max_sample_count,
  wholegraph_tensor_t output_sample_offset_tensor,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholegraph_env_func_t* p_env_fns,
  void* stream)
{
  bool const csr_row_ptr_has_handle = wholegraph_tensor_has_handle(wg_csr_row_ptr_tensor);
  wholegraph_memory_type_t csr_row_ptr_memory_type = WHOLEGRAPH_MT_NONE;
  if (csr_row_ptr_has_handle) {
    csr_row_ptr_memory_type =
      wholegraph_get_memory_type(wholegraph_tensor_get_memory_handle(wg_csr_row_ptr_tensor));
  }
  WHOLEGRAPH_EXPECTS_NOTHROW(!csr_row_ptr_has_handle ||
                                csr_row_ptr_memory_type == WHOLEGRAPH_MT_CONTINUOUS,
                              "Memory type not supported.");
  bool const csr_col_ptr_has_handle = wholegraph_tensor_has_handle(wg_csr_col_ptr_tensor);
  wholegraph_memory_type_t csr_col_ptr_memory_type = WHOLEGRAPH_MT_NONE;
  if (csr_col_ptr_has_handle) {
    csr_col_ptr_memory_type =
      wholegraph_get_memory_type(wholegraph_tensor_get_memory_handle(wg_csr_col_ptr_tensor));
  }
  WHOLEGRAPH_EXPECTS_NOTHROW(!csr_col_ptr_has_handle ||
                                csr_col_ptr_memory_type == WHOLEGRAPH_MT_CONTINUOUS,
                              "Memory type not supported.");
  bool const csr_weight_ptr_has_handle = wholegraph_tensor_has_handle(wg_csr_weight_ptr_tensor);
  wholegraph_memory_type_t csr_weight_ptr_memory_type = WHOLEGRAPH_MT_NONE;
  if (csr_weight_ptr_has_handle) {
    csr_weight_ptr_memory_type =
      wholegraph_get_memory_type(wholegraph_tensor_get_memory_handle(wg_csr_weight_ptr_tensor));
  }
  WHOLEGRAPH_EXPECTS_NOTHROW(!csr_weight_ptr_has_handle ||
                                csr_weight_ptr_memory_type == WHOLEGRAPH_MT_CONTINUOUS,
                              "Memory type not supported.");

  auto csr_row_ptr_tensor_description =
    *wholegraph_tensor_get_tensor_description(wg_csr_row_ptr_tensor);
  auto csr_col_ptr_tensor_description =
    *wholegraph_tensor_get_tensor_description(wg_csr_col_ptr_tensor);
  auto csr_weight_ptr_tensor_description =
    *wholegraph_tensor_get_tensor_description(wg_csr_weight_ptr_tensor);
  if (csr_row_ptr_tensor_description.dim != 1) {
    WHOLEGRAPH_ERROR("wg_csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_description.dim != 1) {
    WHOLEGRAPH_ERROR("wg_csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (csr_weight_ptr_tensor_description.dim != 1) {
    WHOLEGRAPH_ERROR("wg_csr_weight_ptr_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  wholegraph_array_description_t wg_csr_row_ptr_desc, wg_csr_col_ptr_desc, wg_csr_weight_ptr_desc;
  if (!wholegraph_convert_tensor_desc_to_array(&wg_csr_row_ptr_desc,
                                                &csr_row_ptr_tensor_description)) {
    WHOLEGRAPH_ERROR("Input wg_csr_row_ptr_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (!wholegraph_convert_tensor_desc_to_array(&wg_csr_col_ptr_desc,
                                                &csr_col_ptr_tensor_description)) {
    WHOLEGRAPH_ERROR("Input wg_csr_col_ptr_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (!wholegraph_convert_tensor_desc_to_array(&wg_csr_weight_ptr_desc,
                                                &csr_weight_ptr_tensor_description)) {
    WHOLEGRAPH_ERROR("Input wg_csr_weight_ptr_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }

  wholegraph_tensor_description_t center_nodes_tensor_desc =
    *wholegraph_tensor_get_tensor_description(center_nodes_tensor);
  if (center_nodes_tensor_desc.dim != 1) {
    WHOLEGRAPH_ERROR("Input center_nodes_tensor should be 1D tensor");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  wholegraph_array_description_t center_nodes_desc;
  if (!wholegraph_convert_tensor_desc_to_array(&center_nodes_desc, &center_nodes_tensor_desc)) {
    WHOLEGRAPH_ERROR("Input center_nodes_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }

  wholegraph_tensor_description_t output_sample_offset_tensor_desc =
    *wholegraph_tensor_get_tensor_description(output_sample_offset_tensor);
  if (output_sample_offset_tensor_desc.dim != 1) {
    WHOLEGRAPH_ERROR("Output output_sample_offset_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  wholegraph_array_description_t output_sample_offset_desc;
  if (!wholegraph_convert_tensor_desc_to_array(&output_sample_offset_desc,
                                                &output_sample_offset_tensor_desc)) {
    WHOLEGRAPH_ERROR("Output output_sample_offset_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }

  void* center_nodes         = wholegraph_tensor_get_data_pointer(center_nodes_tensor);
  void* output_sample_offset = wholegraph_tensor_get_data_pointer(output_sample_offset_tensor);
  wholegraph_gref_t wg_csr_row_ptr_gref, wg_csr_col_ptr_gref, wg_csr_weight_ptr_gref;
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_tensor_get_global_reference(wg_csr_row_ptr_tensor, &wg_csr_row_ptr_gref));
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_tensor_get_global_reference(wg_csr_col_ptr_tensor, &wg_csr_col_ptr_gref));
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_tensor_get_global_reference(wg_csr_weight_ptr_tensor, &wg_csr_weight_ptr_gref));

  return wholegraph_ops::wholegraph_csr_weighted_sample_without_replacement_mapped(
    wg_csr_row_ptr_gref,
    wg_csr_row_ptr_desc,
    wg_csr_col_ptr_gref,
    wg_csr_col_ptr_desc,
    wg_csr_weight_ptr_gref,
    wg_csr_weight_ptr_desc,
    center_nodes,
    center_nodes_desc,
    max_sample_count,
    output_sample_offset,
    output_sample_offset_desc,
    output_dest_memory_context,
    output_center_localid_memory_context,
    output_edge_gid_memory_context,
    random_seed,
    p_env_fns,
    static_cast<cudaStream_t>(stream));
}
