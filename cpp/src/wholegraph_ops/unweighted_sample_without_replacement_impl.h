/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/global_reference.h>
#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

namespace wholegraph_ops {
wholegraph_error_code_t wholegraph_csr_unweighted_sample_without_replacement_mapped(
  wholegraph_gref_t wg_csr_row_ptr,
  wholegraph_array_description_t wg_csr_row_ptr_desc,
  wholegraph_gref_t wg_csr_col_ptr,
  wholegraph_array_description_t wg_csr_col_ptr_desc,
  void* center_nodes,
  wholegraph_array_description_t center_nodes_desc,
  int max_sample_count,
  void* output_sample_offset,
  wholegraph_array_description_t output_sample_offset_desc,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream);

wholegraph_error_code_t wholegraph_csr_unweighted_sample_without_replacement_nccl(
  wholegraph_handle_t csr_row_wholegraph_handle,
  wholegraph_handle_t csr_col_wholegraph_handle,
  wholegraph_tensor_description_t wg_csr_row_ptr_desc,
  wholegraph_tensor_description_t wg_csr_col_ptr_desc,
  void* center_nodes,
  wholegraph_array_description_t center_nodes_desc,
  int max_sample_count,
  void* output_sample_offset,
  wholegraph_array_description_t output_sample_offset_desc,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream);
}  // namespace wholegraph_ops
