/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuda_runtime_api.h>

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph.h>
#include <wholegraph/wholegraph_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Append Unique op
 * @param target_nodes_tensor : WholeGraph Tensor of graph csr_row_ptr
 * @param neighbor_nodes_tensor : WholeGraph Tensor of graph csr_col_ptr
 * @param output_unique_node_memory_context : memory context to output dest nodes
 * @param output_neighbor_raw_to_unique_mapping_tensor : pointer to output sample offset, optional
 * output
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t graph_append_unique(
  wholegraph_tensor_t target_nodes_tensor,
  wholegraph_tensor_t neighbor_nodes_tensor,
  void* output_unique_node_memory_context,
  wholegraph_tensor_t output_neighbor_raw_to_unique_mapping_tensor,
  wholegraph_env_func_t* p_env_fns,
  void* stream);

/**
 * Csr Add Self Loop Op
 * @param csr_row_ptr_tensor : WholeGraph Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : WholeGraph Tensor of csr_col_ptr
 * @param output_csr_row_ptr_tensor : WholeGraph Tensor of output_csr_row_ptr
 * @param output_csr_col_ptr_tensor : WholeGraph Tensor of output_csr_col_ptr
 * @param stream : CUDA stream to use
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t csr_add_self_loop(wholegraph_tensor_t csr_row_ptr_tensor,
                                           wholegraph_tensor_t csr_col_ptr_tensor,
                                           wholegraph_tensor_t output_csr_row_ptr_tensor,
                                           wholegraph_tensor_t output_csr_col_ptr_tensor,
                                           void* stream);

#ifdef __cplusplus
}
#endif
