/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

namespace wholegraph_ops {
namespace testing {

void host_random_init_array(void* array,
                            wholegraph_array_description_t array_desc,
                            int64_t low,
                            int64_t high);
void host_prefix_sum_array(void* array, wholegraph_array_description_t array_desc);

void copy_host_array_to_wholegraph(void* host_array,
                                    wholegraph_handle_t array_handle,
                                    wholegraph_array_description_t array_desc,
                                    cudaStream_t stream);

void wholegraph_csr_unweighted_sample_without_replacement_cpu(
  void* host_csr_row_ptr,
  wholegraph_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholegraph_array_description_t csr_col_ptr_desc,
  void* host_center_nodes,
  wholegraph_array_description_t center_node_desc,
  int max_sample_count,
  void** host_ref_output_sample_offset,
  wholegraph_array_description_t output_sample_offset_desc,
  void** host_ref_output_dest_nodes,
  void** host_ref_output_center_nodes_local_id,
  void** host_ref_output_global_edge_id,
  int* output_sample_dest_nodes_count,
  unsigned long long random_seed);

void wholegraph_csr_weighted_sample_without_replacement_cpu(
  void* host_csr_row_ptr,
  wholegraph_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholegraph_array_description_t csr_col_ptr_desc,
  void* host_csr_weight_ptr,
  wholegraph_array_description_t csr_weight_ptr_desc,
  void* host_center_nodes,
  wholegraph_array_description_t center_node_desc,
  int max_sample_count,
  void** host_ref_output_sample_offset,
  wholegraph_array_description_t output_sample_offset_desc,
  void** host_ref_output_dest_nodes,
  void** host_ref_output_center_nodes_local_id,
  void** host_ref_output_global_edge_id,
  int* output_sample_dest_nodes_count,
  unsigned long long random_seed);

void gen_csr_graph(
  int64_t graph_node_count,
  int64_t graph_edge_count,
  void* host_csr_row_ptr,
  wholegraph_array_description_t graph_csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholegraph_array_description_t graph_csr_col_ptr_desc,
  void* host_csr_weight_ptr                                 = nullptr,
  wholegraph_array_description_t graph_csr_weight_ptr_desc = wholegraph_array_description_t{});

void host_check_two_array_same(void* host_array,
                               wholegraph_array_description_t host_array_desc,
                               void* host_ref,
                               wholegraph_array_description_t host_ref_desc);

void segment_sort_output(void* host_output_sample_offset,
                         wholegraph_array_description_t output_sample_offset_desc,
                         void* host_output_dest_nodes,
                         wholegraph_array_description_t output_dest_nodes_desc,
                         void* host_output_global_edge_id,
                         wholegraph_array_description_t output_global_edge_id_desc);

}  // namespace testing
}  // namespace wholegraph_ops
