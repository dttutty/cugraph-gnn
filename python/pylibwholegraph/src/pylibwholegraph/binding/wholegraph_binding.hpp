/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "communicator_binding.hpp"
#include "embedding_binding.hpp"
#include "global_context_binding.hpp"
#include "tensor_binding.hpp"

PyWholeGraphUniqueIDNB create_unique_id();
PyWholeGraphCommNB create_communicator(PyWholeGraphUniqueIDNB const& unique_id,
                                       int world_rank,
                                       int world_size);
void destroy_communicator(PyWholeGraphCommNB const& comm);
PyWholeGraphCommNB split_communicator(PyWholeGraphCommNB const& comm, int color, int key);
void communicator_set_distributed_backend(
  PyWholeGraphCommNB& comm, wholegraph_distributed_backend_t distributed_backend);
size_t equal_partition_plan(size_t entry_count, int world_size);

PyWholeGraphHandleNB malloc_wholegraph(size_t total_size,
                                       PyWholeGraphCommNB const& comm,
                                       wholegraph_memory_type_t memory_type,
                                       wholegraph_memory_location_t memory_location,
                                       size_t data_granularity,
                                       nb::object rank_entry_partition);
void free_handle(PyWholeGraphHandleNB const& handle);
PyWholeGraphTensorNB create_wholegraph_array(wholegraph_dtype_t dtype,
                                             int64_t size,
                                             PyWholeGraphCommNB const& comm,
                                             wholegraph_memory_type_t memory_type,
                                             wholegraph_memory_location_t memory_location,
                                             nb::object tensor_entry_partition);
PyWholeGraphTensorNB create_wholegraph_matrix(wholegraph_dtype_t dtype,
                                              int64_t row,
                                              int64_t column,
                                              int64_t stride,
                                              PyWholeGraphCommNB const& comm,
                                              wholegraph_memory_type_t memory_type,
                                              wholegraph_memory_location_t memory_location,
                                              nb::object tensor_entry_partition);
PyWholeGraphTensorNB create_wholegraph_tensor(
  PyWholeGraphTensorDescriptionNB const& tensor_description,
  PyWholeGraphCommNB const& comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  nb::object tensor_entry_partition);
PyWholeGraphTensorNB make_tensor_as_wholegraph(
  PyWholeGraphTensorDescriptionNB const& tensor_description, int64_t data_ptr);
PyWholeGraphTensorNB make_handle_as_wholegraph(
  PyWholeGraphTensorDescriptionNB const& tensor_description, PyWholeGraphHandleNB const& handle);
void destroy_wholegraph_tensor(PyWholeGraphTensorNB const& tensor);
void load_wholegraph_handle_from_filelist(int64_t wholegraph_handle_int_ptr,
                                          size_t memory_offset,
                                          size_t memory_entry_size,
                                          size_t file_entry_size,
                                          int round_robin_size,
                                          nb::object file_list);
void store_wholegraph_handle_to_file(int64_t wholegraph_handle_int_ptr,
                                     size_t memory_offset,
                                     size_t memory_entry_size,
                                     size_t file_entry_size,
                                     std::string const& file_name);

WholeGraphCachePolicyNB create_non_cache_policy();
PyWholeGraphEmbeddingNB create_embedding(
  PyWholeGraphTensorDescriptionNB const& tensor_description,
  PyWholeGraphCommNB const& comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  WholeGraphCachePolicyNB const& cache_policy,
  nb::object embedding_entry_partition,
  int user_defined_sms,
  int round_robin_size);

void py_wholegraph_env_test_op(WrappedLocalTensorNB const& input_tensor,
                               WrappedLocalTensorNB const& output_fixed_tensor,
                               int64_t output_variable_device_tensor_handle,
                               int64_t output_variable_pinned_tensor_handle,
                               int64_t output_variable_host_tensor_handle,
                               int64_t output_variable_entry_count,
                               int64_t p_env_fns_int,
                               int64_t stream_int);
void wholegraph_gather_op(PyWholeGraphTensorNB const& wholegraph_tensor,
                          WrappedLocalTensorNB const& indices_tensor,
                          WrappedLocalTensorNB const& output_tensor,
                          int64_t p_env_fns_int,
                          int64_t stream_int);
void wholegraph_scatter_op(WrappedLocalTensorNB const& input_tensor,
                           WrappedLocalTensorNB const& indices_tensor,
                           PyWholeGraphTensorNB const& wholegraph_tensor,
                           int64_t p_env_fns_int,
                           int64_t stream_int);
void embedding_gather_forward(PyWholeGraphEmbeddingNB const& embedding,
                              WrappedLocalTensorNB const& indices_tensor,
                              WrappedLocalTensorNB const& output_tensor,
                              bool adjust_cache,
                              int64_t p_env_fns_int,
                              int64_t stream_int);
void embedding_gather_gradient_apply(PyWholeGraphEmbeddingNB const& embedding,
                                     WrappedLocalTensorNB const& indices_tensor,
                                     WrappedLocalTensorNB const& grads_tensor,
                                     bool adjust_cache,
                                     float lr,
                                     int64_t p_env_fns_int,
                                     int64_t stream_int);
void csr_unweighted_sample_without_replacement(
  PyWholeGraphTensorNB const& wg_csr_row_ptr_tensor,
  PyWholeGraphTensorNB const& wg_csr_col_ptr_tensor,
  WrappedLocalTensorNB const& center_nodes_tensor,
  int max_sample_count,
  WrappedLocalTensorNB const& output_sample_offset_tensor,
  int64_t output_dest_memory_context,
  int64_t output_center_localid_memory_context,
  int64_t output_edge_gid_memory_context,
  unsigned long long random_seed,
  int64_t p_env_fns_int,
  int64_t stream_int);
void csr_weighted_sample_without_replacement(
  PyWholeGraphTensorNB const& wg_csr_row_ptr_tensor,
  PyWholeGraphTensorNB const& wg_csr_col_ptr_tensor,
  PyWholeGraphTensorNB const& wg_csr_weight_ptr_tensor,
  WrappedLocalTensorNB const& center_nodes_tensor,
  int max_sample_count,
  WrappedLocalTensorNB const& output_sample_offset_tensor,
  int64_t output_dest_memory_context,
  int64_t output_center_localid_memory_context,
  int64_t output_edge_gid_memory_context,
  unsigned long long random_seed,
  int64_t p_env_fns_int,
  int64_t stream_int);
void host_generate_random_positive_int(int64_t random_seed,
                                       int64_t subsequence,
                                       WrappedLocalTensorNB const& output);
void host_generate_exponential_distribution_negative_float(
  int64_t random_seed, int64_t subsequence, WrappedLocalTensorNB const& output);
void append_unique(WrappedLocalTensorNB const& target_nodes_tensor,
                   WrappedLocalTensorNB const& neighbor_nodes_tensor,
                   int64_t output_unique_node_memory_context,
                   WrappedLocalTensorNB const& output_neighbor_raw_to_unique_mapping_tensor,
                   int64_t p_env_fns_int,
                   int64_t stream_int);
void add_csr_self_loop(WrappedLocalTensorNB const& csr_row_ptr_tensor,
                       WrappedLocalTensorNB const& csr_col_ptr_tensor,
                       WrappedLocalTensorNB const& output_csr_row_ptr_tensor,
                       WrappedLocalTensorNB const& output_csr_col_ptr_tensor,
                       int64_t stream_int);
int fork_get_gpu_count();

void register_wholegraph_binding(nb::module_& m);
