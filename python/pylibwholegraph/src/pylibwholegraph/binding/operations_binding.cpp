/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "wholegraph_binding.hpp"

wholegraph_env_func_t* env_functions_from_int(int64_t p_env_fns_int)
{
  return reinterpret_cast<wholegraph_env_func_t*>(static_cast<uintptr_t>(p_env_fns_int));
}

void* stream_from_int(int64_t stream_int)
{
  return reinterpret_cast<void*>(static_cast<uintptr_t>(stream_int));
}

void* void_pointer_from_int(int64_t pointer_int)
{
  return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer_int));
}

void py_wholegraph_env_test_op(WrappedLocalTensorNB const& input_tensor,
                               WrappedLocalTensorNB const& output_fixed_tensor,
                               int64_t output_variable_device_tensor_handle,
                               int64_t output_variable_pinned_tensor_handle,
                               int64_t output_variable_host_tensor_handle,
                               int64_t output_variable_entry_count,
                               int64_t p_env_fns_int,
                               int64_t stream_int)
{
  check_wholegraph_error_code(
    wholegraph_env_test_op(input_tensor.c_handle(),
                           output_fixed_tensor.c_handle(),
                           void_pointer_from_int(output_variable_device_tensor_handle),
                           void_pointer_from_int(output_variable_pinned_tensor_handle),
                           void_pointer_from_int(output_variable_host_tensor_handle),
                           output_variable_entry_count,
                           env_functions_from_int(p_env_fns_int),
                           stream_from_int(stream_int)));
}

void wholegraph_gather_op(PyWholeGraphTensorNB const& wholegraph_tensor,
                          WrappedLocalTensorNB const& indices_tensor,
                          WrappedLocalTensorNB const& output_tensor,
                          int64_t p_env_fns_int,
                          int64_t stream_int)
{
  check_wholegraph_error_code(wholegraph_gather(wholegraph_tensor.c_handle(),
                                                indices_tensor.c_handle(),
                                                output_tensor.c_handle(),
                                                env_functions_from_int(p_env_fns_int),
                                                stream_from_int(stream_int)));
}

void wholegraph_scatter_op(WrappedLocalTensorNB const& input_tensor,
                           WrappedLocalTensorNB const& indices_tensor,
                           PyWholeGraphTensorNB const& wholegraph_tensor,
                           int64_t p_env_fns_int,
                           int64_t stream_int)
{
  check_wholegraph_error_code(wholegraph_scatter(input_tensor.c_handle(),
                                                 indices_tensor.c_handle(),
                                                 wholegraph_tensor.c_handle(),
                                                 env_functions_from_int(p_env_fns_int),
                                                 stream_from_int(stream_int)));
}

void embedding_gather_forward(PyWholeGraphEmbeddingNB const& embedding,
                              WrappedLocalTensorNB const& indices_tensor,
                              WrappedLocalTensorNB const& output_tensor,
                              bool adjust_cache,
                              int64_t p_env_fns_int,
                              int64_t stream_int)
{
  check_wholegraph_error_code(
    wholegraph_embedding_gather(embedding.c_handle(),
                                indices_tensor.c_handle(),
                                output_tensor.c_handle(),
                                adjust_cache,
                                env_functions_from_int(p_env_fns_int),
                                stream_int));
}

void embedding_gather_gradient_apply(PyWholeGraphEmbeddingNB const& embedding,
                                     WrappedLocalTensorNB const& indices_tensor,
                                     WrappedLocalTensorNB const& grads_tensor,
                                     bool adjust_cache,
                                     float lr,
                                     int64_t p_env_fns_int,
                                     int64_t stream_int)
{
  check_wholegraph_error_code(
    wholegraph_embedding_gather_gradient_apply(embedding.c_handle(),
                                               indices_tensor.c_handle(),
                                               grads_tensor.c_handle(),
                                               adjust_cache,
                                               lr,
                                               env_functions_from_int(p_env_fns_int),
                                               stream_int));
}

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
  int64_t stream_int)
{
  check_wholegraph_error_code(wholegraph_csr_unweighted_sample_without_replacement(
    wg_csr_row_ptr_tensor.c_handle(),
    wg_csr_col_ptr_tensor.c_handle(),
    center_nodes_tensor.c_handle(),
    max_sample_count,
    output_sample_offset_tensor.c_handle(),
    void_pointer_from_int(output_dest_memory_context),
    void_pointer_from_int(output_center_localid_memory_context),
    void_pointer_from_int(output_edge_gid_memory_context),
    random_seed,
    env_functions_from_int(p_env_fns_int),
    stream_from_int(stream_int)));
}

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
  int64_t stream_int)
{
  check_wholegraph_error_code(wholegraph_csr_weighted_sample_without_replacement(
    wg_csr_row_ptr_tensor.c_handle(),
    wg_csr_col_ptr_tensor.c_handle(),
    wg_csr_weight_ptr_tensor.c_handle(),
    center_nodes_tensor.c_handle(),
    max_sample_count,
    output_sample_offset_tensor.c_handle(),
    void_pointer_from_int(output_dest_memory_context),
    void_pointer_from_int(output_center_localid_memory_context),
    void_pointer_from_int(output_edge_gid_memory_context),
    random_seed,
    env_functions_from_int(p_env_fns_int),
    stream_from_int(stream_int)));
}

void host_generate_random_positive_int(int64_t random_seed,
                                       int64_t subsequence,
                                       WrappedLocalTensorNB const& output)
{
  check_wholegraph_error_code(
    generate_random_positive_int_cpu(random_seed, subsequence, output.c_handle()));
}

void host_generate_exponential_distribution_negative_float(
  int64_t random_seed, int64_t subsequence, WrappedLocalTensorNB const& output)
{
  check_wholegraph_error_code(generate_exponential_distribution_negative_float_cpu(
    random_seed, subsequence, output.c_handle()));
}

void append_unique(WrappedLocalTensorNB const& target_nodes_tensor,
                   WrappedLocalTensorNB const& neighbor_nodes_tensor,
                   int64_t output_unique_node_memory_context,
                   WrappedLocalTensorNB const& output_neighbor_raw_to_unique_mapping_tensor,
                   int64_t p_env_fns_int,
                   int64_t stream_int)
{
  check_wholegraph_error_code(graph_append_unique(
    target_nodes_tensor.c_handle(),
    neighbor_nodes_tensor.c_handle(),
    void_pointer_from_int(output_unique_node_memory_context),
    output_neighbor_raw_to_unique_mapping_tensor.c_handle(),
    env_functions_from_int(p_env_fns_int),
    stream_from_int(stream_int)));
}

void add_csr_self_loop(WrappedLocalTensorNB const& csr_row_ptr_tensor,
                       WrappedLocalTensorNB const& csr_col_ptr_tensor,
                       WrappedLocalTensorNB const& output_csr_row_ptr_tensor,
                       WrappedLocalTensorNB const& output_csr_col_ptr_tensor,
                       int64_t stream_int)
{
  check_wholegraph_error_code(csr_add_self_loop(csr_row_ptr_tensor.c_handle(),
                                                csr_col_ptr_tensor.c_handle(),
                                                output_csr_row_ptr_tensor.c_handle(),
                                                output_csr_col_ptr_tensor.c_handle(),
                                                stream_from_int(stream_int)));
}

int fork_get_gpu_count() { return fork_get_device_count(); }
