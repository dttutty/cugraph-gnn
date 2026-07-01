/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "error.hpp"
#include "logger.hpp"
#include <graph_ops/append_unique_impl.h>
#include <wholegraph/graph_op.h>

wholegraph_error_code_t graph_append_unique(
  wholegraph_tensor_t target_nodes_tensor,
  wholegraph_tensor_t neighbor_nodes_tensor,
  void* output_unique_node_memory_context,
  wholegraph_tensor_t output_neighbor_raw_to_unique_mapping_tensor,
  wholegraph_env_func_t* p_env_fns,
  void* stream)
{
  auto target_nodes_tensor_description =
    *wholegraph_tensor_get_tensor_description(target_nodes_tensor);
  if (target_nodes_tensor_description.dim != 1) {
    WHOLEGRAPH_ERROR("target_nodes_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto neighbor_nodes_tensor_description =
    *wholegraph_tensor_get_tensor_description(neighbor_nodes_tensor);
  if (neighbor_nodes_tensor_description.dim != 1) {
    WHOLEGRAPH_ERROR("neighbor_nodes_tensor should be 1D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }

  auto output_neighbor_raw_to_unique_mapping_tensor_description =
    *wholegraph_tensor_get_tensor_description(output_neighbor_raw_to_unique_mapping_tensor);

  if (output_neighbor_raw_to_unique_mapping_tensor_description.dim != 1 &&
      output_neighbor_raw_to_unique_mapping_tensor_description.dim != 0) {
    WHOLEGRAPH_ERROR("output_neighbor_raw_to_unique_mapping_tensor should be 1D tensor or None.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (output_neighbor_raw_to_unique_mapping_tensor_description.dim == 1 &&
      output_neighbor_raw_to_unique_mapping_tensor_description.dtype != WHOLEGRAPH_DT_INT) {
    WHOLEGRAPH_ERROR("output_neighbor_raw_to_unique_mapping_tensor should be int tensor or None.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  wholegraph_array_description_t target_nodes_array_desc, neighbor_nodes_array_desc,
    output_neighbor_raw_to_unique_mapping_array_desc;

  if (!wholegraph_convert_tensor_desc_to_array(&target_nodes_array_desc,
                                                &target_nodes_tensor_description)) {
    WHOLEGRAPH_ERROR("Input target_nodes_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }

  if (!wholegraph_convert_tensor_desc_to_array(&neighbor_nodes_array_desc,
                                                &neighbor_nodes_tensor_description)) {
    WHOLEGRAPH_ERROR("Input neighbor_nodes_tensor convert to array failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }

  if (target_nodes_array_desc.dtype != neighbor_nodes_array_desc.dtype) {
    WHOLEGRAPH_ERROR("target_nodes_dtype should be the same with neighbor_nodes_dtype");
    return WHOLEGRAPH_LOGIC_ERROR;
  }

  void* target_nodes_ptr   = wholegraph_tensor_get_data_pointer(target_nodes_tensor);
  void* neighbor_nodes_ptr = wholegraph_tensor_get_data_pointer(neighbor_nodes_tensor);
  void* output_neighbor_raw_to_unique_mapping_ptr =
    wholegraph_tensor_get_data_pointer(output_neighbor_raw_to_unique_mapping_tensor);

  return graph_ops::graph_append_unique_impl(
    target_nodes_ptr,
    target_nodes_array_desc,
    neighbor_nodes_ptr,
    neighbor_nodes_array_desc,
    output_unique_node_memory_context,
    static_cast<int*>(output_neighbor_raw_to_unique_mapping_ptr),
    p_env_fns,
    static_cast<cudaStream_t>(stream));
}
