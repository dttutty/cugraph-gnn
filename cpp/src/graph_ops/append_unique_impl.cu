/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include "append_unique_func.cuh"
#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph.h>

#include "wholegraph_tensor_ops/register.hpp"

namespace graph_ops {

REGISTER_DISPATCH_ONE_TYPE(GraphAppendUnique, graph_append_unique_func, SINT3264)
wholegraph_error_code_t graph_append_unique_impl(
  void* target_nodes_ptr,
  wholegraph_array_description_t target_nodes_desc,
  void* neighbor_nodes_ptr,
  wholegraph_array_description_t neighbor_nodes_desc,
  void* output_unique_node_memory_context,
  int* output_neighbor_raw_to_unique_mapping_ptr,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(target_nodes_desc.dtype,
                      GraphAppendUnique,
                      target_nodes_ptr,
                      target_nodes_desc,
                      neighbor_nodes_ptr,
                      neighbor_nodes_desc,
                      output_unique_node_memory_context,
                      output_neighbor_raw_to_unique_mapping_ptr,
                      p_env_fns,
                      stream);

  } catch (const wholegraph::cuda_error& rle) {
    // WHOLEGRAPH_FAIL_NOTHROW("%s", rle.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (const wholegraph::logic_error& le) {
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (...) {
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  return WHOLEGRAPH_SUCCESS;
}

}  // namespace graph_ops
