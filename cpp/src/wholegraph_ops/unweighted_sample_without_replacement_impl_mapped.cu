/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph.h>

#include "unweighted_sample_without_replacement_func.cuh"
#include "wholegraph_tensor_ops/register.hpp"

namespace wholegraph_ops {

REGISTER_DISPATCH_TWO_TYPES(UnweightedSampleWithoutReplacementCSR,
                            wholegraph_csr_unweighted_sample_without_replacement_func,
                            SINT3264,
                            SINT3264)

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
  cudaStream_t stream)
{
  try {
    DISPATCH_TWO_TYPES(center_nodes_desc.dtype,
                       wg_csr_col_ptr_desc.dtype,
                       UnweightedSampleWithoutReplacementCSR,
                       wg_csr_row_ptr,
                       wg_csr_row_ptr_desc,
                       wg_csr_col_ptr,
                       wg_csr_col_ptr_desc,
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

}  // namespace wholegraph_ops
