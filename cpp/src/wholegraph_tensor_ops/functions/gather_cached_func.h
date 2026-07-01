/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

namespace wholegraph_tensor_ops {

wholegraph_error_code_t gather_cached_func(wholegraph_gref_t padded_embedding_gref,
                                            wholegraph_tensor_description_t* embedding_desc,
                                            wholegraph_gref_t cached_embedding_gref,
                                            wholegraph_tensor_description_t* cached_embedding_desc,
                                            wholegraph_gref_t cache_line_tag_gref,
                                            void* indices,
                                            wholegraph_tensor_description_t* indices_desc,
                                            void* output,
                                            wholegraph_tensor_description_t* output_desc,
                                            int cache_set_coverage,
                                            int64_t cache_start_gid,
                                            int64_t raw_start_gid,
                                            cudaStream_t stream);

wholegraph_error_code_t try_gather_cached_func(
  wholegraph_gref_t cached_embedding_gref,
  wholegraph_tensor_description_t* cached_embedding_desc,
  wholegraph_gref_t cache_line_tag_gref,
  void* indices,
  wholegraph_tensor_description_t* indices_desc,
  void* hit_indices,
  void* miss_indices,
  void* output,
  wholegraph_tensor_description_t* output_desc,
  int cache_set_coverage,
  int64_t cache_start_gid,
  cudaStream_t stream);

}  // namespace wholegraph_tensor_ops
