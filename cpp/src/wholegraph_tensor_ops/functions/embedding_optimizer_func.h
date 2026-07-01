/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>

#include <wholegraph/wholegraph.h>
#include <wholegraph/wholegraph_tensor.h>

namespace wholegraph_tensor_ops {

void set_memory_to_float_value(float* data_ptr, float value, size_t elt_count, cudaStream_t stream);

wholegraph_error_code_t sgd_optimizer_step(wholegraph_tensor_t indices,
                                            wholegraph_tensor_t grads,
                                            wholegraph_tensor_t local_embedding,
                                            wholegraph_tensor_t local_embedding_cache_tag,
                                            wholegraph_tensor_t local_embedding_cache_data,
                                            int64_t local_entry_offset,
                                            int cache_set_coverage,
                                            float weight_decay,
                                            float lr,
                                            cudaStream_t stream);

wholegraph_error_code_t lazy_adam_optimizer_step(wholegraph_tensor_t indices,
                                                  wholegraph_tensor_t grads,
                                                  wholegraph_tensor_t local_embedding,
                                                  wholegraph_tensor_t local_embedding_cache_tag,
                                                  wholegraph_tensor_t local_embedding_cache_data,
                                                  wholegraph_tensor_t per_element_local_state,
                                                  wholegraph_tensor_t per_element_local_cache_tag,
                                                  wholegraph_tensor_t per_element_local_cache_data,
                                                  wholegraph_tensor_t per_embedding_local_state,
                                                  int64_t local_entry_offset,
                                                  int cache_set_coverage,
                                                  float weight_decay,
                                                  float epsilon,
                                                  float beta1,
                                                  float beta2,
                                                  bool adam_w,
                                                  float lr,
                                                  cudaStream_t stream);

wholegraph_error_code_t ada_grad_optimizer_step(wholegraph_tensor_t indices,
                                                 wholegraph_tensor_t grads,
                                                 wholegraph_tensor_t local_embedding,
                                                 wholegraph_tensor_t local_embedding_cache_tag,
                                                 wholegraph_tensor_t local_embedding_cache_data,
                                                 wholegraph_tensor_t per_element_local_state,
                                                 wholegraph_tensor_t per_element_local_cache_tag,
                                                 wholegraph_tensor_t per_element_local_cache_data,
                                                 int64_t local_entry_offset,
                                                 int cache_set_coverage,
                                                 float weight_decay,
                                                 float epsilon,
                                                 float lr,
                                                 cudaStream_t stream);

wholegraph_error_code_t rms_prop_optimizer_step(wholegraph_tensor_t indices,
                                                 wholegraph_tensor_t grads,
                                                 wholegraph_tensor_t local_embedding,
                                                 wholegraph_tensor_t local_embedding_cache_tag,
                                                 wholegraph_tensor_t local_embedding_cache_data,
                                                 wholegraph_tensor_t per_element_local_state,
                                                 wholegraph_tensor_t per_element_local_cache_tag,
                                                 wholegraph_tensor_t per_element_local_cache_data,
                                                 int64_t local_entry_offset,
                                                 int cache_set_coverage,
                                                 float weight_decay,
                                                 float epsilon,
                                                 float alpha,
                                                 float lr,
                                                 cudaStream_t stream);

}  // namespace wholegraph_tensor_ops
