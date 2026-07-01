/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

namespace wholegraph_tensor_ops {
namespace testing {

void device_matrix_type_cast(void* dst,
                             wholegraph_matrix_description_t dst_desc,
                             const void* src,
                             wholegraph_matrix_description_t src_desc,
                             cudaStream_t stream);

void device_array_type_cast(void* dst,
                            wholegraph_array_description_t dst_desc,
                            const void* src,
                            wholegraph_array_description_t src_desc,
                            cudaStream_t stream);

void device_random_init_local_embedding_table(wholegraph_handle_t embedding_handle,
                                              wholegraph_matrix_description_t embedding_desc,
                                              cudaStream_t stream);

void device_get_expected_embedding(void* output,
                                   wholegraph_matrix_description_t output_desc,
                                   wholegraph_dtype_t embedding_dtype,
                                   void* indices,
                                   wholegraph_array_description_t indices_desc,
                                   wholegraph_env_func_t* p_env_fns,
                                   cudaStream_t stream);

/**
 * random generate indices from [0, max_indices)
 * @param indices : pointer of output
 * @param indices_desc : description of output
 * @param max_indices : max_indices
 */
void host_random_init_indices(void* indices,
                              wholegraph_array_description_t indices_desc,
                              int64_t max_indices);

void host_check_embedding_same(void* host_embedding,
                               wholegraph_matrix_description_t embedding_desc,
                               void* host_reference,
                               wholegraph_matrix_description_t reference_desc);

void host_random_init_float(float* data, int64_t len, float max_value, float min_value);

void host_random_partition(size_t* partition_sizes, size_t total_size, int partition_count);

}  // namespace testing
}  // namespace wholegraph_tensor_ops
