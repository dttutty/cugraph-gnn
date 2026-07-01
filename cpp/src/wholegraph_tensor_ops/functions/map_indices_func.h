/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>
#include <wholegraph/wholegraph_tensor.h>

namespace wholegraph_tensor_ops {

wholegraph_error_code_t storage_index2wg_embedding_index(wholegraph_tensor_t indices,
                                                          wholegraph_tensor_t mapped_indices,
                                                          wholegraph_tensor_t allocated_embedding,
                                                          int round_robin_size,
                                                          int64_t stream_int);

}  // namespace wholegraph_tensor_ops
