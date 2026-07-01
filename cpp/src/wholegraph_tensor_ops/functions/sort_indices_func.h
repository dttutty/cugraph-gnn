/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

#include <wholegraph_tensor_ops/temp_memory_handle.hpp>
#include <wholegraph_tensor_ops/thrust_allocator.hpp>

namespace wholegraph_tensor_ops {

wholegraph_error_code_t sort_indices_func(const void* indices_before_sort,
                                           wholegraph_array_description_t indice_desc,
                                           void* indices_after_sort,
                                           void* raw_indices,
                                           wg_thrust_allocator* p_thrust_allocator,
                                           wholegraph_env_func_t* p_env_fns,
                                           cudaStream_t stream);

}  // namespace wholegraph_tensor_ops
