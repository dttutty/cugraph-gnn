/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/wholegraph.h>

namespace wholegraph {

wholegraph_error_code_t load_file_to_handle(wholegraph_handle_t wholegraph_handle,
                                             size_t memory_offset,
                                             size_t memory_entry_stride,
                                             size_t entry_size,
                                             const char** file_names,
                                             int file_count,
                                             int round_robin_size) noexcept;

wholegraph_error_code_t store_handle_to_file(wholegraph_handle_t wholegraph_handle,
                                              size_t memory_offset,
                                              size_t memory_entry_stride,
                                              size_t entry_size,
                                              const char* local_file_name) noexcept;

}  // namespace wholegraph
