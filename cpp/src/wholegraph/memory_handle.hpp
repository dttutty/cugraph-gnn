/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/wholegraph.h>

#include <cuda_runtime_api.h>

#include "communicator.hpp"

namespace wholegraph {

class wholegraph_impl;

}

struct wholegraph_handle_ {
  int handle_id;
  wholegraph::wholegraph_impl* impl = nullptr;
  ~wholegraph_handle_();
};

namespace wholegraph {

wholegraph_error_code_t create_wholegraph(wholegraph_handle_t* wholegraph_handle_ptr,
                                            size_t total_size,
                                            wholegraph_comm_t comm,
                                            wholegraph_memory_type_t memory_type,
                                            wholegraph_memory_location_t memory_location,
                                            size_t data_granularity,
                                            size_t* rank_entry_partition = nullptr) noexcept;

wholegraph_error_code_t destroy_wholegraph_with_comm_locked(
  wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t destroy_wholegraph(wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_communicator_from_handle(
  wholegraph_comm_t* comm, wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_memory_type_t get_memory_type(wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_memory_location_t get_memory_location(wholegraph_handle_t wholegraph_handle) noexcept;

size_t get_total_size(wholegraph_handle_t wholegraph_handle) noexcept;

size_t get_data_granularity(wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_local_memory_from_handle(
  void** local_ptr,
  size_t* local_size,
  size_t* local_offset,
  wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_local_node_memory_from_handle(
  void** local_ptr,
  size_t* local_size,
  size_t* local_offset,
  wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_rank_memory_from_handle(
  void** rank_memory_ptr,
  size_t* rank_memory_size,
  size_t* rank_memory_offset,
  int rank,
  wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_local_size_from_handle(
  size_t* size, wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_local_offset_from_handle(
  size_t* offset, wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_global_pointer_from_handle(
  void** global_ptr, wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_global_reference_from_handle(
  wholegraph_gref_t* wholegraph_gref, wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t equal_partition_plan(size_t* entry_per_rank,
                                              size_t total_entry_count,
                                              int world_size) noexcept;

wholegraph_error_code_t get_rank_partition_sizes_from_handle(
  size_t* rank_sizes, wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_error_code_t get_rank_partition_offsets_from_handle(
  size_t* rank_offsets, wholegraph_handle_t wholegraph_handle) noexcept;

wholegraph_distributed_backend_t get_distributed_backend_t(
  wholegraph_handle_t wholegraph_handle) noexcept;

}  // namespace wholegraph
