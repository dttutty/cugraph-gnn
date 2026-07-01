/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholegraph/wholegraph.h>

#include "communicator.hpp"
#include "file_io.h"
#include "initialize.hpp"
#include "memory_handle.hpp"
#include "parallel_utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif

wholegraph_error_code_t wholegraph_init(unsigned int flags, LogLevel log_level)
{
  return wholegraph::init(flags, log_level);
}

wholegraph_error_code_t wholegraph_finalize() { return wholegraph::finalize(); }

wholegraph_error_code_t wholegraph_create_unique_id(wholegraph_unique_id_t* unique_id)
{
  return wholegraph::create_unique_id(unique_id);
}

wholegraph_error_code_t wholegraph_create_communicator(wholegraph_comm_t* comm,
                                                         wholegraph_unique_id_t unique_id,
                                                         int rank,
                                                         int size)
{
  return wholegraph::create_communicator(comm, unique_id, rank, size);
}

wholegraph_error_code_t wholegraph_split_communicator(wholegraph_comm_t* new_comm,
                                                        wholegraph_comm_t comm,
                                                        int color,
                                                        int key)
{
  return wholegraph::split_communicator(new_comm, comm, color, key);
}

wholegraph_error_code_t wholegraph_destroy_communicator(wholegraph_comm_t comm)
{
  return wholegraph::destroy_communicator(comm);
}

wholegraph_error_code_t wholegraph_communicator_support_type_location(
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location)
{
  return wholegraph::communicator_support_type_location(comm, memory_type, memory_location);
}

wholegraph_error_code_t wholegraph_communicator_get_rank(int* rank, wholegraph_comm_t comm)
{
  return wholegraph::communicator_get_rank(rank, comm);
}

wholegraph_error_code_t wholegraph_communicator_get_size(int* size, wholegraph_comm_t comm)
{
  return wholegraph::communicator_get_size(size, comm);
}

wholegraph_error_code_t wholegraph_communicator_get_local_size(int* local_size,
                                                                 wholegraph_comm_t comm)
{
  return wholegraph::communicator_get_local_size(local_size, comm);
}

wholegraph_error_code_t wholegraph_communicator_set_distributed_backend(
  wholegraph_comm_t comm, wholegraph_distributed_backend_t distributed_backend)
{
  return wholegraph::communicator_set_distributed_backend(comm, distributed_backend);
}

wholegraph_distributed_backend_t wholegraph_communicator_get_distributed_backend(
  wholegraph_comm_t comm)
{
  return wholegraph::communicator_get_distributed_backend(comm);
}

wholegraph_error_code_t wholegraph_communicator_barrier(wholegraph_comm_t comm)
{
  wholegraph::communicator_barrier(comm);
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_malloc(wholegraph_handle_t* wholegraph_handle_ptr,
                                            size_t total_size,
                                            wholegraph_comm_t comm,
                                            wholegraph_memory_type_t memory_type,
                                            wholegraph_memory_location_t memory_location,
                                            size_t data_granularity,
                                            size_t* rank_entry_partition)
{
  return wholegraph::create_wholegraph(wholegraph_handle_ptr,
                                         total_size,
                                         comm,
                                         memory_type,
                                         memory_location,
                                         data_granularity,
                                         rank_entry_partition);
}

wholegraph_error_code_t wholegraph_free(wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::destroy_wholegraph(wholegraph_handle);
}

wholegraph_error_code_t wholegraph_get_communicator(wholegraph_comm_t* comm,
                                                      wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_communicator_from_handle(comm, wholegraph_handle);
}

wholegraph_memory_type_t wholegraph_get_memory_type(wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_memory_type(wholegraph_handle);
}

wholegraph_memory_location_t wholegraph_get_memory_location(
  wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_memory_location(wholegraph_handle);
}

wholegraph_distributed_backend_t wholegraph_get_distributed_backend(
  wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_distributed_backend_t(wholegraph_handle);
}

size_t wholegraph_get_total_size(wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_total_size(wholegraph_handle);
}

size_t wholegraph_get_data_granularity(wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_data_granularity(wholegraph_handle);
}

wholegraph_error_code_t wholegraph_get_local_memory(void** local_ptr,
                                                      size_t* local_size,
                                                      size_t* local_offset,
                                                      wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_local_memory_from_handle(
    local_ptr, local_size, local_offset, wholegraph_handle);
}

wholegraph_error_code_t wholegraph_get_rank_memory(void** rank_memory_ptr,
                                                     size_t* rank_memory_size,
                                                     size_t* rank_memory_offset,
                                                     int rank,
                                                     wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_rank_memory_from_handle(
    rank_memory_ptr, rank_memory_size, rank_memory_offset, rank, wholegraph_handle);
}

wholegraph_error_code_t wholegraph_equal_entry_partition_plan(size_t* entry_per_rank,
                                                                size_t total_entry_count,
                                                                int world_size)
{
  return wholegraph::equal_partition_plan(entry_per_rank, total_entry_count, world_size);
}

wholegraph_error_code_t wholegraph_get_global_pointer(void** global_ptr,
                                                        wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_global_pointer_from_handle(global_ptr, wholegraph_handle);
}

wholegraph_error_code_t wholegraph_get_global_reference(wholegraph_gref_t* wholegraph_gref,
                                                          wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_global_reference_from_handle(wholegraph_gref, wholegraph_handle);
}

wholegraph_error_code_t wholegraph_get_rank_partition_sizes(
  size_t* rank_sizes, wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_rank_partition_sizes_from_handle(rank_sizes, wholegraph_handle);
}

wholegraph_error_code_t wholegraph_get_rank_partition_offsets(
  size_t* rank_offsets, wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_rank_partition_offsets_from_handle(rank_offsets, wholegraph_handle);
}

wholegraph_error_code_t wholegraph_get_local_size(size_t* local_size,
                                                    wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_local_size_from_handle(local_size, wholegraph_handle);
}

wholegraph_error_code_t wholegraph_get_local_offset(size_t* local_size,
                                                      wholegraph_handle_t wholegraph_handle)
{
  return wholegraph::get_local_offset_from_handle(local_size, wholegraph_handle);
}

int fork_get_device_count()
{
  try {
    return ForkGetDeviceCount();
  } catch (...) {
    WHOLEGRAPH_ERROR("fork_get_device_count failed.");
    return -1;
  }
}

wholegraph_error_code_t wholegraph_load_from_file(wholegraph_handle_t wholegraph_handle,
                                                    size_t memory_offset,
                                                    size_t memory_entry_size,
                                                    size_t file_entry_size,
                                                    const char** file_names,
                                                    int file_count,
                                                    int round_robin_size)
{
  return wholegraph::load_file_to_handle(wholegraph_handle,
                                          memory_offset,
                                          memory_entry_size,
                                          file_entry_size,
                                          file_names,
                                          file_count,
                                          round_robin_size);
}

wholegraph_error_code_t wholegraph_store_to_file(wholegraph_handle_t wholegraph_handle,
                                                   size_t memory_offset,
                                                   size_t memory_entry_stride,
                                                   size_t file_entry_size,
                                                   const char* local_file_name)
{
  return wholegraph::store_handle_to_file(
    wholegraph_handle, memory_offset, memory_entry_stride, file_entry_size, local_file_name);
}

wholegraph_error_code_t wholegraph_load_hdfs_support() { return WHOLEGRAPH_NOT_IMPLEMENTED; }

wholegraph_error_code_t wholegraph_load_from_hdfs_file(wholegraph_handle_t wholegraph_handle,
                                                         size_t memory_offset,
                                                         size_t memory_entry_size,
                                                         size_t file_entry_size,
                                                         const char* hdfs_host,
                                                         int hdfs_port,
                                                         const char* hdfs_user,
                                                         const char* hdfs_path,
                                                         const char* hdfs_prefix)
{
  return WHOLEGRAPH_NOT_IMPLEMENTED;
}

bool wholegraph_is_intranode_communicator(wholegraph_comm_t comm)
{
  return wholegraph::is_intranode_communicator(comm);
}

bool wholegraph_is_intra_mnnvl_communicator(wholegraph_comm_t comm)
{
  return wholegraph::is_intra_mnnvl_communicator(comm);
}

wholegraph_error_code_t wholegraph_communicator_get_clique_info(clique_info_t* clique_info,
                                                                  wholegraph_comm_t comm)
{
  return wholegraph::communicator_get_clique_info(clique_info, comm);
}

#ifdef __cplusplus
}
#endif
