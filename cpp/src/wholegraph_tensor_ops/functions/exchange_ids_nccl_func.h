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

/**
 * Bucket and exchange ids using collective communication
 *
 * @param indices : pointer to indices array
 * @param indice_desc : indices array description, should have storage offset = 0, indice can be
 * int32 or int64
 * @param host_recv_rank_id_count_ptr : pointer to int64_t array of received id count from each rank
 * @param host_rank_id_count_ptr : pointer to int64_t array of id count to send to each rank.
 * @param dev_recv_indices_buffer_handle : temp_memory_handle to create buffer for received indices.
 * @param dev_raw_indice_ptr : pointer to allocated int64_t array to storage raw indices mapping of
 * sort
 * @param embedding_entry_offsets : embedding entry offsets
 * @param wg_comm : WholeGraph Communicator
 * @param p_thrust_allocator : thrust allocator
 * @param p_env_fns : EnvFns
 * @param stream : CUDA stream to use.
 * @return : WHOLEGRAPH_SUCCESS on success, others on failure
 */
wholegraph_error_code_t bucket_and_exchange_ids_func(
  void* indices,
  wholegraph_array_description_t indice_desc,
  int64_t* host_recv_rank_id_count_ptr,
  int64_t* host_rank_id_count_ptr,
  temp_memory_handle* dev_recv_indices_buffer_handle,
  int64_t* dev_raw_indice_ptr,
  size_t* embedding_entry_offsets,
  wholegraph_comm_t wg_comm,
  wg_thrust_allocator* p_thrust_allocator,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream);

}  // namespace wholegraph_tensor_ops
