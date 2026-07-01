/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph.h>

#include "logger.hpp"
#include "wholegraph/communicator.hpp"
#include "wholegraph/memory_handle.hpp"
#include "wholegraph_tensor_ops/functions/bucket_ids_func.h"
#include "wholegraph_tensor_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholegraph_tensor_ops/functions/exchange_ids_nccl_func.h"
#include "wholegraph_tensor_ops/functions/gather_scatter_func.h"
#include "wholegraph_tensor_ops/scatter_op_impl.h"
#include "wholegraph_tensor_ops/temp_memory_handle.hpp"
#include "wholegraph_tensor_ops/thrust_allocator.hpp"

namespace wholegraph_tensor_ops {

wholegraph_error_code_t wholegraph_scatter_nccl(void* input,
                                                  wholegraph_matrix_description_t input_desc,
                                                  void* indices,
                                                  wholegraph_array_description_t indices_desc,
                                                  wholegraph_handle_t wholegraph_handle,
                                                  wholegraph_matrix_description_t wholegraph_desc,
                                                  wholegraph_env_func_t* p_env_fns,
                                                  cudaStream_t stream,
                                                  int scatter_sms)
{
  try {
    if (wholegraph_desc.storage_offset < 0 ||
        wholegraph_desc.storage_offset + wholegraph_desc.sizes[1] > wholegraph_desc.stride) {
      WHOLEGRAPH_ERROR("invalid input offset=%ld, size[1]=%ld, stride=%ld\n",
                        wholegraph_desc.storage_offset,
                        wholegraph_desc.sizes[1],
                        wholegraph_desc.stride);
      return WHOLEGRAPH_INVALID_INPUT;
    }

    wg_thrust_allocator thrust_allocator(p_env_fns);

    size_t element_size         = wholegraph_dtype_get_element_size(wholegraph_desc.dtype);
    size_t embedding_entry_size = element_size * wholegraph_desc.stride;

    wholegraph_comm_t wg_comm;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(&wg_comm, wholegraph_handle));

    int world_size;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&world_size, wg_comm));

    temp_memory_handle host_rank_id_count(p_env_fns), host_recv_rank_id_count(p_env_fns);
    int64_t* host_rank_id_count_ptr =
      static_cast<int64_t*>(host_rank_id_count.host_malloc(world_size, WHOLEGRAPH_DT_INT64));
    int64_t* host_recv_rank_id_count_ptr =
      static_cast<int64_t*>(host_recv_rank_id_count.host_malloc(world_size, WHOLEGRAPH_DT_INT64));

    temp_memory_handle dev_recv_indice_buffer(p_env_fns);
    temp_memory_handle dev_raw_indice(p_env_fns);
    int64_t* dev_raw_indice_ptr =
      static_cast<int64_t*>(dev_raw_indice.device_malloc(indices_desc.size, WHOLEGRAPH_DT_INT64));

    int64_t total_recv_count = 0;

    temp_memory_handle dev_embedding_entry_offsets_handle(p_env_fns);
    size_t* dev_embedding_entry_offsets_ptr = static_cast<size_t*>(
      dev_embedding_entry_offsets_handle.device_malloc(world_size + 1, WHOLEGRAPH_DT_INT64));
    temp_memory_handle host_embedding_entry_offsets_handle(p_env_fns);
    size_t* host_embedding_entry_offsets_ptr = static_cast<size_t*>(
      host_embedding_entry_offsets_handle.host_malloc(world_size + 1, WHOLEGRAPH_DT_INT64));

    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_get_rank_partition_offsets(host_embedding_entry_offsets_ptr, wholegraph_handle));
    for (int i = 0; i < world_size + 1; i++) {
      size_t offset = host_embedding_entry_offsets_ptr[i];
      WHOLEGRAPH_EXPECTS_NOTHROW(
        offset % embedding_entry_size == 0,
        "embedding memory offset of rank%d=%ld is not multiple of embedding_entry_size=%ldx%ld",
        i,
        offset,
        element_size,
        wholegraph_desc.stride);
      host_embedding_entry_offsets_ptr[i] /= embedding_entry_size;
    }
    WG_CUDA_CHECK(cudaMemcpyAsync(dev_embedding_entry_offsets_ptr,
                                  host_embedding_entry_offsets_ptr,
                                  (world_size + 1) * sizeof(size_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    WHOLEGRAPH_RETURN_ON_FAIL(bucket_and_exchange_ids_func(indices,
                                                            indices_desc,
                                                            host_recv_rank_id_count_ptr,
                                                            host_rank_id_count_ptr,
                                                            &dev_recv_indice_buffer,
                                                            dev_raw_indice_ptr,
                                                            dev_embedding_entry_offsets_ptr,
                                                            wg_comm,
                                                            &thrust_allocator,
                                                            p_env_fns,
                                                            stream));

    // Local Reorder
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    temp_memory_handle dev_local_reorder_buffer(p_env_fns), dev_embedding_recv_buffer(p_env_fns);
    auto local_reorder_desc =
      wholegraph_create_matrix_desc(input_desc.sizes, input_desc.sizes[1], 0, input_desc.dtype);
    void* dev_local_reorder_buffer_ptr = dev_local_reorder_buffer.device_malloc(
      wholegraph_get_memory_element_count_from_matrix(&local_reorder_desc), input_desc.dtype);
    wholegraph_gref_t input_gref = wholegraph_create_continuous_global_reference(input);
    auto dev_raw_indice_desc =
      wholegraph_create_array_desc(indices_desc.size, 0, WHOLEGRAPH_DT_INT64);
    WHOLEGRAPH_RETURN_ON_FAIL(gather_func(input_gref,
                                           input_desc,
                                           dev_raw_indice_ptr,
                                           dev_raw_indice_desc,
                                           dev_local_reorder_buffer_ptr,
                                           local_reorder_desc,
                                           stream));
    // AllToAllV for embeddings
    void* dev_embedding_recv_buffer_ptr = dev_embedding_recv_buffer.device_malloc(
      total_recv_count * input_desc.sizes[1], input_desc.dtype);
    size_t embedding_size =
      wholegraph_desc.sizes[1] * wholegraph_dtype_get_element_size(input_desc.dtype);
    WHOLEGRAPH_RETURN_ON_FAIL(exchange_embeddings_nccl_func(dev_local_reorder_buffer_ptr,
                                                             host_rank_id_count_ptr,
                                                             host_recv_rank_id_count_ptr,
                                                             dev_embedding_recv_buffer_ptr,
                                                             embedding_size,
                                                             wg_comm,
                                                             stream));
    // Local scatter
    size_t local_mem_offset, local_mem_size;
    void* local_fake_ptr = nullptr;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_local_memory(
      &local_fake_ptr, &local_mem_size, &local_mem_offset, wholegraph_handle));
    local_fake_ptr = static_cast<char*>(local_fake_ptr) - local_mem_offset;
    wholegraph_gref_t local_fake_embedding_gref =
      wholegraph_create_continuous_global_reference(local_fake_ptr);

    std::vector<int64_t> recv_embedding_sizes            = {total_recv_count, input_desc.sizes[1]};
    wholegraph_matrix_description_t recv_embedding_desc = wholegraph_create_matrix_desc(
      recv_embedding_sizes.data(), input_desc.sizes[1], 0, input_desc.dtype);
    auto recv_indices_desc = wholegraph_create_array_desc(total_recv_count, 0, indices_desc.dtype);
    WHOLEGRAPH_RETURN_ON_FAIL(scatter_func(dev_embedding_recv_buffer_ptr,
                                            recv_embedding_desc,
                                            dev_recv_indice_buffer.pointer(),
                                            recv_indices_desc,
                                            local_fake_embedding_gref,
                                            wholegraph_desc,
                                            stream,
                                            scatter_sms));
    WG_CUDA_CHECK(cudaGetLastError());
    WG_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (wholegraph::cuda_error& wce) {
    WHOLEGRAPH_ERROR("CUDA logic Error %s\n", wce.what());
    return WHOLEGRAPH_CUDA_ERROR;
  } catch (wholegraph::logic_error& wle) {
    WHOLEGRAPH_ERROR("LOGIC Error %s\n", wle.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (...) {
    WHOLEGRAPH_ERROR("Unknown Error\n");
    return WHOLEGRAPH_UNKNOW_ERROR;
  }

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_scatter_distributed(
  void* input,
  wholegraph_matrix_description_t input_desc,
  void* indices,
  wholegraph_array_description_t indices_desc,
  wholegraph_handle_t wholegraph_handle,
  wholegraph_matrix_description_t wholegraph_desc,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms)
{
  return wholegraph_scatter_nccl(input,
                                  input_desc,
                                  indices,
                                  indices_desc,
                                  wholegraph_handle,
                                  wholegraph_desc,
                                  p_env_fns,
                                  stream,
                                  scatter_sms);
}
}  // namespace wholegraph_tensor_ops
