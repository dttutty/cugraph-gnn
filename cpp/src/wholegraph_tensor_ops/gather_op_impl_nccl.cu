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
#include "wholegraph_tensor_ops/gather_op_impl.h"
#include "wholegraph_tensor_ops/temp_memory_handle.hpp"
#include "wholegraph_tensor_ops/thrust_allocator.hpp"

namespace wholegraph_tensor_ops {

wholegraph_error_code_t wholegraph_gather_nccl(wholegraph_handle_t wholegraph_handle,
                                                 wholegraph_matrix_description_t wholegraph_desc,
                                                 void* indices,
                                                 wholegraph_array_description_t indice_desc,
                                                 void* output,
                                                 wholegraph_matrix_description_t output_desc,
                                                 wholegraph_env_func_t* p_env_fns,
                                                 cudaStream_t stream,
                                                 int gather_sms)
{
  try {
    if (wholegraph_desc.storage_offset < 0 ||
        wholegraph_desc.storage_offset + wholegraph_desc.sizes[1] > wholegraph_desc.stride) {
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
      static_cast<int64_t*>(dev_raw_indice.device_malloc(indice_desc.size, WHOLEGRAPH_DT_INT64));

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
                                                            indice_desc,
                                                            host_recv_rank_id_count_ptr,
                                                            host_rank_id_count_ptr,
                                                            &dev_recv_indice_buffer,
                                                            dev_raw_indice_ptr,
                                                            dev_embedding_entry_offsets_ptr,
                                                            wg_comm,
                                                            &thrust_allocator,
                                                            p_env_fns,
                                                            stream));
    // Local Gather
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    size_t local_mem_offset, local_mem_size;
    temp_memory_handle dev_local_gather_buffer(p_env_fns);
    temp_memory_handle dev_embedding_recv_buffer(p_env_fns);
    void* dev_local_gather_buffer_ptr = dev_local_gather_buffer.device_malloc(
      wholegraph_desc.sizes[1] * total_recv_count, output_desc.dtype);
    void* dev_embedding_recv_buffer_ptr = dev_embedding_recv_buffer.device_malloc(
      wholegraph_desc.sizes[1] * indice_desc.size, output_desc.dtype);
    void* local_fake_ptr = nullptr;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_local_memory(
      &local_fake_ptr, &local_mem_size, &local_mem_offset, wholegraph_handle));
    local_fake_ptr = static_cast<char*>(local_fake_ptr) - local_mem_offset;
    wholegraph_gref_t local_fake_gref =
      wholegraph_create_continuous_global_reference(local_fake_ptr);
    int64_t local_buffer_size[2] = {total_recv_count, wholegraph_desc.sizes[1]};
    wholegraph_matrix_description_t local_gather_buffer_desc = wholegraph_create_matrix_desc(
      local_buffer_size, wholegraph_desc.sizes[1], 0, output_desc.dtype);
    auto dev_recv_indice_desc =
      wholegraph_create_array_desc(total_recv_count, 0, indice_desc.dtype);
    WHOLEGRAPH_RETURN_ON_FAIL(gather_func(local_fake_gref,
                                           wholegraph_desc,
                                           dev_recv_indice_buffer.pointer(),
                                           dev_recv_indice_desc,
                                           dev_local_gather_buffer_ptr,
                                           local_gather_buffer_desc,
                                           stream,
                                           gather_sms));
    // AllToAllV for embeddings
    size_t embedding_size =
      wholegraph_desc.sizes[1] * wholegraph_dtype_get_element_size(output_desc.dtype);
    WHOLEGRAPH_RETURN_ON_FAIL(exchange_embeddings_nccl_func(dev_local_gather_buffer_ptr,
                                                             host_recv_rank_id_count_ptr,
                                                             host_rank_id_count_ptr,
                                                             dev_embedding_recv_buffer_ptr,
                                                             embedding_size,
                                                             wg_comm,
                                                             stream));
    // Local reorder
    int64_t total_need_indice_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_need_indice_count += host_rank_id_count_ptr[i];
    }
    wholegraph_gref_t output_gref = wholegraph_create_continuous_global_reference(output);
    wholegraph_matrix_description_t local_recv_buffer_desc =
      wholegraph_create_matrix_desc(output_desc.sizes, output_desc.sizes[1], 0, output_desc.dtype);
    local_recv_buffer_desc.sizes[0] = total_need_indice_count;
    auto raw_indice_desc =
      wholegraph_create_array_desc(total_need_indice_count, 0, WHOLEGRAPH_DT_INT64);
    WHOLEGRAPH_RETURN_ON_FAIL(scatter_func(dev_embedding_recv_buffer_ptr,
                                            local_recv_buffer_desc,
                                            dev_raw_indice_ptr,
                                            raw_indice_desc,
                                            output_gref,
                                            output_desc,
                                            stream));
    WG_CUDA_CHECK(cudaGetLastError());
    // WG_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (wholegraph::cuda_error& wce) {
    WHOLEGRAPH_ERROR("CUDA logic Error %s\n", wce.what());
    return WHOLEGRAPH_CUDA_ERROR;
  } catch (wholegraph::logic_error& wle) {
    WHOLEGRAPH_ERROR("LOGIC Error %s\n", wle.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (...) {
    return WHOLEGRAPH_UNKNOW_ERROR;
  }

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_gather_distributed(
  wholegraph_handle_t wholegraph_handle,
  wholegraph_matrix_description_t wholegraph_desc,
  void* indices,
  wholegraph_array_description_t indice_desc,
  void* output,
  wholegraph_matrix_description_t output_desc,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms)
{
  return wholegraph_gather_nccl(wholegraph_handle,
                                 wholegraph_desc,
                                 indices,
                                 indice_desc,
                                 output,
                                 output_desc,
                                 p_env_fns,
                                 stream,
                                 gather_sms);
}
}  // namespace wholegraph_tensor_ops
