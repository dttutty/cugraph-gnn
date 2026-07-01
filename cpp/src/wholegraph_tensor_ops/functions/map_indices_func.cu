/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "map_indices_func.h"
#include "wholegraph/integer_utils.hpp"
#include "wholegraph_tensor_ops/register.hpp"
#include <cuda_runtime.h>

namespace wholegraph_tensor_ops {

template <typename IndexT>
__global__ void storage_idx2wg_emb_idx_kernel(IndexT* indice,
                                              IndexT* mapped_indice,
                                              int64_t indice_size,
                                              int world_size,
                                              int64_t entry_start,
                                              int round_robin_size)
{
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = tid; i < indice_size; i += (blockDim.x * gridDim.x)) {
    IndexT target_idx  = indice[i];
    IndexT table_idx   = target_idx / round_robin_size;
    IndexT table_off   = target_idx % round_robin_size;
    int rank_id        = table_idx % world_size;
    int rank_table_idx = table_idx / world_size;
    IndexT wgidx       = entry_start + round_robin_size * rank_table_idx + table_off;
    mapped_indice[i]   = wgidx;
  }
  return;
}

template <typename IndexT>
void storage_idx2wg_emb_idx_temp_fn(void* indice_ptr,
                                    void* mapped_indice_ptr,
                                    int64_t indice_size,
                                    int world_size,
                                    int64_t entry_start,
                                    int round_robin_size,
                                    cudaStream_t stream)
{
  int block_size    = 256;
  int64_t block_num = (indice_size + block_size - 1) / block_size;
  if (block_num > 1568) block_num = 1568;
  IndexT* indice        = static_cast<IndexT*>(indice_ptr);
  IndexT* mapped_indice = static_cast<IndexT*>(mapped_indice_ptr);
  storage_idx2wg_emb_idx_kernel<<<block_num, block_size, 0, stream>>>(
    indice, mapped_indice, indice_size, world_size, entry_start, round_robin_size);
  WG_CUDA_CHECK(cudaStreamSynchronize(stream));
  return;
}

REGISTER_DISPATCH_ONE_TYPE(storageidx2wgembidx, storage_idx2wg_emb_idx_temp_fn, SINT3264)

wholegraph_error_code_t storage_index2wg_embedding_index(wholegraph_tensor_t indices,
                                                          wholegraph_tensor_t mapped_indices,
                                                          wholegraph_tensor_t allocated_embedding,
                                                          int round_robin_size,
                                                          int64_t stream_int)
{
  if (round_robin_size == 0) return WHOLEGRAPH_SUCCESS;
  try {
    auto* indice_desc       = wholegraph_tensor_get_tensor_description(indices);
    void* indice_ptr        = wholegraph_tensor_get_data_pointer(indices);
    void* mapped_indice_ptr = wholegraph_tensor_get_data_pointer(mapped_indices);
    int64_t indice_size     = indice_desc->sizes[0];

    wholegraph_comm_t wg_comm;
    int world_size;
    auto* handle = wholegraph_tensor_get_memory_handle(allocated_embedding);
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(&wg_comm, handle));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&world_size, wg_comm));

    size_t entry_start = 0;
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_get_local_entry_start(&entry_start, allocated_embedding));
    DISPATCH_ONE_TYPE(indice_desc->dtype,
                      storageidx2wgembidx,
                      indice_ptr,
                      mapped_indice_ptr,
                      indice_size,
                      world_size,
                      entry_start,
                      round_robin_size,
                      (cudaStream_t)stream_int);
    WG_CUDA_CHECK(cudaGetLastError());
  } catch (wholegraph::cuda_error& wce) {
    WHOLEGRAPH_ERROR("index map CUDA LOGIC Error %s\n", wce.what());
    return WHOLEGRAPH_CUDA_ERROR;
  }
  return WHOLEGRAPH_SUCCESS;
}

}  // namespace wholegraph_tensor_ops
