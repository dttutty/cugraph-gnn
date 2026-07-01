/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph.h>
#include <wholegraph/wholegraph_tensor.h>

#include "wholegraph/embedding_cache.hpp"

namespace wholegraph_tensor_ops {

/**
 * Direct update cache in local rank, local tensor of wg_raw_memory_embedding is cached by
 * cache_local_data, cache and raw embedding should have same communicator.
 * @param indices : global indices to update, should all in current rank, can have duplicated gids
 * In normal use cases, indices are from alltoallv result
 * @param indice_desc : tensor description of indices, may be gids after alltoallv.
 * @param wg_raw_memory_embedding : the WholeGraph Tensor that is to be cached which stores all
 * embeddings.
 * @param cache_local_data : embedding_cache_local_data of wg_raw_memory_embedding
 * @param cache_set_coverage : cache set coverage
 * @param p_env_fns : env fns
 * @param stream : cudaStream to use
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t update_cache_direct_same_comm(
  void* indices,
  wholegraph_array_description_t indice_desc,
  wholegraph_tensor_t wg_raw_memory_embedding,
  const wholegraph::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream);

/**
 * Update cache in local rank, local tensor of wg_raw_memory_embedding is cached by
 * cache_local_data, cache and raw embedding can have same communicator.
 * @param indices : global indices to update, should all in current rank, can have duplicated gids
 * In normal use cases, indices are from alltoallv result
 * @param indice_desc : tensor description of indices, may be gids after alltoallv.
 * @param wg_raw_memory_embedding : the WholeGraph Tensor that is to be cached which stores all
 * embeddings.
 * @param cache_comm : communicator of cache
 * @param embedding_entry_offsets : embedding entry offset of each cache rank
 * @param cache_local_data : embedding_cache_local_data of wg_raw_memory_embedding
 * @param cache_set_coverage : cache set coverage
 * @param p_env_fns : env fns
 * @param stream : cudaStream to use
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t update_cache_different_comm(
  void* indices,
  wholegraph_array_description_t indice_desc,
  wholegraph_tensor_t wg_raw_memory_embedding,
  wholegraph_comm_t cache_comm,
  size_t* embedding_entry_offsets,
  const wholegraph::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream);

wholegraph_error_code_t writeback_cache_direct_same_comm(
  wholegraph_tensor_t wg_raw_memory_embedding,
  const wholegraph::embedding_cache_local_data* cache_local_data,
  int cache_set_coverage,
  bool drop_all,
  cudaStream_t stream);

}  // namespace wholegraph_tensor_ops
