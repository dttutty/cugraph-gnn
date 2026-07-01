/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

#include <wholegraph_tensor_ops/temp_memory_handle.hpp>
#include <wholegraph_tensor_ops/thrust_allocator.hpp>

namespace wholegraph_tensor_ops {

/**
 * Exchange embeddings between ranks
 * @param dev_local_gather_buffer_ptr : local buffer to send
 * @param host_send_to_rank_count_ptr : id count that current rank send to other ranks
 * @param host_recv_from_rank_count_ptr : id count that current rank receive from each rank
 * @param dev_embedding_recv_buffer_ptr : local buffer to receive embedding data
 * @param embedding_size : embedding size in bytes.
 * @param wg_comm : WholeGraph communicator
 * @param stream : CUDA stream to use
 * @return : WHOLEGRAPH_SUCCESS on success, others on failure.
 */
wholegraph_error_code_t exchange_embeddings_nccl_func(const void* dev_local_gather_buffer_ptr,
                                                       const int64_t* host_send_to_rank_count_ptr,
                                                       const int64_t* host_recv_from_rank_count_ptr,
                                                       void* dev_embedding_recv_buffer_ptr,
                                                       size_t embedding_size,
                                                       wholegraph_comm_t wg_comm,
                                                       cudaStream_t stream);

/**
 * Dedup indice and gradients
 * @param indices : indices
 * @param indice_desc : array description of indice
 * @param grads : gradients
 * @param grads_desc : matrix description of gradients
 * @param dedup_indice : output indice
 * @param dedup_grads : output gradients
 * @param p_env_fn : env_fns
 * @param stream : CUDA stream to use
 * @return : deduped indice count
 */
int64_t dedup_indice_and_gradients(const void* indices,
                                   wholegraph_array_description_t indice_desc,
                                   const void* grads,
                                   wholegraph_matrix_description_t grads_desc,
                                   void* dedup_indice,
                                   float* dedup_grads,
                                   wholegraph_env_func_t* p_env_fn,
                                   cudaStream_t stream);

}  // namespace wholegraph_tensor_ops
