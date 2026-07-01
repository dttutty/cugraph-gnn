/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholegraph/embedding.h>

#include <cuda_runtime_api.h>

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph_tensor_op.h>

#include <memory>

#include "cuda_macros.hpp"
#include "embedding.hpp"
#include "embedding_optimizer.hpp"
#include "error.hpp"
#include "integer_utils.hpp"
#include "logger.hpp"
#include "wholegraph/wholegraph.h"
#include "wholegraph_tensor_ops/functions/embedding_cache_func.h"
#include "wholegraph_tensor_ops/functions/exchange_embeddings_nccl_func.h"
#include "wholegraph_tensor_ops/functions/exchange_ids_nccl_func.h"
#include "wholegraph_tensor_ops/functions/gather_cached_func.h"
#include "wholegraph_tensor_ops/functions/gather_scatter_func.h"
#include "wholegraph_tensor_ops/functions/map_indices_func.h"
#include "wholegraph_tensor_ops/temp_memory_handle.hpp"
#include "wholegraph_tensor_ops/thrust_allocator.hpp"

namespace wholegraph {

static int64_t align_embedding_dim(int64_t embedding_dim, size_t element_size)
{
  int64_t const align_count      = 16 / element_size;
  int64_t const embedding_stride = embedding_dim % align_count == 0
                                     ? embedding_dim
                                     : (embedding_dim / align_count + 1) * align_count;
  return embedding_stride;
}

wholegraph_error_code_t embedding_base::set_optimizer(wholegraph_embedding_optimizer_t opt)
{
  try {
    if (optimizer != nullptr) {
      WHOLEGRAPH_ERROR("optimizer can only be set once.");
      return WHOLEGRAPH_NOT_SUPPORTED;
    }
    optimizer = opt;
    if (optimizer != nullptr) {
      if (embedding_dtype_ != WHOLEGRAPH_DT_FLOAT && embedding_dtype_ != WHOLEGRAPH_DT_HALF &&
          embedding_dtype_ != WHOLEGRAPH_DT_BF16) {
        WHOLEGRAPH_ERROR("Only float, half and bf16 embeddings support training.");
        return WHOLEGRAPH_NOT_IMPLEMENTED;
      }
      if (cache_policy != nullptr) {
        WHOLEGRAPH_CHECK_NOTHROW(cache_policy->access_type == WHOLEGRAPH_AT_READWRITE);
        if (cache_policy->cache_comm != raw_embedding_comm_) {
          WHOLEGRAPH_ERROR("optimizer not supported for local cached global readonly embedding.");
          return WHOLEGRAPH_INVALID_INPUT;
        }
      }
      optimizer_impl_base_ = static_cast<embedding_optimizer_impl_base*>(optimizer);
      WHOLEGRAPH_RETURN_ON_FAIL(create_optimizer_states());
      WHOLEGRAPH_RETURN_ON_FAIL(init_optimizer_states());
    }
  } catch (std::bad_alloc& sba) {
    WHOLEGRAPH_ERROR("bad_alloc");
    return WHOLEGRAPH_OUT_OF_MEMORY;
  } catch (...) {
    WHOLEGRAPH_ERROR("Unknown error");
    return WHOLEGRAPH_UNKNOW_ERROR;
  }

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_base::allocate(
  wholegraph_matrix_description_t* embedding_description,
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  wholegraph_embedding_cache_policy_t policy,
  size_t* embedding_entry_partition) noexcept
{
  cache_policy        = policy;
  raw_embedding_comm_ = comm;
  embedding_dtype_    = embedding_description->dtype;
  wholegraph_tensor_description_t padded_embedding_tensor_description;
  try {
    if (cache_policy != nullptr) {
      WHOLEGRAPH_CHECK_NOTHROW(cache_policy->cache_comm != nullptr);
      if (cache_policy->cache_comm != comm) {
        cache_ptr_ = new wholegraph::local_cache_for_global(cache_policy);
      } else {
        cache_ptr_ = new wholegraph::device_cache_for_host(cache_policy);
      }
      WHOLEGRAPH_RETURN_ON_FAIL(
        cache_ptr_->get_embedding_requirement(&padded_embedding_tensor_description,
                                              *embedding_description,
                                              comm,
                                              memory_type,
                                              memory_location));
      embedding_entry_partition = nullptr;
    } else {
      wholegraph_copy_matrix_desc_to_tensor(&padded_embedding_tensor_description,
                                             embedding_description);
      int64_t const embedding_dim = embedding_description->sizes[1];
      size_t const element_size = wholegraph_dtype_get_element_size(embedding_description->dtype);
      int64_t const embedding_stride = align_embedding_dim(embedding_dim, element_size);
      padded_embedding_tensor_description.storage_offset = 0;
      padded_embedding_tensor_description.strides[0]     = embedding_stride;
      padded_embedding_tensor_description.strides[1]     = 1;
    }
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_create_tensor(&allocated_embedding,
                                                         &padded_embedding_tensor_description,
                                                         comm,
                                                         memory_type,
                                                         memory_location,
                                                         embedding_entry_partition));
    int64_t starts[2] = {0, 0};
    int64_t ends[2]   = {embedding_description->sizes[0], embedding_description->sizes[1]};
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_get_subtensor(allocated_embedding, &starts[0], &ends[0], &user_embedding));
    if (cache_ptr_ != nullptr) { WHOLEGRAPH_RETURN_ON_FAIL(cache_ptr_->allocate(user_embedding)); }
  } catch (std::bad_alloc& sba) {
    WHOLEGRAPH_ERROR("bad_alloc");
    return WHOLEGRAPH_OUT_OF_MEMORY;
  } catch (...) {
    WHOLEGRAPH_ERROR("Unknown error");
    return WHOLEGRAPH_UNKNOW_ERROR;
  }

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_base::gather_gradient_apply(wholegraph_tensor_t indices,
                                                               wholegraph_tensor_t grads,
                                                               bool adjust_cache,
                                                               float lr,
                                                               wholegraph_env_func_t* p_env_fns,
                                                               cudaStream_t stream)
{
  auto* indice_desc    = wholegraph_tensor_get_tensor_description(indices);
  auto* grads_desc     = wholegraph_tensor_get_tensor_description(grads);
  auto* embedding_desc = wholegraph_tensor_get_tensor_description(allocated_embedding);
  WHOLEGRAPH_CHECK_NOTHROW(indice_desc->dim == 1);
  wholegraph_tensor_ops::temp_memory_handle host_recv_rank_id_count_handle(p_env_fns),
    host_rank_id_count_handle(p_env_fns);
  wholegraph_tensor_ops::temp_memory_handle dev_recv_indices_buffer_handle(p_env_fns);
  wholegraph_tensor_ops::temp_memory_handle dev_raw_indice_handle(p_env_fns);
  wholegraph_tensor_ops::wg_thrust_allocator thrust_allocator(p_env_fns);
  int world_size = -1, world_rank = -1;
  int64_t* host_recv_rank_id_count_ptr = nullptr;
  int64_t* host_rank_id_count_ptr      = nullptr;
  int64_t* dev_raw_indice_ptr          = nullptr;

  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&world_size, raw_embedding_comm_));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_rank(&world_rank, raw_embedding_comm_));
  host_recv_rank_id_count_ptr = static_cast<int64_t*>(
    host_recv_rank_id_count_handle.pinned_malloc(world_size, WHOLEGRAPH_DT_INT64));
  host_rank_id_count_ptr = static_cast<int64_t*>(
    host_rank_id_count_handle.pinned_malloc(world_size, WHOLEGRAPH_DT_INT64));
  dev_raw_indice_ptr = static_cast<int64_t*>(
    dev_raw_indice_handle.device_malloc(indice_desc->sizes[0], WHOLEGRAPH_DT_INT64));
  wholegraph_array_description_t indice_array_desc;
  WHOLEGRAPH_CHECK_NOTHROW(
    wholegraph_convert_tensor_desc_to_array(&indice_array_desc, indice_desc));

  wholegraph_tensor_ops::temp_memory_handle dev_embedding_entry_offsets_handle(p_env_fns);
  size_t* dev_embedding_entry_offsets_ptr = static_cast<size_t*>(
    dev_embedding_entry_offsets_handle.device_malloc(world_size + 1, WHOLEGRAPH_DT_INT64));
  wholegraph_tensor_ops::temp_memory_handle host_embedding_entry_offsets_handle(p_env_fns);
  size_t* host_embedding_entry_offsets_ptr = static_cast<size_t*>(
    host_embedding_entry_offsets_handle.host_malloc(world_size + 1, WHOLEGRAPH_DT_INT64));

  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_tensor_get_entry_offsets(host_embedding_entry_offsets_ptr, allocated_embedding));
  WG_CUDA_CHECK_NO_THROW(cudaMemcpy(dev_embedding_entry_offsets_ptr,
                                    host_embedding_entry_offsets_ptr,
                                    (world_size + 1) * sizeof(size_t),
                                    cudaMemcpyHostToDevice));

  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_tensor_ops::bucket_and_exchange_ids_func(wholegraph_tensor_get_data_pointer(indices),
                                                  indice_array_desc,
                                                  host_recv_rank_id_count_ptr,
                                                  host_rank_id_count_ptr,
                                                  &dev_recv_indices_buffer_handle,
                                                  dev_raw_indice_ptr,
                                                  dev_embedding_entry_offsets_ptr,
                                                  raw_embedding_comm_,
                                                  &thrust_allocator,
                                                  p_env_fns,
                                                  stream));

  int64_t total_recv_count = 0;
  for (int rank_id = 0; rank_id < world_size; rank_id++) {
    total_recv_count += host_recv_rank_id_count_ptr[rank_id];
  }

  wholegraph_tensor_ops::temp_memory_handle temp_grad_send_buffer_handle(p_env_fns),
    temp_grad_recv_buffer_handle(p_env_fns);
  void* temp_grad_send_buffer = temp_grad_send_buffer_handle.device_malloc(
    grads_desc->sizes[0] * grads_desc->sizes[1], grads_desc->dtype);
  void* temp_grad_recv_buffer = temp_grad_recv_buffer_handle.device_malloc(
    total_recv_count * grads_desc->sizes[1], grads_desc->dtype);

  auto grads_gref =
    wholegraph_create_continuous_global_reference(wholegraph_tensor_get_data_pointer(grads));
  wholegraph_matrix_description_t grads_mat_desc, temp_grad_send_desc;
  WHOLEGRAPH_CHECK_NOTHROW(wholegraph_convert_tensor_desc_to_matrix(&grads_mat_desc, grads_desc));
  temp_grad_send_desc        = grads_mat_desc;
  temp_grad_send_desc.stride = temp_grad_send_desc.sizes[1];

  wholegraph_array_description_t raw_indice_desc = indice_array_desc;
  raw_indice_desc.dtype                           = WHOLEGRAPH_DT_INT64;
  raw_indice_desc.storage_offset                  = 0;

  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::gather_func(grads_gref,
                                                          grads_mat_desc,
                                                          dev_raw_indice_ptr,
                                                          raw_indice_desc,
                                                          temp_grad_send_buffer,
                                                          temp_grad_send_desc,
                                                          stream));

  WG_CUDA_DEBUG_SYNC_STREAM(stream);

  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::exchange_embeddings_nccl_func(
    temp_grad_send_buffer,
    host_rank_id_count_ptr,
    host_recv_rank_id_count_ptr,
    temp_grad_recv_buffer,
    grads_desc->sizes[1] * wholegraph_dtype_get_element_size(grads_desc->dtype),
    raw_embedding_comm_,
    stream));

  wholegraph_tensor_ops::temp_memory_handle dedup_indice_recv_buffer_handle(p_env_fns);
  wholegraph_tensor_ops::temp_memory_handle dedup_grad_recv_buffer_handle(p_env_fns);
  void* dedup_indice =
    dedup_indice_recv_buffer_handle.device_malloc(total_recv_count, indice_desc->dtype);
  float* dedup_grads = static_cast<float*>(dedup_grad_recv_buffer_handle.device_malloc(
    total_recv_count * grads_desc->sizes[1], WHOLEGRAPH_DT_FLOAT));

  wholegraph_array_description_t recv_indice_array_desc = indice_array_desc;
  recv_indice_array_desc.size                            = total_recv_count;
  wholegraph_matrix_description_t recv_grad_matrix_desc = grads_mat_desc;
  recv_grad_matrix_desc.sizes[0]                         = total_recv_count;
  recv_grad_matrix_desc.stride                           = grads_mat_desc.sizes[1];

  int64_t const deduped_count =
    wholegraph_tensor_ops::dedup_indice_and_gradients(dev_recv_indices_buffer_handle.pointer(),
                                                recv_indice_array_desc,
                                                temp_grad_recv_buffer,
                                                recv_grad_matrix_desc,
                                                dedup_indice,
                                                dedup_grads,
                                                p_env_fns,
                                                stream);

  wholegraph_array_description_t update_indice_desc = indice_array_desc;
  update_indice_desc.size                            = deduped_count;
  if (adjust_cache && cache_ptr_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(cache_ptr_ != nullptr);
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_ops::update_cache_direct_same_comm(dedup_indice,
                                                     update_indice_desc,
                                                     user_embedding,
                                                     cache_ptr_->get_cache_local_data(),
                                                     cache_ptr_->get_cache_set_coverage(),
                                                     p_env_fns,
                                                     stream));
  }
  auto* state_embedding = optimizer_state_->cachable_state_embedding;
  if (adjust_cache && cache_ptr_ != nullptr && state_embedding != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(cache_ptr_ != nullptr);
    WHOLEGRAPH_CHECK_NOTHROW(optimizer_state_.get() != nullptr);
    WHOLEGRAPH_CHECK_NOTHROW(state_embedding != nullptr);
    embedding_base* state_embedding_base = static_cast<embedding_base*>(state_embedding);
    WHOLEGRAPH_CHECK_NOTHROW(state_embedding_base->cache_ptr_ != nullptr);
    wholegraph_embedding_get_embedding_tensor(state_embedding);
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::update_cache_direct_same_comm(
      dedup_indice,
      update_indice_desc,
      wholegraph_embedding_get_embedding_tensor(state_embedding),
      state_embedding_base->cache_ptr_->get_cache_local_data(),
      state_embedding_base->cache_ptr_->get_cache_set_coverage(),
      p_env_fns,
      stream));
  }

  WHOLEGRAPH_CHECK_NOTHROW(optimizer_impl_base_ != nullptr);
  wholegraph_tensor_t dedup_indice_tensor, dedup_grad_tensor;
  wholegraph_tensor_description_t recv_indice_tensor_desc = *indice_desc;
  recv_indice_tensor_desc.sizes[0]                         = deduped_count;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_make_tensor_from_pointer(
    &dedup_indice_tensor, dedup_indice, &recv_indice_tensor_desc));
  wholegraph_tensor_description_t recv_grad_tensor_desc = *grads_desc;
  recv_grad_tensor_desc.dtype                            = WHOLEGRAPH_DT_FLOAT;
  recv_grad_tensor_desc.sizes[0]                         = deduped_count;
  recv_grad_tensor_desc.strides[0]                       = recv_grad_tensor_desc.sizes[1];
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_make_tensor_from_pointer(&dedup_grad_tensor, dedup_grads, &recv_grad_tensor_desc));

  wholegraph_tensor_t local_embedding;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_map_local_tensor(user_embedding, &local_embedding));

  WHOLEGRAPH_RETURN_ON_FAIL(optimizer_impl_base_->step(
    dedup_indice_tensor, dedup_grad_tensor, local_embedding, optimizer_state_.get(), lr, stream));
  wholegraph_destroy_tensor(dedup_indice_tensor);
  wholegraph_destroy_tensor(dedup_grad_tensor);

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_base::create_optimizer_states() noexcept
{
  wholegraph_handle_t wg_handle = wholegraph_tensor_get_memory_handle(allocated_embedding);
  wholegraph_comm_t wg_raw_comm;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(&wg_raw_comm, wg_handle));

  int world_rank, world_size;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_rank(&world_rank, wg_raw_comm));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&world_size, wg_raw_comm));

  auto* allocated_tensor_desc = wholegraph_tensor_get_tensor_description(allocated_embedding);
  auto* user_tensor_desc      = wholegraph_tensor_get_tensor_description(user_embedding);
  int64_t start[2]            = {0, 0};
  int64_t end[2]              = {user_tensor_desc->sizes[1], -1};

  std::vector<size_t> allocated_tensor_entry_partition(world_size);
  std::vector<size_t> user_tensor_entry_partition(world_size);
  wholegraph_tensor_get_entry_partition_sizes(allocated_tensor_entry_partition.data(),
                                               allocated_embedding);
  wholegraph_tensor_get_entry_partition_sizes(user_tensor_entry_partition.data(), user_embedding);

  optimizer_state_                    = std::make_unique<optimizer_state_t>();
  optimizer_state_->local_start_index = 0;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_local_entry_start(
    (size_t*)&optimizer_state_->local_start_index, allocated_embedding));
  optimizer_impl_base_->create_optimizer_states(optimizer_state_.get(), user_tensor_desc->sizes[1]);
  bool const need_cachable_states = !optimizer_state_->cachable_states.empty();
  wholegraph_tensor_description_t cachable_state_desc;

  if (cache_ptr_ != nullptr) {
    try {
      optimizer_state_->device_cache_for_host_ = dynamic_cast<device_cache_for_host*>(cache_ptr_);
    } catch (...) {
      WHOLEGRAPH_FAIL_NOTHROW("cast from embedding_cache_base* to device_cache_for_host* failed.");
    }
  }

  if (need_cachable_states) {
    std::vector<int> embedding_offset(optimizer_state_->cachable_states.size(), 0);
    size_t element_size           = wholegraph_dtype_get_element_size(user_tensor_desc->dtype);
    int all_state_embedding_count = 0;
    for (size_t i = 0; i < embedding_offset.size(); i++) {
      auto& c_state             = optimizer_state_->cachable_states[i];
      embedding_offset[i]       = all_state_embedding_count;
      int state_embedding_dim   = c_state.dim;
      int aligned_embedding_dim = align_embedding_dim(state_embedding_dim, element_size);
      all_state_embedding_count += aligned_embedding_dim;
    }
    cachable_state_desc            = *user_tensor_desc;
    cachable_state_desc.dtype      = WHOLEGRAPH_DT_FLOAT;
    cachable_state_desc.sizes[1]   = all_state_embedding_count;
    cachable_state_desc.strides[0] = all_state_embedding_count;
    auto allocated_handle          = wholegraph_tensor_get_memory_handle(allocated_embedding);
    auto memory_type               = wholegraph_get_memory_type(allocated_handle);
    auto memory_location           = wholegraph_get_memory_location(allocated_handle);

    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_create_embedding(&optimizer_state_->cachable_state_embedding,
                                   &cachable_state_desc,
                                   raw_embedding_comm_,
                                   memory_type,
                                   memory_location,
                                   cache_policy,
                                   user_tensor_entry_partition.data()));

    optimizer_state_->global_cachable_raw_user_tensor =
      wholegraph_embedding_get_embedding_tensor(optimizer_state_->cachable_state_embedding);

    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_map_local_tensor(optimizer_state_->global_cachable_raw_user_tensor,
                                          &optimizer_state_->local_cachable_wg_tensor));
    for (size_t i = 0; i < embedding_offset.size(); i++) {
      auto& c_state     = optimizer_state_->cachable_states[i];
      c_state.start_dim = embedding_offset[i];
      start[1]          = embedding_offset[i];
      end[1]            = start[1] + c_state.dim;
      WHOLEGRAPH_RETURN_ON_FAIL(
        wholegraph_tensor_get_subtensor(optimizer_state_->global_cachable_raw_user_tensor,
                                         start,
                                         end,
                                         &c_state.global_raw_state_tensor));
    }
  }
  for (auto& uc_state : optimizer_state_->uncachable_states) {
    auto uc_desc     = *allocated_tensor_desc;
    uc_desc.dtype    = uc_state.dtype;
    uc_desc.sizes[1] = uc_desc.strides[0] = uc_state.dim;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_create_tensor(&uc_state.global_raw_padded_tensor,
                                                         &uc_desc,
                                                         wg_raw_comm,
                                                         WHOLEGRAPH_MT_DISTRIBUTED,
                                                         WHOLEGRAPH_ML_DEVICE,
                                                         allocated_tensor_entry_partition.data()));
    start[0] = 0;
    start[1] = 0;
    end[0]   = user_tensor_desc->sizes[0];
    end[1]   = uc_state.dim;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_subtensor(
      uc_state.global_raw_padded_tensor, start, end, &uc_state.global_raw_sub_tensor));
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_map_local_tensor(uc_state.global_raw_sub_tensor, &uc_state.local_tensor));
  }

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_base::destroy_optimizer_states() noexcept
{
  for (auto& c_state : optimizer_state_->cachable_states) {
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_destroy_tensor(c_state.global_raw_state_tensor));
  }
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_destroy_tensor(optimizer_state_->local_cachable_wg_tensor));
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_destroy_tensor(optimizer_state_->global_cachable_raw_user_tensor));
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_destroy_embedding(optimizer_state_->cachable_state_embedding));
  optimizer_state_->cachable_states.clear();
  for (auto& uc_state : optimizer_state_->uncachable_states) {
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_destroy_tensor(uc_state.local_tensor));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_destroy_tensor(uc_state.global_raw_sub_tensor));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_destroy_tensor(uc_state.global_raw_padded_tensor));
  }
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_base::set_gather_sms(int sms) noexcept
{
  if (sms != -1) {
    if (sms <= 0) {
      WHOLEGRAPH_WARN("Illegal SM number for gather/scatter! Will use default size.");
      sms = -1;
    } else if (sms > 1568) {
      WHOLEGRAPH_WARN("SM number for gather/scatter is too large! Will use default size.");
      sms = -1;
    }
  }
  gather_sms_ = sms;
  return WHOLEGRAPH_SUCCESS;
}

int embedding_base::get_round_robin_size() noexcept { return round_robin_size_; }

wholegraph_error_code_t embedding_base::set_shard_method(
  wholegraph_matrix_description_t* embedding_matrix_description,
  int embedding_world_size,
  int round_robin_size) noexcept
{
  round_robin_size_ = round_robin_size;
  if (round_robin_size != 0) {
    int64_t total_entry_count  = embedding_matrix_description->sizes[0];
    int first_rank_extra_entry = total_entry_count % (embedding_world_size * round_robin_size);
    if (first_rank_extra_entry > round_robin_size) first_rank_extra_entry = round_robin_size;
    int64_t first_rank_entry_size =
      total_entry_count / (embedding_world_size * round_robin_size) * round_robin_size;
    first_rank_entry_size += first_rank_extra_entry;
    total_entry_count                      = first_rank_entry_size * embedding_world_size;
    embedding_matrix_description->sizes[0] = total_entry_count;
  }
  return WHOLEGRAPH_SUCCESS;
}

void embedding_base::deallocate() noexcept
{
  if (optimizer != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(destroy_optimizer_states() == WHOLEGRAPH_SUCCESS);
  }
  if (cache_ptr_ != nullptr) {
    delete cache_ptr_;
    cache_ptr_ = nullptr;
  }
  WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(user_embedding) == WHOLEGRAPH_SUCCESS);
  WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(allocated_embedding) == WHOLEGRAPH_SUCCESS);
}

wholegraph_error_code_t embedding_base::writeback_embedding_cache(
  cudaStream_t stream) const noexcept
{
  if (cache_ptr_ != nullptr) {
    WHOLEGRAPH_RETURN_ON_FAIL(cache_ptr_->writeback_all_cache(stream));
  }
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_base::writeback_all_caches(cudaStream_t stream) const noexcept
{
  WHOLEGRAPH_RETURN_ON_FAIL(writeback_embedding_cache(stream));
  if (optimizer_impl_base_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(optimizer_state_.get() != nullptr);
    if (optimizer_state_->cachable_state_embedding != nullptr) {
      WHOLEGRAPH_RETURN_ON_FAIL(
        static_cast<embedding_base*>(optimizer_state_->cachable_state_embedding)
          ->writeback_all_caches(stream));
    }
  }
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_base::drop_embedding_cache(cudaStream_t stream) const noexcept
{
  if (cache_ptr_ != nullptr) { WHOLEGRAPH_RETURN_ON_FAIL(cache_ptr_->drop_all_cache(stream)); }
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_base::drop_all_caches(cudaStream_t stream) const noexcept
{
  WHOLEGRAPH_RETURN_ON_FAIL(drop_embedding_cache(stream));
  if (optimizer_impl_base_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(optimizer_state_.get() != nullptr);
    if (optimizer_state_->cachable_state_embedding != nullptr) {
      WHOLEGRAPH_RETURN_ON_FAIL(
        static_cast<embedding_base*>(optimizer_state_->cachable_state_embedding)
          ->drop_all_caches(stream));
    }
  }
  return WHOLEGRAPH_SUCCESS;
}

class noncached_embedding : public embedding_base {
 public:
  noncached_embedding()          = default;
  virtual ~noncached_embedding() = default;
  wholegraph_error_code_t gather(wholegraph_tensor_t indices,
                                  wholegraph_tensor_t output,
                                  bool adjust_cache,
                                  wholegraph_env_func_t* p_env_fns,
                                  cudaStream_t stream) noexcept override;
};

wholegraph_error_code_t noncached_embedding::gather(wholegraph_tensor_t indices,
                                                     wholegraph_tensor_t output,
                                                     bool adjust_cache,
                                                     wholegraph_env_func_t* p_env_fns,
                                                     cudaStream_t stream) noexcept
{
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_gather(allocated_embedding, indices, output, p_env_fns, stream, gather_sms_));
  return WHOLEGRAPH_SUCCESS;
}

class device_cached_host_embedding : public embedding_base {
 public:
  device_cached_host_embedding()          = default;
  virtual ~device_cached_host_embedding() = default;
  wholegraph_error_code_t gather(wholegraph_tensor_t indices,
                                  wholegraph_tensor_t output,
                                  bool adjust_cache,
                                  wholegraph_env_func_t* p_env_fns,
                                  cudaStream_t stream) noexcept override;
};

wholegraph_error_code_t device_cached_host_embedding::gather(wholegraph_tensor_t indices,
                                                              wholegraph_tensor_t output,
                                                              bool adjust_cache,
                                                              wholegraph_env_func_t* p_env_fns,
                                                              cudaStream_t stream) noexcept
{
  auto* indice_desc    = wholegraph_tensor_get_tensor_description(indices);
  auto* output_desc    = wholegraph_tensor_get_tensor_description(output);
  auto* embedding_desc = wholegraph_tensor_get_tensor_description(allocated_embedding);
  WHOLEGRAPH_CHECK_NOTHROW(indice_desc->dim == 1);
  wholegraph_tensor_ops::temp_memory_handle host_recv_rank_id_count_handle(p_env_fns),
    host_rank_id_count_handle(p_env_fns);
  wholegraph_tensor_ops::temp_memory_handle dev_recv_indices_buffer_handle(p_env_fns);
  wholegraph_tensor_ops::temp_memory_handle dev_raw_indice_handle(p_env_fns);
  wholegraph_tensor_ops::wg_thrust_allocator thrust_allocator(p_env_fns);
  int world_size = -1, world_rank = -1;
  int64_t* host_recv_rank_id_count_ptr = nullptr;
  int64_t* host_rank_id_count_ptr      = nullptr;
  int64_t* dev_raw_indice_ptr          = nullptr;
  int64_t total_recv_count             = 0;
  if (adjust_cache || cache_policy->cache_memory_type == WHOLEGRAPH_MT_DISTRIBUTED) {
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&world_size, raw_embedding_comm_));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_rank(&world_rank, raw_embedding_comm_));
    host_recv_rank_id_count_ptr = static_cast<int64_t*>(
      host_recv_rank_id_count_handle.pinned_malloc(world_size, WHOLEGRAPH_DT_INT64));
    host_rank_id_count_ptr = static_cast<int64_t*>(
      host_rank_id_count_handle.pinned_malloc(world_size, WHOLEGRAPH_DT_INT64));
    dev_raw_indice_ptr = static_cast<int64_t*>(
      dev_raw_indice_handle.device_malloc(indice_desc->sizes[0], WHOLEGRAPH_DT_INT64));
    wholegraph_array_description_t indice_array_desc;
    WHOLEGRAPH_CHECK_NOTHROW(
      wholegraph_convert_tensor_desc_to_array(&indice_array_desc, indice_desc));

    wholegraph_tensor_ops::temp_memory_handle dev_embedding_entry_offsets_handle(p_env_fns);
    size_t* dev_embedding_entry_offsets_ptr = static_cast<size_t*>(
      dev_embedding_entry_offsets_handle.device_malloc(world_size + 1, WHOLEGRAPH_DT_INT64));
    wholegraph_tensor_ops::temp_memory_handle host_embedding_entry_offsets_handle(p_env_fns);
    size_t* host_embedding_entry_offsets_ptr = static_cast<size_t*>(
      host_embedding_entry_offsets_handle.host_malloc(world_size + 1, WHOLEGRAPH_DT_INT64));

    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_get_entry_offsets(host_embedding_entry_offsets_ptr, allocated_embedding));
    WG_CUDA_CHECK_NO_THROW(cudaMemcpy(dev_embedding_entry_offsets_ptr,
                                      host_embedding_entry_offsets_ptr,
                                      (world_size + 1) * sizeof(size_t),
                                      cudaMemcpyHostToDevice));
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_ops::bucket_and_exchange_ids_func(wholegraph_tensor_get_data_pointer(indices),
                                                    indice_array_desc,
                                                    host_recv_rank_id_count_ptr,
                                                    host_rank_id_count_ptr,
                                                    &dev_recv_indices_buffer_handle,
                                                    dev_raw_indice_ptr,
                                                    dev_embedding_entry_offsets_ptr,
                                                    raw_embedding_comm_,
                                                    &thrust_allocator,
                                                    p_env_fns,
                                                    stream));
    if (adjust_cache) {
      total_recv_count = 0;
      for (int i = 0; i < world_size; i++) {
        total_recv_count += host_recv_rank_id_count_ptr[i];
      }
      auto update_indice_desc =
        wholegraph_create_array_desc(total_recv_count, 0, indice_desc->dtype);
      WHOLEGRAPH_RETURN_ON_FAIL(
        wholegraph_tensor_ops::update_cache_direct_same_comm(dev_recv_indices_buffer_handle.pointer(),
                                                       update_indice_desc,
                                                       allocated_embedding,
                                                       cache_ptr_->get_cache_local_data(),
                                                       cache_ptr_->get_cache_set_coverage(),
                                                       p_env_fns,
                                                       stream));
      WG_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
      WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_barrier(cache_policy->cache_comm));
    }
  }
  if (cache_policy->cache_memory_type == WHOLEGRAPH_MT_DISTRIBUTED) {
    // Local Gather
    total_recv_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_recv_count += host_recv_rank_id_count_ptr[i];
    }
    wholegraph_tensor_ops::temp_memory_handle dev_local_gather_buffer(p_env_fns);
    wholegraph_tensor_ops::temp_memory_handle dev_embedding_recv_buffer(p_env_fns);
    void* dev_local_gather_buffer_ptr = dev_local_gather_buffer.device_malloc(
      embedding_desc->sizes[1] * total_recv_count, output_desc->dtype);
    void* dev_embedding_recv_buffer_ptr = dev_embedding_recv_buffer.device_malloc(
      embedding_desc->sizes[1] * indice_desc->sizes[0], output_desc->dtype);
    wholegraph_tensor_t local_raw_tensor;
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_map_local_tensor(allocated_embedding, &local_raw_tensor));
    wholegraph_gref_t local_raw_gref;
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_get_global_reference(local_raw_tensor, &local_raw_gref));

    wholegraph_tensor_t cached_embedding_local_tensor =
      cache_ptr_->get_cache_local_data()->cache_line_data_;
    wholegraph_gref_t cached_embedding_gref;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_global_reference(
      cached_embedding_local_tensor, &cached_embedding_gref));
    wholegraph_gref_t cache_line_tag_gref;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_global_reference(
      cache_ptr_->get_cache_local_data()->cache_line_tag_, &cache_line_tag_gref));

    size_t rank_start_gid = 0;
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_get_local_entry_start(&rank_start_gid, allocated_embedding));
    wholegraph_tensor_description_t recv_indices_desc;
    auto recv_indices_array_desc =
      wholegraph_create_array_desc(total_recv_count, 0, indice_desc->dtype);
    wholegraph_copy_array_desc_to_tensor(&recv_indices_desc, &recv_indices_array_desc);
    wholegraph_tensor_description_t local_gather_desc = *output_desc;
    local_gather_desc.sizes[0]                         = total_recv_count;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::gather_cached_func(
      local_raw_gref,
      wholegraph_tensor_get_tensor_description(local_raw_tensor),
      cached_embedding_gref,
      wholegraph_tensor_get_tensor_description(cached_embedding_local_tensor),
      cache_line_tag_gref,
      dev_recv_indices_buffer_handle.pointer(),
      &recv_indices_desc,
      dev_local_gather_buffer_ptr,
      &local_gather_desc,
      cache_ptr_->get_cache_set_coverage(),
      rank_start_gid,
      rank_start_gid,
      stream));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_destroy_tensor(local_raw_tensor));
    // AllToAllV
    wholegraph_comm_t wg_comm;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(
      &wg_comm, wholegraph_tensor_get_memory_handle(allocated_embedding)));
    size_t const embedding_size =
      embedding_desc->sizes[1] * wholegraph_dtype_get_element_size(output_desc->dtype);
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_ops::exchange_embeddings_nccl_func(dev_local_gather_buffer_ptr,
                                                     host_recv_rank_id_count_ptr,
                                                     host_rank_id_count_ptr,
                                                     dev_embedding_recv_buffer_ptr,
                                                     embedding_size,
                                                     wg_comm,
                                                     stream));
    WG_CUDA_DEBUG_SYNC_STREAM(stream);
    // Local reorder
    wholegraph_gref_t output_gref =
      wholegraph_create_continuous_global_reference(wholegraph_tensor_get_data_pointer(output));
    wholegraph_matrix_description_t local_recv_buffer_desc = wholegraph_create_matrix_desc(
      output_desc->sizes, output_desc->sizes[1], 0, output_desc->dtype);
    auto raw_indice_desc =
      wholegraph_create_array_desc(indice_desc->sizes[0], 0, WHOLEGRAPH_DT_INT64);
    int64_t total_need_scatter_count = 0;
    for (int i = 0; i < world_size; i++) {
      total_need_scatter_count += host_rank_id_count_ptr[i];
    }
    local_recv_buffer_desc.sizes[0] = total_need_scatter_count;
    raw_indice_desc.size            = total_need_scatter_count;
    wholegraph_matrix_description_t output_matrix_desc;
    WHOLEGRAPH_CHECK_NOTHROW(
      wholegraph_convert_tensor_desc_to_matrix(&output_matrix_desc, output_desc));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::scatter_func(dev_embedding_recv_buffer_ptr,
                                                             local_recv_buffer_desc,
                                                             dev_raw_indice_ptr,
                                                             raw_indice_desc,
                                                             output_gref,
                                                             output_matrix_desc,
                                                             stream));
    WG_CUDA_DEBUG_SYNC_STREAM(stream);
  } else {
    wholegraph_gref_t global_raw_gref, global_cached_gref, global_cached_line_tag_gref;
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_get_global_reference(allocated_embedding, &global_raw_gref));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_global_reference(
      cache_ptr_->cache_line_data_wg_tensor_, &global_cached_gref));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_global_reference(
      cache_ptr_->cache_line_tag_wg_tensor_, &global_cached_line_tag_gref));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::gather_cached_func(
      global_raw_gref,
      wholegraph_tensor_get_tensor_description(allocated_embedding),
      global_cached_gref,
      wholegraph_tensor_get_tensor_description(cache_ptr_->cache_line_data_wg_tensor_),
      global_cached_line_tag_gref,
      wholegraph_tensor_get_data_pointer(indices),
      indice_desc,
      wholegraph_tensor_get_data_pointer(output),
      output_desc,
      cache_ptr_->get_cache_set_coverage(),
      0,
      0,
      stream));
  }
  return WHOLEGRAPH_SUCCESS;
}

class local_cached_global_readonly_embedding : public embedding_base {
 public:
  local_cached_global_readonly_embedding()          = default;
  virtual ~local_cached_global_readonly_embedding() = default;
  wholegraph_error_code_t gather(wholegraph_tensor_t indices,
                                  wholegraph_tensor_t output,
                                  bool adjust_cache,
                                  wholegraph_env_func_t* p_env_fns,
                                  cudaStream_t stream) noexcept override;
};

wholegraph_error_code_t local_cached_global_readonly_embedding::gather(
  wholegraph_tensor_t indices,
  wholegraph_tensor_t output,
  bool adjust_cache,
  wholegraph_env_func_t* p_env_fns,
  cudaStream_t stream) noexcept
{
  WHOLEGRAPH_CHECK_NOTHROW(cache_policy->cache_memory_type != WHOLEGRAPH_MT_DISTRIBUTED);
  auto* indice_desc = wholegraph_tensor_get_tensor_description(indices);
  auto* output_desc = wholegraph_tensor_get_tensor_description(output);
  WHOLEGRAPH_CHECK_NOTHROW(indice_desc->dim == 1);
  wholegraph_tensor_ops::temp_memory_handle host_recv_rank_id_count_handle(p_env_fns),
    host_rank_id_count_handle(p_env_fns);
  wholegraph_tensor_ops::temp_memory_handle dev_recv_indices_buffer_handle(p_env_fns);
  wholegraph_tensor_ops::temp_memory_handle dev_raw_indice_handle(p_env_fns);
  wholegraph_tensor_ops::wg_thrust_allocator thrust_allocator(p_env_fns);
  int cache_world_size = -1, cache_world_rank = -1;
  int64_t* host_recv_rank_id_count_ptr = nullptr;
  int64_t* host_rank_id_count_ptr      = nullptr;
  int64_t* dev_raw_indice_ptr          = nullptr;
  int64_t total_recv_count             = 0;
  // Actually, WHOLEGRAPH_MT_DISTRIBUTED is actully not supported now
  if (adjust_cache) {
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_communicator_get_size(&cache_world_size, cache_policy->cache_comm));
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_communicator_get_rank(&cache_world_rank, cache_policy->cache_comm));
    host_recv_rank_id_count_ptr = static_cast<int64_t*>(
      host_recv_rank_id_count_handle.pinned_malloc(cache_world_size, WHOLEGRAPH_DT_INT64));
    host_rank_id_count_ptr = static_cast<int64_t*>(
      host_rank_id_count_handle.pinned_malloc(cache_world_size, WHOLEGRAPH_DT_INT64));
    dev_raw_indice_ptr = static_cast<int64_t*>(
      dev_raw_indice_handle.device_malloc(indice_desc->sizes[0], WHOLEGRAPH_DT_INT64));
    wholegraph_array_description_t indice_array_desc;
    WHOLEGRAPH_CHECK_NOTHROW(
      wholegraph_convert_tensor_desc_to_array(&indice_array_desc, indice_desc));

    wholegraph_tensor_ops::temp_memory_handle dev_embedding_entry_offsets_handle(p_env_fns);
    size_t* dev_embedding_entry_offsets_ptr = static_cast<size_t*>(
      dev_embedding_entry_offsets_handle.device_malloc(cache_world_size + 1, WHOLEGRAPH_DT_INT64));
    wholegraph_tensor_ops::temp_memory_handle host_embedding_entry_offsets_handle(p_env_fns);
    size_t* host_embedding_entry_offsets_ptr = static_cast<size_t*>(
      host_embedding_entry_offsets_handle.host_malloc(cache_world_size + 1, WHOLEGRAPH_DT_INT64));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_entry_offsets(
      host_embedding_entry_offsets_ptr, cache_ptr_->access_count_wg_tensor_));
    WG_CUDA_CHECK_NO_THROW(cudaMemcpy(dev_embedding_entry_offsets_ptr,
                                      host_embedding_entry_offsets_ptr,
                                      (cache_world_size + 1) * sizeof(size_t),
                                      cudaMemcpyHostToDevice));
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_tensor_ops::bucket_and_exchange_ids_func(wholegraph_tensor_get_data_pointer(indices),
                                                    indice_array_desc,
                                                    host_recv_rank_id_count_ptr,
                                                    host_rank_id_count_ptr,
                                                    &dev_recv_indices_buffer_handle,
                                                    dev_raw_indice_ptr,
                                                    dev_embedding_entry_offsets_ptr,
                                                    cache_policy->cache_comm,
                                                    &thrust_allocator,
                                                    p_env_fns,
                                                    stream));
    // adjust cache
    {
      total_recv_count = 0;
      for (int i = 0; i < cache_world_size; i++) {
        total_recv_count += host_recv_rank_id_count_ptr[i];
      }
      auto update_indice_desc =
        wholegraph_create_array_desc(total_recv_count, 0, indice_desc->dtype);
      WHOLEGRAPH_RETURN_ON_FAIL(
        wholegraph_tensor_ops::update_cache_different_comm(dev_recv_indices_buffer_handle.pointer(),
                                                     update_indice_desc,
                                                     allocated_embedding,
                                                     cache_policy->cache_comm,
                                                     host_embedding_entry_offsets_ptr,
                                                     cache_ptr_->get_cache_local_data(),
                                                     cache_ptr_->get_cache_set_coverage(),
                                                     p_env_fns,
                                                     stream));
      WG_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
      WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_barrier(cache_policy->cache_comm));
    }
  }
  wholegraph_gref_t cached_gref, cached_line_tag_gref;
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_tensor_get_global_reference(cache_ptr_->cache_line_data_wg_tensor_, &cached_gref));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_global_reference(
    cache_ptr_->cache_line_tag_wg_tensor_, &cached_line_tag_gref));
  wholegraph_tensor_ops::temp_memory_handle dev_miss_ids_handle(p_env_fns);
  void* dev_miss_ids_ptr =
    dev_miss_ids_handle.device_malloc(indice_desc->sizes[0], indice_desc->dtype);
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::try_gather_cached_func(
    cached_gref,
    wholegraph_tensor_get_tensor_description(cache_ptr_->cache_line_data_wg_tensor_),
    cached_line_tag_gref,
    wholegraph_tensor_get_data_pointer(indices),
    indice_desc,
    nullptr,
    dev_miss_ids_ptr,
    wholegraph_tensor_get_data_pointer(output),
    output_desc,
    cache_ptr_->get_cache_set_coverage(),
    0,
    stream));
  wholegraph_tensor_t missed_indices_tensor;
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_make_tensor_from_pointer(&missed_indices_tensor, dev_miss_ids_ptr, indice_desc));
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_gather(allocated_embedding, missed_indices_tensor, output, p_env_fns, stream));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_destroy_tensor(missed_indices_tensor));

  return WHOLEGRAPH_SUCCESS;
}

}  // namespace wholegraph

#ifdef __cplusplus
extern "C" {
#endif

wholegraph_error_code_t wholegraph_create_embedding_optimizer(
  wholegraph_embedding_optimizer_t* optimizer, wholegraph_optimizer_type_t optimizer_type)
{
  return wholegraph::create_embedding_optimizer(optimizer, optimizer_type);
}

wholegraph_error_code_t wholegraph_optimizer_set_parameter(
  wholegraph_embedding_optimizer_t optimizer, const char* parameter_name, void* value)
{
  return wholegraph::optimizer_set_parameter(optimizer, parameter_name, value);
}

void wholegraph_destroy_embedding_optimizer(wholegraph_embedding_optimizer_t optimizer)
{
  wholegraph::destroy_embedding_optimizer(optimizer);
}

wholegraph_error_code_t wholegraph_create_embedding_cache_policy(
  wholegraph_embedding_cache_policy_t* cache_policy,
  wholegraph_comm_t cache_level_comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  wholegraph_access_type_t access_type,
  float cache_ratio)
{
  if (cache_ratio > 1.0F || cache_ratio < 1.0F / 512) {
    WHOLEGRAPH_ERROR("cache_ratio should in range [1/512, 1.0]");
    return WHOLEGRAPH_INVALID_VALUE;
  }
  auto* embedding_cache_policy                  = new wholegraph_embedding_cache_policy_;
  embedding_cache_policy->cache_comm            = cache_level_comm;
  embedding_cache_policy->cache_memory_type     = memory_type;
  embedding_cache_policy->cache_memory_location = memory_location;
  embedding_cache_policy->access_type           = access_type;
  embedding_cache_policy->cache_ratio           = cache_ratio;
  *cache_policy                                 = embedding_cache_policy;
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_destroy_embedding_cache_policy(
  wholegraph_embedding_cache_policy_t cache_policy)
{
  delete cache_policy;
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_create_embedding(
  wholegraph_embedding_t* wholegraph_embedding,
  wholegraph_tensor_description_t* embedding_description,
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  wholegraph_embedding_cache_policy_t cache_policy,
  size_t* embedding_entry_partition,
  int user_defined_sms,
  int round_robin_size)
{
  wholegraph_matrix_description_t embedding_matrix_description;
  if (!wholegraph_convert_tensor_desc_to_matrix(&embedding_matrix_description,
                                                 embedding_description)) {
    WHOLEGRAPH_ERROR("wholegraph_create_embedding input description must be 2D matrix");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  wholegraph::embedding_base* embedding_impl_ptr = nullptr;
  int embedding_world_size                        = 1;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&embedding_world_size, comm));
  if (cache_policy != nullptr) {
    if (cache_policy->cache_comm == comm) {
      if (cache_policy->cache_memory_location != WHOLEGRAPH_ML_DEVICE) {
        WHOLEGRAPH_ERROR(
          "Cache has same communicator with raw embedding, should be device cached host embedding,"
          " but cache memory location is not WHOLEGRAPH_ML_DEVICE.");
        return WHOLEGRAPH_INVALID_INPUT;
      }
      if (cache_policy->cache_memory_type < memory_type) {
        WHOLEGRAPH_ERROR(
          "For device cached host memory, raw embedding should cover cache's address modes.");
        return WHOLEGRAPH_INVALID_INPUT;
      }
      embedding_impl_ptr = new wholegraph::device_cached_host_embedding();
    } else {
      int const cache_world_size = 1;
      WHOLEGRAPH_RETURN_ON_FAIL(
        wholegraph_communicator_get_size(&embedding_world_size, cache_policy->cache_comm));
      WHOLEGRAPH_CHECK_NOTHROW(cache_world_size <= embedding_world_size);
      if (cache_policy->cache_memory_type != WHOLEGRAPH_MT_CONTINUOUS) {
        WHOLEGRAPH_ERROR(
          "For local cached global readonly embedding, cache_memory_type should be continuous.");
        return WHOLEGRAPH_INVALID_INPUT;
      }
      if (cache_policy->access_type != WHOLEGRAPH_AT_READONLY) {
        WHOLEGRAPH_ERROR(
          "Only ReadOnly access type supported for local cached global readonly embedding.");
        return WHOLEGRAPH_INVALID_INPUT;
      }
      embedding_impl_ptr = new wholegraph::local_cached_global_readonly_embedding();
    }
    embedding_entry_partition = nullptr;
  } else {
    embedding_impl_ptr = new wholegraph::noncached_embedding();
  }
  if (embedding_entry_partition) {
    if (round_robin_size != 0) { WHOLEGRAPH_WARN("Parameter 'round_robin_size' is ignored."); }
    round_robin_size = 0;
  }
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&embedding_world_size, comm));
  embedding_impl_ptr->set_shard_method(
    &embedding_matrix_description, embedding_world_size, round_robin_size);
  embedding_impl_ptr->set_gather_sms(user_defined_sms);
  WHOLEGRAPH_RETURN_ON_FAIL(embedding_impl_ptr->allocate(&embedding_matrix_description,
                                                          comm,
                                                          memory_type,
                                                          memory_location,
                                                          cache_policy,
                                                          embedding_entry_partition));
  *wholegraph_embedding = static_cast<wholegraph_embedding_t>(embedding_impl_ptr);
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_embedding_set_optimizer(
  wholegraph_embedding_t wholegraph_embedding, wholegraph_embedding_optimizer_t optimizer)
{
  auto* embedding_impl_ptr = static_cast<wholegraph::embedding_base*>(wholegraph_embedding);
  WHOLEGRAPH_RETURN_ON_FAIL(embedding_impl_ptr->set_optimizer(optimizer));
  return WHOLEGRAPH_SUCCESS;
}
wholegraph_error_code_t wholegraph_destroy_embedding(
  wholegraph_embedding_t wholegraph_embedding)
{
  if (wholegraph_embedding == nullptr) { return WHOLEGRAPH_INVALID_INPUT; }
  auto* embedding_impl_ptr = static_cast<wholegraph::embedding_base*>(wholegraph_embedding);
  delete embedding_impl_ptr;
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_embedding_gather(wholegraph_embedding_t wholegraph_embedding,
                                                      wholegraph_tensor_t indices,
                                                      wholegraph_tensor_t output,
                                                      bool adjust_cache,
                                                      wholegraph_env_func_t* p_env_fns,
                                                      int64_t stream_int)
{
  auto* embedding_impl_ptr = static_cast<wholegraph::embedding_base*>(wholegraph_embedding);
  if (embedding_impl_ptr->get_round_robin_size() == 0)
    return embedding_impl_ptr->gather(
      indices, output, adjust_cache, p_env_fns, (cudaStream_t)stream_int);

  wholegraph_tensor_t mapped_indices;
  auto* indice_desc = wholegraph_tensor_get_tensor_description(indices);
  wholegraph_tensor_ops::temp_memory_handle mapped_indice_handle(p_env_fns);
  void* mapped_indice_ptr =
    mapped_indice_handle.device_malloc(indice_desc->sizes[0], indice_desc->dtype);
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_make_tensor_from_pointer(&mapped_indices, mapped_indice_ptr, indice_desc));
  wholegraph_tensor_ops::storage_index2wg_embedding_index(indices,
                                                    mapped_indices,
                                                    embedding_impl_ptr->allocated_embedding,
                                                    embedding_impl_ptr->get_round_robin_size(),
                                                    stream_int);
  WHOLEGRAPH_RETURN_ON_FAIL(embedding_impl_ptr->gather(
    mapped_indices, output, adjust_cache, p_env_fns, (cudaStream_t)stream_int));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_destroy_tensor(mapped_indices));
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_embedding_gather_gradient_apply(
  wholegraph_embedding_t wholegraph_embedding,
  wholegraph_tensor_t indices,
  wholegraph_tensor_t grads,
  bool adjust_cache,
  float lr,
  wholegraph_env_func_t* p_env_fns,
  int64_t stream_int)
{
  auto* embedding_impl_ptr = static_cast<wholegraph::embedding_base*>(wholegraph_embedding);
  if (embedding_impl_ptr->get_round_robin_size() == 0)
    return embedding_impl_ptr->gather_gradient_apply(
      indices, grads, adjust_cache, lr, p_env_fns, (cudaStream_t)stream_int);

  wholegraph_tensor_t mapped_indices;
  auto* indice_desc = wholegraph_tensor_get_tensor_description(indices);
  wholegraph_tensor_ops::temp_memory_handle mapped_indice_handle(p_env_fns);
  void* mapped_indice_ptr =
    mapped_indice_handle.device_malloc(indice_desc->sizes[0], indice_desc->dtype);
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_make_tensor_from_pointer(&mapped_indices, mapped_indice_ptr, indice_desc));
  wholegraph_tensor_ops::storage_index2wg_embedding_index(indices,
                                                    mapped_indices,
                                                    embedding_impl_ptr->allocated_embedding,
                                                    embedding_impl_ptr->get_round_robin_size(),
                                                    stream_int);
  WHOLEGRAPH_RETURN_ON_FAIL(embedding_impl_ptr->gather_gradient_apply(
    mapped_indices, grads, adjust_cache, lr, p_env_fns, (cudaStream_t)stream_int));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_destroy_tensor(mapped_indices));
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_tensor_t wholegraph_embedding_get_embedding_tensor(
  wholegraph_embedding_t wholegraph_embedding)
{
  wholegraph::embedding_base* embedding_impl_ptr =
    static_cast<wholegraph::embedding_base*>(wholegraph_embedding);
  return embedding_impl_ptr->user_embedding;
}

const char* const* wholegraph_embedding_get_optimizer_state_names(
  wholegraph_embedding_t wholegraph_embedding)
{
  wholegraph::embedding_base* embedding_impl_ptr =
    static_cast<wholegraph::embedding_base*>(wholegraph_embedding);
  return embedding_impl_ptr->get_optimizer_state_names();
}

wholegraph_tensor_t wholegraph_embedding_get_optimizer_state(
  wholegraph_embedding_t wholegraph_embedding, const char* name)
{
  wholegraph::embedding_base* embedding_impl_ptr =
    static_cast<wholegraph::embedding_base*>(wholegraph_embedding);
  return embedding_impl_ptr->get_optimizer_state(name);
}

wholegraph_error_code_t wholegraph_embedding_writeback_cache(
  wholegraph_embedding_t wholegraph_embedding, int64_t stream_int)
{
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);
  return static_cast<wholegraph::embedding_base*>(wholegraph_embedding)
    ->writeback_all_caches(stream);
}

wholegraph_error_code_t wholegraph_embedding_drop_all_cache(
  wholegraph_embedding_t wholegraph_embedding, int64_t stream_int)
{
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);
  return static_cast<wholegraph::embedding_base*>(wholegraph_embedding)->drop_all_caches(stream);
}

#ifdef __cplusplus
}
#endif
