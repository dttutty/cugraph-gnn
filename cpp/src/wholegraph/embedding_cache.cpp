/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "embedding_cache.hpp"

#include <cmath>

#include "integer_utils.hpp"
#include "logger.hpp"
#include "memory_handle.hpp"
#include "wholegraph_tensor_ops/functions/embedding_cache_func.h"

namespace wholegraph {

embedding_cache_local_data::~embedding_cache_local_data()
{
  if (cache_line_tag_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(cache_line_tag_) == WHOLEGRAPH_SUCCESS);
    cache_line_tag_ = nullptr;
  }
  if (cache_line_lfu_count_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(cache_line_lfu_count_) ==
                              WHOLEGRAPH_SUCCESS);
    cache_line_lfu_count_ = nullptr;
  }
  if (cache_line_data_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(cache_line_data_) == WHOLEGRAPH_SUCCESS);
    cache_line_data_ = nullptr;
  }
  if (access_count_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(access_count_) == WHOLEGRAPH_SUCCESS);
    access_count_ = nullptr;
  }
}

embedding_cache_base::embedding_cache_base(wholegraph_embedding_cache_policy_t cache_policy)
{
  cache_policy_ = cache_policy;
}

embedding_cache_base::~embedding_cache_base()
{
  if (cache_line_tag_wg_tensor_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(cache_line_tag_wg_tensor_) ==
                              WHOLEGRAPH_SUCCESS);
    cache_line_tag_wg_tensor_ = nullptr;
  }
  if (cache_line_lfu_count_wg_tensor_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(cache_line_lfu_count_wg_tensor_) ==
                              WHOLEGRAPH_SUCCESS);
    cache_line_lfu_count_wg_tensor_ = nullptr;
  }
  if (cache_line_data_wg_tensor_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(cache_line_data_wg_tensor_) ==
                              WHOLEGRAPH_SUCCESS);
    cache_line_data_wg_tensor_ = nullptr;
  }
  if (access_count_wg_tensor_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_tensor(access_count_wg_tensor_) ==
                              WHOLEGRAPH_SUCCESS);
    access_count_wg_tensor_ = nullptr;
  }
  if (cache_policy_ != nullptr) {
    WHOLEGRAPH_CHECK_NOTHROW(wholegraph_destroy_embedding_cache_policy(cache_policy_));
    cache_policy_ = nullptr;
  }
}

void embedding_cache_base::pad_last_dim(wholegraph_matrix_description_t data_desc) noexcept
{
  matrix_description_           = data_desc;
  int64_t const embedding_count = matrix_description_.sizes[0];
  int64_t const embedding_dim   = matrix_description_.sizes[1];
  size_t const element_size     = wholegraph_dtype_get_element_size(matrix_description_.dtype);
  WHOLEGRAPH_CHECK_NOTHROW(element_size != -1);
  int64_t const align_count      = kEmbeddingAlignmentInBytes / element_size;
  int64_t const embedding_stride = round_up_unsafe<int64_t>(embedding_dim, align_count);
  matrix_description_.stride     = embedding_stride;
  padded_matrix_description_     = matrix_description_;
}

wholegraph_error_code_t embedding_cache_base::check_raw_tensor(
  wholegraph_tensor_t raw_data_tensor) noexcept
{
  // Check all are same as requested.
  if (raw_data_tensor == nullptr) {
    WHOLEGRAPH_ERROR("raw_data_tensor is null");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (!wholegraph_tensor_has_handle(raw_data_tensor)) {
    WHOLEGRAPH_ERROR("raw_data_tensor is not WholeGraph Tensor");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto* mem_handle = wholegraph_tensor_get_memory_handle(raw_data_tensor);
  if (mem_handle == nullptr) {
    WHOLEGRAPH_ERROR("raw_data_tensor WholeGraph Handle is nullptr");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (wholegraph_get_memory_type(mem_handle) != raw_memory_type_ ||
      get_memory_location(mem_handle) != raw_memory_location_) {
    WHOLEGRAPH_ERROR(
      "raw_data_tensor WholeGraph type or location is not same as get_embedding_requirement");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  wholegraph_comm_t comm = nullptr;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(&comm, mem_handle));
  if (comm != raw_comm_) {
    WHOLEGRAPH_ERROR(
      "raw_data_tensor WholeGraph communicator is not same as get_embedding_requirement");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto* raw_desc = wholegraph_tensor_get_tensor_description(raw_data_tensor);
  try {
    WHOLEGRAPH_CHECK(raw_desc->dim == 2 && raw_desc->storage_offset == 0);
    WHOLEGRAPH_CHECK(raw_desc->dtype == matrix_description_.dtype);
    WHOLEGRAPH_CHECK(raw_desc->strides[0] == matrix_description_.stride &&
                      raw_desc->strides[1] == 1);
    WHOLEGRAPH_CHECK(raw_desc->sizes[0] == matrix_description_.sizes[0] &&
                      raw_desc->sizes[1] == matrix_description_.sizes[1]);
  } catch (wholegraph::logic_error& wle) {
    WHOLEGRAPH_ERROR("check_raw_tensor failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (!wholegraph_tensor_has_handle(raw_data_tensor)) {
    WHOLEGRAPH_ERROR("should be WholeGraph Tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto* root_tensor = wholegraph_tensor_get_root(raw_data_tensor);
  auto* root_desc   = wholegraph_tensor_get_tensor_description(root_tensor);
  try {
    WHOLEGRAPH_CHECK(root_desc->dim == 2 && root_desc->storage_offset == 0);
    WHOLEGRAPH_CHECK(root_desc->dtype == padded_matrix_description_.dtype);
    WHOLEGRAPH_CHECK(root_desc->strides[0] == padded_matrix_description_.stride &&
                      root_desc->strides[1] == 1);
    WHOLEGRAPH_CHECK(root_desc->sizes[0] == padded_matrix_description_.sizes[0] &&
                      root_desc->sizes[1] == padded_matrix_description_.sizes[1]);
  } catch (wholegraph::logic_error& wle) {
    WHOLEGRAPH_ERROR("check_raw_tensor failed for root tensor.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_cache_base::compute_cache_set_coverage() noexcept
{
  if (cache_policy_ == nullptr) {
    WHOLEGRAPH_ERROR("cache_policy_ not set.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  float const cache_ratio = cache_policy_->cache_ratio;
  if (cache_ratio >= 1.0F || cache_ratio <= 0.0F) {
    WHOLEGRAPH_ERROR("Invalid cache ratio %f, should be in range (0.0, 1.0).", cache_ratio);
    return WHOLEGRAPH_INVALID_VALUE;
  }
  cache_set_coverage_ = std::round(kCacheSetSize / cache_ratio);
  cache_set_coverage_ = std::min(cache_set_coverage_, kMaxCacheSetCoverage);
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_cache_base::allocate(
  wholegraph_tensor_t raw_data_tensor) noexcept
{
  WHOLEGRAPH_RETURN_ON_FAIL(check_raw_tensor(raw_data_tensor));
  padded_raw_tensor_    = wholegraph_tensor_get_root(raw_data_tensor);
  auto* padded_raw_desc = wholegraph_tensor_get_tensor_description(padded_raw_tensor_);
  WHOLEGRAPH_CHECK_NOTHROW(padded_raw_desc != nullptr);
  WHOLEGRAPH_CHECK_NOTHROW(padded_raw_desc->dim == 2);
  int64_t const padded_embedding_count = padded_embedding_count_for_cache_;
  WHOLEGRAPH_CHECK_NOTHROW(padded_embedding_count % cache_set_coverage_ == 0);
  int64_t const total_cache_set_count = padded_embedding_count / cache_set_coverage_;
  int cache_world_size                = 1;
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_communicator_get_size(&cache_world_size, cache_policy_->cache_comm));
  WHOLEGRAPH_CHECK_NOTHROW(total_cache_set_count % cache_world_size == 0);
  wholegraph_tensor_description_t cache_line_meta_desc;
  cache_line_meta_desc.dim            = 2;
  cache_line_meta_desc.dtype          = WHOLEGRAPH_DT_INT16;
  cache_line_meta_desc.storage_offset = 0;
  cache_line_meta_desc.sizes[0]       = total_cache_set_count;
  cache_line_meta_desc.sizes[1] = cache_line_meta_desc.strides[0] = kCacheSetSize;
  cache_line_meta_desc.strides[1]                                 = 1;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_create_tensor(&cache_line_tag_wg_tensor_,
                                                       &cache_line_meta_desc,
                                                       cache_policy_->cache_comm,
                                                       cache_policy_->cache_memory_type,
                                                       cache_policy_->cache_memory_location));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_create_tensor(&cache_line_lfu_count_wg_tensor_,
                                                       &cache_line_meta_desc,
                                                       cache_policy_->cache_comm,
                                                       cache_policy_->cache_memory_type,
                                                       cache_policy_->cache_memory_location));
  wholegraph_tensor_description_t cache_line_data_desc = cache_line_meta_desc;
  cache_line_data_desc.dtype                            = padded_raw_desc->dtype;
  cache_line_data_desc.sizes[0]                         = total_cache_set_count * kCacheSetSize;
  cache_line_data_desc.sizes[1]                         = padded_raw_desc->sizes[1];
  cache_line_data_desc.strides[0]                       = padded_raw_desc->strides[0];
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_create_tensor(&cache_line_data_wg_tensor_,
                                                       &cache_line_data_desc,
                                                       cache_policy_->cache_comm,
                                                       cache_policy_->cache_memory_type,
                                                       cache_policy_->cache_memory_location));
  wholegraph_tensor_description_t access_count_desc;
  access_count_desc.dim            = 1;
  access_count_desc.storage_offset = 0;
  access_count_desc.sizes[0]       = padded_embedding_count_for_cache_;
  access_count_desc.dtype          = WHOLEGRAPH_DT_INT64;
  access_count_desc.strides[0]     = 1;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_create_tensor(&access_count_wg_tensor_,
                                                       &access_count_desc,
                                                       cache_policy_->cache_comm,
                                                       cache_policy_->cache_memory_type,
                                                       cache_policy_->cache_memory_location));

  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_tensor_map_local_tensor(cache_line_tag_wg_tensor_, &local_cache_.cache_line_tag_));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_map_local_tensor(
    cache_line_lfu_count_wg_tensor_, &local_cache_.cache_line_lfu_count_));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_map_local_tensor(cache_line_data_wg_tensor_,
                                                                 &local_cache_.cache_line_data_));
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_tensor_map_local_tensor(access_count_wg_tensor_, &local_cache_.access_count_));

  size_t const local_cache_line_count = wholegraph_get_memory_element_count_from_tensor(
    wholegraph_tensor_get_tensor_description(local_cache_.cache_line_tag_));
  WG_CUDA_CHECK_NO_THROW(
    cudaMemset(wholegraph_tensor_get_data_pointer(local_cache_.cache_line_tag_),
               0,
               local_cache_line_count * sizeof(int16_t)));
  WG_CUDA_CHECK_NO_THROW(
    cudaMemset(wholegraph_tensor_get_data_pointer(local_cache_.cache_line_lfu_count_),
               0,
               local_cache_line_count * sizeof(int16_t)));
  size_t const local_access_count_count = wholegraph_get_memory_element_count_from_tensor(
    wholegraph_tensor_get_tensor_description(local_cache_.access_count_));
  WG_CUDA_CHECK_NO_THROW(cudaMemset(wholegraph_tensor_get_data_pointer(local_cache_.access_count_),
                                    0,
                                    local_access_count_count * sizeof(int64_t)));

  WG_CUDA_CHECK_NO_THROW(cudaDeviceSynchronize());
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_barrier(cache_policy_->cache_comm));

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_cache_base::writeback_all_cache(cudaStream_t stream) noexcept
{
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t embedding_cache_base::drop_all_cache(cudaStream_t stream) noexcept
{
  return WHOLEGRAPH_SUCCESS;
}

device_cache_for_host::device_cache_for_host(wholegraph_embedding_cache_policy_t cache_policy)
  : embedding_cache_base(cache_policy)
{
}

device_cache_for_host::~device_cache_for_host() {}

wholegraph_error_code_t device_cache_for_host::get_embedding_requirement(
  wholegraph_tensor_description_t* padded_desc,
  wholegraph_matrix_description_t data_desc,
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location) noexcept
{
  if (cache_policy_ == nullptr) {
    WHOLEGRAPH_ERROR("No cache policy set.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (cache_policy_->cache_memory_location != WHOLEGRAPH_ML_DEVICE) {
    WHOLEGRAPH_ERROR("device_cache_for_host cache memory should be device.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (comm != cache_policy_->cache_comm) {
    WHOLEGRAPH_ERROR("device_cache_for_host cache should use the same communicator as raw data.");
    return WHOLEGRAPH_INVALID_VALUE;
  }
  if (padded_raw_tensor_ != nullptr) {
    WHOLEGRAPH_ERROR("embedding_cache already cached other embedding.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (memory_type > cache_policy_->cache_memory_type) {
    WHOLEGRAPH_ERROR("embedding memory_type should support at least cache memory_type.");
    return WHOLEGRAPH_INVALID_VALUE;
  }

  compute_cache_set_coverage();
  pad_last_dim(data_desc);

  int64_t const embedding_count = matrix_description_.sizes[0];

  int world_size = 1;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&world_size, comm));
  padded_embedding_count_for_cache_ = round_up_unsafe<int64_t>(
    embedding_count, static_cast<int64_t>(world_size) * cache_set_coverage_);
  padded_matrix_description_.sizes[0] = padded_embedding_count_for_cache_;
  wholegraph_copy_matrix_desc_to_tensor(padded_desc, &padded_matrix_description_);

  raw_comm_            = comm;
  raw_memory_location_ = memory_location;
  raw_memory_type_     = memory_type;
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t device_cache_for_host::writeback_all_cache(cudaStream_t stream) noexcept
{
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::writeback_cache_direct_same_comm(
    padded_raw_tensor_, &local_cache_, cache_set_coverage_, false, stream));
  WG_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
  wholegraph_comm_t wg_comm;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(
    &wg_comm, wholegraph_tensor_get_memory_handle(padded_raw_tensor_)));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_barrier(wg_comm));
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t device_cache_for_host::drop_all_cache(cudaStream_t stream) noexcept
{
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_ops::writeback_cache_direct_same_comm(
    padded_raw_tensor_, &local_cache_, cache_set_coverage_, true, stream));
  WG_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
  wholegraph_comm_t wg_comm;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(
    &wg_comm, wholegraph_tensor_get_memory_handle(padded_raw_tensor_)));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_barrier(wg_comm));
  return WHOLEGRAPH_SUCCESS;
}

local_cache_for_global::local_cache_for_global(wholegraph_embedding_cache_policy_t cache_policy)
  : embedding_cache_base(cache_policy)
{
}

local_cache_for_global::~local_cache_for_global() {}

wholegraph_error_code_t local_cache_for_global::get_embedding_requirement(
  wholegraph_tensor_description_t* padded_desc,
  wholegraph_matrix_description_t data_desc,
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location) noexcept
{
  if (cache_policy_ == nullptr) {
    WHOLEGRAPH_ERROR("No cache policy set.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (cache_policy_->cache_memory_type != WHOLEGRAPH_MT_CONTINUOUS) {
    WHOLEGRAPH_ERROR(
      "local_cache_for_global cache only supports WHOLEGRAPH_MT_CONTINUOUS for now.");
    return WHOLEGRAPH_NOT_IMPLEMENTED;
  }
  if (padded_raw_tensor_ != nullptr) {
    WHOLEGRAPH_ERROR("embedding_cache already cached other embedding.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (cache_policy_->access_type != WHOLEGRAPH_AT_READONLY) {
    WHOLEGRAPH_ERROR("local_cache_for_global only READONLY cache supported.");
    return WHOLEGRAPH_NOT_IMPLEMENTED;
  }

  compute_cache_set_coverage();
  pad_last_dim(data_desc);

  int64_t const embedding_count = matrix_description_.sizes[0];
  int cache_world_size          = 1;
  if (cache_policy_->cache_comm != nullptr) {
    WHOLEGRAPH_RETURN_ON_FAIL(
      wholegraph_communicator_get_size(&cache_world_size, cache_policy_->cache_comm));
  }
  padded_embedding_count_for_cache_ = round_up_unsafe<int64_t>(
    embedding_count, static_cast<int64_t>(cache_world_size) * cache_set_coverage_);

  padded_matrix_description_.sizes[0] = padded_embedding_count_for_cache_;
  wholegraph_copy_matrix_desc_to_tensor(padded_desc, &padded_matrix_description_);
  raw_comm_            = comm;
  raw_memory_location_ = memory_location;
  raw_memory_type_     = memory_type;
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t local_cache_for_global::drop_all_cache(cudaStream_t stream) noexcept
{
  wholegraph_tensor_t local_tag_tensor = local_cache_.cache_line_tag_;
  size_t local_cache_line_size          = wholegraph_get_memory_size_from_tensor(
    wholegraph_tensor_get_tensor_description(local_tag_tensor));
  WG_CUDA_CHECK_NO_THROW(cudaMemsetAsync(
    wholegraph_tensor_get_data_pointer(local_tag_tensor), 0, local_cache_line_size, stream));
  WG_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));

  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_barrier(cache_policy_->cache_comm));
  return WHOLEGRAPH_SUCCESS;
}

}  // namespace wholegraph
