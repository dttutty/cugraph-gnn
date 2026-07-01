/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/embedding.h>
#include <wholegraph/wholegraph_tensor.h>

#include "embedding_optimizer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

struct wholegraph_embedding_ {
  wholegraph_tensor_t allocated_embedding          = nullptr;
  wholegraph_tensor_t user_embedding               = nullptr;  // subtensor of allocated_embedding
  wholegraph_embedding_cache_policy_t cache_policy = nullptr;
  wholegraph_embedding_optimizer_t optimizer       = nullptr;
};

#ifdef __cplusplus
}
#endif

namespace wholegraph {

class embedding_base : public wholegraph_embedding_ {
 public:
  embedding_base()          = default;
  virtual ~embedding_base() = default;
  wholegraph_error_code_t allocate(wholegraph_matrix_description_t* embedding_description,
                                    wholegraph_comm_t comm,
                                    wholegraph_memory_type_t memory_type,
                                    wholegraph_memory_location_t memory_location,
                                    wholegraph_embedding_cache_policy_t policy,
                                    size_t* embedding_entry_partition) noexcept;
  void deallocate() noexcept;
  virtual wholegraph_error_code_t gather(wholegraph_tensor_t indices,
                                          wholegraph_tensor_t output,
                                          bool adjust_cache,
                                          wholegraph_env_func_t* p_env_fns,
                                          cudaStream_t stream) noexcept = 0;

  wholegraph_error_code_t gather_gradient_apply(wholegraph_tensor_t indices,
                                                 wholegraph_tensor_t grads,
                                                 bool adjust_cache,
                                                 float lr,
                                                 wholegraph_env_func_t* p_env_fns,
                                                 cudaStream_t stream);

  wholegraph_error_code_t set_optimizer(wholegraph_embedding_optimizer_t opt);

  [[nodiscard]] const char* const* get_optimizer_state_names() const noexcept
  {
    if (optimizer_impl_base_ != nullptr) {
      return optimizer_impl_base_->get_optimizer_state_names();
    }
    return nullptr;
  }
  virtual wholegraph_tensor_t get_optimizer_state(const char* state_name) const noexcept
  {
    if (optimizer_impl_base_ != nullptr) {
      return optimizer_impl_base_->get_optimizer_state(optimizer_state_.get(), state_name);
    }
    return nullptr;
  }
  virtual wholegraph_error_code_t writeback_embedding_cache(cudaStream_t stream) const noexcept;
  virtual wholegraph_error_code_t writeback_all_caches(cudaStream_t stream) const noexcept;
  virtual wholegraph_error_code_t drop_embedding_cache(cudaStream_t stream) const noexcept;
  virtual wholegraph_error_code_t drop_all_caches(cudaStream_t stream) const noexcept;

  wholegraph::embedding_cache_base* get_cache_ptr() const { return cache_ptr_; }
  wholegraph_error_code_t set_shard_method(
    wholegraph_matrix_description_t* embedding_matrix_description,
    int embedding_world_size,
    int round_robin_size) noexcept;
  wholegraph_error_code_t set_gather_sms(int sms) noexcept;
  int get_round_robin_size() noexcept;

 protected:
  virtual wholegraph_error_code_t init_optimizer_states() noexcept
  {
    if (optimizer_impl_base_ != nullptr) {
      WHOLEGRAPH_RETURN_ON_FAIL(
        optimizer_impl_base_->init_optimizer_states(optimizer_state_.get()));
      WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_barrier(raw_embedding_comm_));
      return WHOLEGRAPH_SUCCESS;
    }
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  wholegraph_error_code_t create_optimizer_states() noexcept;
  wholegraph_error_code_t destroy_optimizer_states() noexcept;

  int gather_sms_;
  int round_robin_size_;
  wholegraph_dtype_t embedding_dtype_                             = WHOLEGRAPH_DT_UNKNOWN;
  wholegraph_comm_t raw_embedding_comm_                           = nullptr;
  wholegraph::embedding_cache_base* cache_ptr_                    = nullptr;
  wholegraph::embedding_optimizer_impl_base* optimizer_impl_base_ = nullptr;
  std::unique_ptr<wholegraph::optimizer_state_t> optimizer_state_ = nullptr;
};

}  // namespace wholegraph
