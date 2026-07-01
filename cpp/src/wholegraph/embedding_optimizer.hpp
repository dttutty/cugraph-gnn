/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/embedding.h>

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "embedding_cache.hpp"

#ifdef __cplusplus
extern "C" {
#endif

struct wholegraph_embedding_optimizer_ {
  wholegraph_optimizer_type_t optimizer_type;
};

#ifdef __cplusplus
}
#endif

namespace wholegraph {

class embedding_optimizer_impl_base;

using optimizer_parameter_setter_fn_t = std::function<wholegraph_error_code_t(const void*)>;

class optimizer_state_t {
 public:
  optimizer_state_t()  = default;
  ~optimizer_state_t() = default;
  // Per element optimizer states are cachable, like momentums.
  // They are packed into same
  struct cachable_state {
    // name of this state
    std::string name;
    int start_dim;
    int dim;
    wholegraph_tensor_t global_raw_state_tensor = nullptr;
  };
  wholegraph_embedding_t cachable_state_embedding = nullptr;

  // wholegraph_tensor_t global_cachable_raw_padded_tensor = nullptr;
  wholegraph_tensor_t global_cachable_raw_user_tensor = nullptr;
  wholegraph_tensor_t local_cachable_wg_tensor        = nullptr;

  // wholegraph_tensor_t global_cacheline_tag_wg_tensor    = nullptr;
  // wholegraph_tensor_t global_cacheline_data_wg_tensor   = nullptr;
  // wholegraph_tensor_t local_cacheline_tag_wg_tensor     = nullptr;
  // wholegraph_tensor_t local_cacheline_data_wg_tensor    = nullptr;
  //  per embedding optimizers are uncachable, like betat1 and batat2 for momentums.
  struct uncachable_state {
    std::string name;
    int dim;
    wholegraph_dtype_t dtype;
    wholegraph_tensor_t global_raw_padded_tensor = nullptr;
    wholegraph_tensor_t global_raw_sub_tensor    = nullptr;
    wholegraph_tensor_t local_tensor             = nullptr;
  };
  int64_t local_start_index                     = -1;
  device_cache_for_host* device_cache_for_host_ = nullptr;
  std::vector<cachable_state> cachable_states;
  std::vector<uncachable_state> uncachable_states;
};

class embedding_optimizer_impl_base : public wholegraph_embedding_optimizer_ {
 public:
  embedding_optimizer_impl_base();
  virtual ~embedding_optimizer_impl_base() = default;
  virtual wholegraph_error_code_t set_parameter(const char* parameter_name, void* value) noexcept;
  /**
   * Apply gradients.
   * As trainable Embedding use READWRITE cache, Cache communicator is the same as Embedding
   * communicator. Gradients will be partitioned and each rank is only responsible for its own
   * partition.
   *
   * @param indices : bucketed indices that belongs to current rank.
   * @param grads : bucketed gradients that belongs to current rank.
   * @param local_embedding : local embedding of current rank.
   * @param optimizer_state : pointer to optimizer state.
   * @param lr : learning rate
   * @param stream : cudaStream_t to use
   * @return : wholegraph_error_code_t
   */
  virtual wholegraph_error_code_t step(wholegraph_tensor_t indices,
                                        wholegraph_tensor_t grads,
                                        wholegraph_tensor_t local_embedding,
                                        optimizer_state_t* optimizer_state,
                                        float lr,
                                        cudaStream_t stream) noexcept = 0;

  virtual void create_optimizer_states(optimizer_state_t* optimizer_state,
                                       int embedding_dim) noexcept
  {
  }

  virtual wholegraph_error_code_t init_optimizer_states(
    optimizer_state_t* optimizer_state) noexcept
  {
    return WHOLEGRAPH_SUCCESS;
  }
  [[nodiscard]] const char* const* get_optimizer_state_names() const noexcept
  {
    return state_names_.data();
  }
  virtual wholegraph_tensor_t get_optimizer_state(optimizer_state_t* optimizer_state,
                                                   const char* state_name);

 protected:
  static optimizer_parameter_setter_fn_t get_float_setter(float* target_ptr);
  static void zero_local_state_tensor(wholegraph_tensor_t local_state_tensor);
  static void set_float_local_state_tensor(wholegraph_tensor_t local_state_tensor, float value);

  std::map<std::string, optimizer_parameter_setter_fn_t> setter_fns_;
  const char* name_ = nullptr;

  std::vector<const char*> state_names_ = {nullptr};
};

wholegraph_error_code_t create_embedding_optimizer(
  wholegraph_embedding_optimizer_t* optimizer,
  wholegraph_optimizer_type_t optimizer_type) noexcept;

wholegraph_error_code_t optimizer_set_parameter(wholegraph_embedding_optimizer_t optimizer,
                                                 const char* parameter_name,
                                                 void* value) noexcept;

void destroy_embedding_optimizer(wholegraph_embedding_optimizer_t optimizer) noexcept;

}  // namespace wholegraph
