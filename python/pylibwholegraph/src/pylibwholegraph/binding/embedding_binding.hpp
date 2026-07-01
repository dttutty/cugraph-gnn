/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "communicator_binding.hpp"
#include "tensor_binding.hpp"

class WholeGraphCachePolicyNB {
 public:
  WholeGraphCachePolicyNB() = default;

  WholeGraphCachePolicyNB(WholeGraphCachePolicyNB const&)            = delete;
  WholeGraphCachePolicyNB& operator=(WholeGraphCachePolicyNB const&) = delete;

  WholeGraphCachePolicyNB(WholeGraphCachePolicyNB&& other) noexcept
    : cache_policy_(std::exchange(other.cache_policy_, nullptr))
  {
  }

  WholeGraphCachePolicyNB& operator=(WholeGraphCachePolicyNB&& other) noexcept
  {
    if (this != &other) {
      release_noexcept();
      cache_policy_ = std::exchange(other.cache_policy_, nullptr);
    }
    return *this;
  }

  ~WholeGraphCachePolicyNB() { release_noexcept(); }

  void create_policy(PyWholeGraphCommNB const& cache_comm,
                     wholegraph_memory_type_t memory_type,
                     wholegraph_memory_location_t memory_location,
                     wholegraph_access_type_t access_type,
                     float ratio)
  {
    destroy_policy();
    check_wholegraph_error_code(wholegraph_create_embedding_cache_policy(&cache_policy_,
                                                                         cache_comm.c_handle(),
                                                                         memory_type,
                                                                         memory_location,
                                                                         access_type,
                                                                         ratio));
  }

  void destroy_policy()
  {
    if (cache_policy_ != nullptr) {
      check_wholegraph_error_code(wholegraph_destroy_embedding_cache_policy(cache_policy_));
      cache_policy_ = nullptr;
    }
  }

  wholegraph_embedding_cache_policy_t c_handle() const { return cache_policy_; }

 private:
  void release_noexcept() noexcept
  {
    if (cache_policy_ != nullptr) {
      static_cast<void>(wholegraph_destroy_embedding_cache_policy(cache_policy_));
      cache_policy_ = nullptr;
    }
  }

  wholegraph_embedding_cache_policy_t cache_policy_ = nullptr;
};

class PyWholeGraphEmbeddingNB {
 public:
  PyWholeGraphEmbeddingNB() = default;

  static PyWholeGraphEmbeddingNB from_c_handle(wholegraph_embedding_t embedding)
  {
    PyWholeGraphEmbeddingNB result;
    result.embedding_ = embedding;
    return result;
  }

  int64_t get_c_handle() const { return reinterpret_cast<int64_t>(embedding_); }

  PyWholeGraphTensorNB get_embedding_tensor() const
  {
    return PyWholeGraphTensorNB::from_c_handle(
      wholegraph_embedding_get_embedding_tensor(embedding_));
  }

  nb::list get_optimizer_state_names() const
  {
    nb::list result;
    char const* const* names = wholegraph_embedding_get_optimizer_state_names(embedding_);
    if (names == nullptr) { return result; }
    for (size_t i = 0; names[i] != nullptr; ++i) {
      result.append(names[i]);
    }
    return result;
  }

  PyWholeGraphTensorNB get_optimizer_state(std::string const& name) const
  {
    return PyWholeGraphTensorNB::from_c_handle(
      wholegraph_embedding_get_optimizer_state(embedding_, name.c_str()));
  }

  void writeback_all_cache(int64_t stream_int) const
  {
    check_wholegraph_error_code(wholegraph_embedding_writeback_cache(embedding_, stream_int));
  }

  void drop_all_cache(int64_t stream_int) const
  {
    check_wholegraph_error_code(wholegraph_embedding_drop_all_cache(embedding_, stream_int));
  }

  void destroy_embedding()
  {
    if (embedding_ != nullptr) {
      check_wholegraph_error_code(wholegraph_destroy_embedding(embedding_));
      embedding_ = nullptr;
    }
  }

  wholegraph_embedding_t c_handle() const { return embedding_; }

 private:
  wholegraph_embedding_t embedding_ = nullptr;
};

class WholeGraphOptimizerNB {
 public:
  WholeGraphOptimizerNB() = default;

  WholeGraphOptimizerNB(WholeGraphOptimizerNB const&)            = delete;
  WholeGraphOptimizerNB& operator=(WholeGraphOptimizerNB const&) = delete;

  WholeGraphOptimizerNB(WholeGraphOptimizerNB&& other) noexcept
    : optimizer_(std::exchange(other.optimizer_, nullptr))
  {
  }

  WholeGraphOptimizerNB& operator=(WholeGraphOptimizerNB&& other) noexcept
  {
    if (this != &other) {
      destroy_optimizer();
      optimizer_ = std::exchange(other.optimizer_, nullptr);
    }
    return *this;
  }

  ~WholeGraphOptimizerNB() { destroy_optimizer(); }

  void create_optimizer(wholegraph_optimizer_type_t optimizer_type, nb::dict param_dict)
  {
    destroy_optimizer();
    check_wholegraph_error_code(
      wholegraph_create_embedding_optimizer(&optimizer_, optimizer_type));

    for (auto [key, value] : param_dict) {
      std::string name = nb::cast<std::string>(key);
      float parameter  = nb::cast<float>(value);
      check_wholegraph_error_code(
        wholegraph_optimizer_set_parameter(optimizer_, name.c_str(), &parameter));
    }
  }

  void add_embedding(PyWholeGraphEmbeddingNB const& embedding)
  {
    check_wholegraph_error_code(
      wholegraph_embedding_set_optimizer(embedding.c_handle(), optimizer_));
  }

  void destroy_optimizer()
  {
    if (optimizer_ != nullptr) {
      wholegraph_destroy_embedding_optimizer(optimizer_);
      optimizer_ = nullptr;
    }
  }

  wholegraph_embedding_optimizer_t c_handle() const { return optimizer_; }

 private:
  wholegraph_embedding_optimizer_t optimizer_ = nullptr;
};
