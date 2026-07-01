/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "binding_utils.hpp"

class PyMemoryAllocTypeNB {
 public:
  PyMemoryAllocTypeNB() = default;

  void set_type(WholeGraphMemoryAllocTypeNB new_type)
  {
    alloc_type_ =
      static_cast<wholegraph_memory_allocation_type_t>(static_cast<int>(new_type));
  }

  int64_t get_type() const { return static_cast<int64_t>(alloc_type_); }

  void set_ctype(int64_t alloc_type)
  {
    alloc_type_ = static_cast<wholegraph_memory_allocation_type_t>(alloc_type);
  }

  int64_t get_ctype() const { return static_cast<int64_t>(alloc_type_); }

 private:
  wholegraph_memory_allocation_type_t alloc_type_ = WHOLEGRAPH_MA_NONE;
};

class GlobalContextWrapperNB {
 public:
  GlobalContextWrapperNB() = default;

  GlobalContextWrapperNB(GlobalContextWrapperNB const&)            = delete;
  GlobalContextWrapperNB& operator=(GlobalContextWrapperNB const&) = delete;
  GlobalContextWrapperNB(GlobalContextWrapperNB&&)                 = delete;
  GlobalContextWrapperNB& operator=(GlobalContextWrapperNB&&)      = delete;

  ~GlobalContextWrapperNB() { release_python_objects(); }

  void create_context(nb::object temp_create_context_fn,
                      nb::object temp_destroy_context_fn,
                      nb::object temp_malloc_fn,
                      nb::object temp_free_fn,
                      nb::object temp_global_context,
                      nb::object output_malloc_fn,
                      nb::object output_free_fn,
                      nb::object output_global_context)
  {
    if (!all_are_callable({temp_create_context_fn,
                           temp_destroy_context_fn,
                           temp_malloc_fn,
                           temp_free_fn,
                           output_malloc_fn,
                           output_free_fn})) {
      throw std::invalid_argument("GlobalContextWrapper callbacks must be callable");
    }

    temp_create_context_fn_  = std::move(temp_create_context_fn);
    temp_destroy_context_fn_ = std::move(temp_destroy_context_fn);
    temp_malloc_fn_          = std::move(temp_malloc_fn);
    temp_free_fn_            = std::move(temp_free_fn);
    temp_global_context_     = std::move(temp_global_context);
    output_malloc_fn_        = std::move(output_malloc_fn);
    output_free_fn_          = std::move(output_free_fn);
    output_global_context_   = std::move(output_global_context);

    env_func_.temporary_fns.create_memory_context_fn = python_cb_wrapper_temp_create_context;
    env_func_.temporary_fns.destroy_memory_context_fn = python_cb_wrapper_temp_destroy_context;
    env_func_.temporary_fns.malloc_fn                 = python_cb_wrapper_temp_malloc;
    env_func_.temporary_fns.free_fn                   = python_cb_wrapper_temp_free;
    env_func_.temporary_fns.global_context            = this;
    env_func_.output_fns.malloc_fn                    = python_cb_wrapper_output_malloc;
    env_func_.output_fns.free_fn                      = python_cb_wrapper_output_free;
    env_func_.output_fns.global_context               = this;
  }

  int64_t get_env_fns()
  {
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(&env_func_));
  }

  static PyObject* memory_context_object(void* memory_context)
  {
    return memory_context == nullptr ? python_none() : static_cast<PyObject*>(memory_context);
  }

 private:
  friend void python_cb_wrapper_temp_create_context(void** memory_context,
                                                    void* global_context) noexcept;
  friend void python_cb_wrapper_temp_destroy_context(void* memory_context,
                                                     void* global_context) noexcept;
  friend void* python_cb_wrapper_temp_malloc(wholegraph_tensor_description_t* tensor_desc,
                                             wholegraph_memory_allocation_type_t malloc_type,
                                             void* memory_context,
                                             void* global_context) noexcept;
  friend void python_cb_wrapper_temp_free(void* memory_context, void* global_context) noexcept;
  friend void* python_cb_wrapper_output_malloc(wholegraph_tensor_description_t* tensor_desc,
                                               wholegraph_memory_allocation_type_t malloc_type,
                                               void* memory_context,
                                               void* global_context) noexcept;
  friend void python_cb_wrapper_output_free(void* memory_context, void* global_context) noexcept;

  PyObject* temp_global_context() const
  {
    return temp_global_context_.is_valid() ? temp_global_context_.ptr() : python_none();
  }

  PyObject* output_global_context() const
  {
    return output_global_context_.is_valid() ? output_global_context_.ptr() : python_none();
  }

  void release_python_objects()
  {
    nb::gil_scoped_acquire gil;
    temp_create_context_fn_.reset();
    temp_destroy_context_fn_.reset();
    temp_malloc_fn_.reset();
    temp_free_fn_.reset();
    temp_global_context_.reset();
    output_malloc_fn_.reset();
    output_free_fn_.reset();
    output_global_context_.reset();
  }

  nb::object temp_create_context_fn_;
  nb::object temp_destroy_context_fn_;
  nb::object temp_malloc_fn_;
  nb::object temp_free_fn_;
  nb::object temp_global_context_;
  nb::object output_malloc_fn_;
  nb::object output_free_fn_;
  nb::object output_global_context_;
  wholegraph_env_func_t env_func_{};
};
