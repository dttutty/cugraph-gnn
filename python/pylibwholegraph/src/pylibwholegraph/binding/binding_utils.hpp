/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <wholegraph/embedding.h>
#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/graph_op.h>
#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>
#include <wholegraph/wholegraph_op.h>
#include <wholegraph/wholegraph_tensor.h>
#include <wholegraph/wholegraph_tensor_op.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

enum class WholeGraphViewTypeNB {
  VtNone   = 0,
  VtLocal  = 1,
  VtGlobal = 2,
  VtRemote = 3,
};

enum class DLDeviceTypeNB {
  kDLCPU      = 1,
  kDLCUDA     = 2,
  kDLCUDAHost = 3,
};

enum class WholeGraphMemoryAllocTypeNB {
  MatNone   = 0,
  MatDevice = 1,
  MatHost   = 2,
  MatPinned = 3,
};

enum class DLDataTypeCodeNB : uint8_t {
  kDLInt    = 0,
  kDLUInt   = 1,
  kDLFloat  = 2,
  kDLBfloat = 4,
};

struct DLDeviceNB {
  int device_type;
  int device_id;
};

struct DLDataTypeNB {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensorNB {
  void* data;
  DLDeviceNB device;
  int ndim;
  DLDataTypeNB dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};

struct DLManagedTensorNB {
  DLTensorNB dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensorNB*);
};

void check_wholegraph_error_code(wholegraph_error_code_t err);
const char* get_type_string(wholegraph_dtype_t data_type);
void init(unsigned int flags, LogLevel log_level);
void finalize();

void dlpack_tensor_deleter(DLManagedTensorNB* tensor);
using DLManagedTensorOwner = std::unique_ptr<DLManagedTensorNB, decltype(&dlpack_tensor_deleter)>;
DLManagedTensorOwner make_dlmanaged_tensor();
nb::object make_dlpack_capsule(DLManagedTensorOwner dlm_tensor);
DLDataTypeNB dlpack_data_type(wholegraph_dtype_t data_type, size_t itemsize);

std::vector<size_t> size_vector_from_iterable(nb::object values);
std::vector<std::string> strings_from_file_list(nb::object file_list);
int64_t python_shape_at(nb::object const& tensor, size_t index);
nb::object python_slice(int64_t start, int64_t stop);
nb::tuple int64_tuple_from_values(int64_t const* values, int dim);
PyObject* python_none();
bool all_are_callable(std::initializer_list<nb::handle> objects);

template <typename T>
PyObject* retained_python_owner(T const& value)
{
  nb::object owner = nb::find(value);
  if (!owner.is_valid()) {
    throw std::runtime_error("Could not find Python owner for DLPack export");
  }
  return owner.release().ptr();
}

class GlobalContextWrapperNB;

void python_cb_wrapper_temp_create_context(void** memory_context, void* global_context) noexcept;
void python_cb_wrapper_temp_destroy_context(void* memory_context, void* global_context) noexcept;
void* python_cb_wrapper_temp_malloc(wholegraph_tensor_description_t* tensor_desc,
                                    wholegraph_memory_allocation_type_t malloc_type,
                                    void* memory_context,
                                    void* global_context) noexcept;
void python_cb_wrapper_temp_free(void* memory_context, void* global_context) noexcept;
void* python_cb_wrapper_output_malloc(wholegraph_tensor_description_t* tensor_desc,
                                      wholegraph_memory_allocation_type_t malloc_type,
                                      void* memory_context,
                                      void* global_context) noexcept;
void python_cb_wrapper_output_free(void* memory_context, void* global_context) noexcept;
