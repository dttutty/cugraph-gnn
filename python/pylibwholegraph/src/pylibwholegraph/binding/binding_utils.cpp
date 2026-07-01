/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binding_utils.hpp"

namespace {
constexpr char const* dlpack_capsule_name = "dltensor";

void pycapsule_deleter(PyObject* capsule)
{
  if (!PyCapsule_IsValid(capsule, dlpack_capsule_name)) { return; }
  auto* dlm_tensor =
    static_cast<DLManagedTensorNB*>(PyCapsule_GetPointer(capsule, dlpack_capsule_name));
  if (dlm_tensor != nullptr && dlm_tensor->deleter != nullptr) {
    dlm_tensor->deleter(dlm_tensor);
  }
}
}  // namespace

void dlpack_tensor_deleter(DLManagedTensorNB* tensor)
{
  if (tensor == nullptr) { return; }
  PyObject* owner = static_cast<PyObject*>(tensor->manager_ctx);
  if (owner != nullptr) {
    nb::gil_scoped_acquire gil;
    nb::handle(owner).dec_ref();
  }
  delete tensor;
}

DLManagedTensorOwner make_dlmanaged_tensor()
{
  return DLManagedTensorOwner{new DLManagedTensorNB{}, dlpack_tensor_deleter};
}

nb::object make_dlpack_capsule(DLManagedTensorOwner dlm_tensor)
{
  // DLPack consumers signal ownership transfer by renaming the capsule to
  // "used_dltensor", so this intentionally uses the Python capsule API instead
  // of nanobind::capsule's generic cleanup path.
  PyObject* capsule = PyCapsule_New(dlm_tensor.get(), dlpack_capsule_name, pycapsule_deleter);
  if (capsule == nullptr) { nb::raise_python_error(); }
  dlm_tensor.release();
  return nb::steal<nb::object>(capsule);
}

DLDataTypeNB dlpack_data_type(wholegraph_dtype_t data_type, size_t itemsize)
{
  DLDataTypeNB dtype{};
  dtype.lanes = 1;
  dtype.bits  = static_cast<uint8_t>(itemsize * 8);
  switch (data_type) {
    case WHOLEGRAPH_DT_INT:
    case WHOLEGRAPH_DT_INT64:
    case WHOLEGRAPH_DT_INT16:
    case WHOLEGRAPH_DT_INT8: dtype.code = static_cast<uint8_t>(DLDataTypeCodeNB::kDLInt); break;
    case WHOLEGRAPH_DT_FLOAT:
    case WHOLEGRAPH_DT_DOUBLE:
    case WHOLEGRAPH_DT_HALF: dtype.code = static_cast<uint8_t>(DLDataTypeCodeNB::kDLFloat); break;
    case WHOLEGRAPH_DT_BF16: dtype.code = static_cast<uint8_t>(DLDataTypeCodeNB::kDLBfloat); break;
    default: throw std::invalid_argument("Invalid data_type");
  }
  return dtype;
}

std::vector<size_t> size_vector_from_iterable(nb::object values)
{
  std::vector<size_t> result;
  if (values.is_none()) { return result; }
  for (nb::handle value : values) {
    result.push_back(nb::cast<size_t>(value));
  }
  return result;
}

std::vector<std::string> strings_from_file_list(nb::object file_list)
{
  std::vector<std::string> filenames;
  if (nb::isinstance<nb::str>(file_list)) {
    filenames.push_back(nb::cast<std::string>(file_list));
    return filenames;
  }
  for (nb::handle filename : file_list) {
    filenames.push_back(nb::cast<std::string>(filename));
  }
  return filenames;
}

int64_t python_shape_at(nb::object const& tensor, size_t index)
{
  nb::object shape = tensor.attr("shape");
  return nb::cast<int64_t>(shape.attr("__getitem__")(index));
}

nb::object python_slice(int64_t start, int64_t stop)
{
  return nb::module_::import_("builtins").attr("slice")(start, stop);
}

nb::tuple int64_tuple_from_values(int64_t const* values, int dim)
{
  std::vector<int64_t> tuple_values;
  tuple_values.reserve(static_cast<size_t>(dim));
  for (int i = 0; i < dim; ++i) {
    tuple_values.push_back(values[i]);
  }
  return nb::tuple(nb::cast(tuple_values));
}

PyObject* python_none()
{
  return Py_None;
}

bool all_are_callable(std::initializer_list<nb::handle> objects)
{
  return std::all_of(objects.begin(), objects.end(), [](nb::handle object) {
    return nb::isinstance<nb::callable>(object);
  });
}

void check_wholegraph_error_code(wholegraph_error_code_t err)
{
  switch (err) {
    case WHOLEGRAPH_SUCCESS: return;
    case WHOLEGRAPH_UNKNOW_ERROR: throw std::runtime_error("Unknown error");
    case WHOLEGRAPH_NOT_IMPLEMENTED: throw std::runtime_error("Not implemented");
    case WHOLEGRAPH_LOGIC_ERROR: throw std::runtime_error("Logic error");
    case WHOLEGRAPH_CUDA_ERROR: throw std::runtime_error("CUDA error");
    case WHOLEGRAPH_COMMUNICATION_ERROR: throw std::runtime_error("Communication error");
    case WHOLEGRAPH_INVALID_INPUT: throw std::invalid_argument("Invalid input");
    case WHOLEGRAPH_INVALID_VALUE: throw std::invalid_argument("Invalid value");
    case WHOLEGRAPH_OUT_OF_MEMORY: throw std::bad_alloc();
    default: throw std::runtime_error("Error code not recognized");
  }
}

void init(unsigned int flags, LogLevel log_level)
{
  check_wholegraph_error_code(wholegraph_init(flags, log_level));
}

void finalize() { check_wholegraph_error_code(wholegraph_finalize()); }

const char* get_type_string(wholegraph_dtype_t data_type)
{
  switch (data_type) {
    case WHOLEGRAPH_DT_FLOAT: return "<f4";
    case WHOLEGRAPH_DT_HALF: return "<f2";
    case WHOLEGRAPH_DT_DOUBLE: return "<f8";
    case WHOLEGRAPH_DT_BF16: return "<f2";
    case WHOLEGRAPH_DT_INT: return "<i4";
    case WHOLEGRAPH_DT_INT64: return "<i8";
    case WHOLEGRAPH_DT_INT16: return "<i2";
    case WHOLEGRAPH_DT_INT8: return "|i1";
    default: throw std::invalid_argument("data type not valid");
  }
}
