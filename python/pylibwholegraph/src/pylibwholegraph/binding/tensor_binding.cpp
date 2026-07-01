/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "wholegraph_binding.hpp"

nb::tuple PyWholeGraphHandleNB::get_global_flatten_tensor(
  nb::callable import_dlpack_fn,
  wholegraph_dtype_t data_type,
  wholegraph_memory_location_t view_from_device,
  int view_from_device_id) const
{
  PyWholeGraphFlattenDlpackNB flatten_tensor;
  flatten_tensor.set_view_device(view_from_device, view_from_device_id);
  auto view =
    flatten_tensor.get_view(*this, data_type, WholeGraphViewTypeNB::VtGlobal, 0);
  size_t const element_offset = view.second;
  if (element_offset != 0) { throw std::runtime_error("global view offset must be zero"); }
  nb::object flatten_tensor_object = nb::cast(std::move(flatten_tensor));
  return nb::make_tuple(import_dlpack_fn(flatten_tensor_object), element_offset);
}

nb::tuple PyWholeGraphHandleNB::get_local_flatten_tensor(
  nb::callable import_dlpack_fn,
  wholegraph_dtype_t data_type,
  wholegraph_memory_location_t view_from_device,
  int view_from_device_id) const
{
  PyWholeGraphFlattenDlpackNB flatten_tensor;
  flatten_tensor.set_view_device(view_from_device, view_from_device_id);
  auto view =
    flatten_tensor.get_view(*this, data_type, WholeGraphViewTypeNB::VtLocal, 0);
  size_t const element_offset = view.second;
  nb::object flatten_tensor_object = nb::cast(std::move(flatten_tensor));
  return nb::make_tuple(import_dlpack_fn(flatten_tensor_object), element_offset);
}


PyWholeGraphHandleNB malloc_wholegraph(size_t total_size,
                                       PyWholeGraphCommNB const& comm,
                                       wholegraph_memory_type_t memory_type,
                                       wholegraph_memory_location_t memory_location,
                                       size_t data_granularity,
                                       nb::object rank_entry_partition)
{
  wholegraph_handle_t handle = nullptr;
  std::vector<size_t> partition = size_vector_from_iterable(rank_entry_partition);
  check_wholegraph_error_code(wholegraph_malloc(&handle,
                                                total_size,
                                                comm.c_handle(),
                                                memory_type,
                                                memory_location,
                                                data_granularity,
                                                partition.empty() ? nullptr : partition.data()));
  return PyWholeGraphHandleNB::from_c_handle(handle);
}

void free_handle(PyWholeGraphHandleNB const& handle)
{
  check_wholegraph_error_code(wholegraph_free(handle.c_handle()));
}

PyWholeGraphTensorNB create_wholegraph_array(wholegraph_dtype_t dtype,
                                             int64_t size,
                                             PyWholeGraphCommNB const& comm,
                                             wholegraph_memory_type_t memory_type,
                                             wholegraph_memory_location_t memory_location,
                                             nb::object tensor_entry_partition)
{
  wholegraph_tensor_description_t tensor_description{};
  wholegraph_initialize_tensor_desc(&tensor_description);
  tensor_description.dtype     = dtype;
  tensor_description.dim       = 1;
  tensor_description.sizes[0]  = size;
  tensor_description.strides[0] = 1;

  wholegraph_tensor_t tensor = nullptr;
  std::vector<size_t> partition = size_vector_from_iterable(tensor_entry_partition);
  check_wholegraph_error_code(wholegraph_create_tensor(
    &tensor,
    &tensor_description,
    comm.c_handle(),
    memory_type,
    memory_location,
    partition.empty() ? nullptr : partition.data()));
  return PyWholeGraphTensorNB::from_c_handle(tensor);
}

PyWholeGraphTensorNB create_wholegraph_matrix(wholegraph_dtype_t dtype,
                                              int64_t row,
                                              int64_t column,
                                              int64_t stride,
                                              PyWholeGraphCommNB const& comm,
                                              wholegraph_memory_type_t memory_type,
                                              wholegraph_memory_location_t memory_location,
                                              nb::object tensor_entry_partition)
{
  wholegraph_tensor_description_t tensor_description{};
  wholegraph_initialize_tensor_desc(&tensor_description);
  tensor_description.dtype      = dtype;
  tensor_description.dim        = 2;
  tensor_description.sizes[0]   = row;
  tensor_description.sizes[1]   = column;
  tensor_description.strides[0] = stride == -1 ? column : stride;
  tensor_description.strides[1] = 1;

  wholegraph_tensor_t tensor = nullptr;
  std::vector<size_t> partition = size_vector_from_iterable(tensor_entry_partition);
  check_wholegraph_error_code(wholegraph_create_tensor(
    &tensor,
    &tensor_description,
    comm.c_handle(),
    memory_type,
    memory_location,
    partition.empty() ? nullptr : partition.data()));
  return PyWholeGraphTensorNB::from_c_handle(tensor);
}

void check_tensor_description_for_create(PyWholeGraphTensorDescriptionNB const& tensor_description)
{
  int const dim = tensor_description.dim();
  if (dim != 1 && dim != 2) {
    throw std::runtime_error("WholeGraph currently only support 1D or 2D tensor");
  }
  auto const* desc = tensor_description.c_ptr();
  if (desc->strides[dim - 1] != 1) { throw std::invalid_argument("last stride should be 1"); }
  if (desc->storage_offset != 0) {
    throw std::invalid_argument("storage_offset be 0 when created");
  }
}

PyWholeGraphTensorNB create_wholegraph_tensor(
  PyWholeGraphTensorDescriptionNB const& tensor_description,
  PyWholeGraphCommNB const& comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  nb::object tensor_entry_partition)
{
  check_tensor_description_for_create(tensor_description);
  wholegraph_tensor_description_t desc = *tensor_description.c_ptr();
  wholegraph_tensor_t tensor           = nullptr;
  std::vector<size_t> partition        = size_vector_from_iterable(tensor_entry_partition);
  check_wholegraph_error_code(wholegraph_create_tensor(
    &tensor,
    &desc,
    comm.c_handle(),
    memory_type,
    memory_location,
    partition.empty() ? nullptr : partition.data()));
  return PyWholeGraphTensorNB::from_c_handle(tensor);
}

PyWholeGraphTensorNB make_tensor_as_wholegraph(
  PyWholeGraphTensorDescriptionNB const& tensor_description, int64_t data_ptr)
{
  if (tensor_description.c_ptr()->strides[tensor_description.dim() - 1] != 1) {
    throw std::invalid_argument("last stride should be 1");
  }
  wholegraph_tensor_description_t desc = *tensor_description.c_ptr();
  wholegraph_tensor_t tensor           = nullptr;
  check_wholegraph_error_code(wholegraph_make_tensor_from_pointer(
    &tensor, reinterpret_cast<void*>(static_cast<uintptr_t>(data_ptr)), &desc));
  return PyWholeGraphTensorNB::from_c_handle(tensor);
}

PyWholeGraphTensorNB make_handle_as_wholegraph(
  PyWholeGraphTensorDescriptionNB const& tensor_description, PyWholeGraphHandleNB const& handle)
{
  if (tensor_description.c_ptr()->strides[tensor_description.dim() - 1] != 1) {
    throw std::invalid_argument("last stride should be 1");
  }
  wholegraph_tensor_description_t desc = *tensor_description.c_ptr();
  wholegraph_tensor_t tensor           = nullptr;
  check_wholegraph_error_code(
    wholegraph_make_tensor_from_handle(&tensor, handle.c_handle(), &desc));
  return PyWholeGraphTensorNB::from_c_handle(tensor);
}

void destroy_wholegraph_tensor(PyWholeGraphTensorNB const& tensor)
{
  check_wholegraph_error_code(wholegraph_destroy_tensor(tensor.c_handle()));
}


void load_wholegraph_handle_from_filelist(int64_t wholegraph_handle_int_ptr,
                                          size_t memory_offset,
                                          size_t memory_entry_size,
                                          size_t file_entry_size,
                                          int round_robin_size,
                                          nb::object file_list)
{
  auto handle = PyWholeGraphHandleNB::from_c_handle(reinterpret_cast<wholegraph_handle_t>(
    static_cast<uintptr_t>(wholegraph_handle_int_ptr)));
  handle.from_filelist(
    memory_offset, memory_entry_size, file_entry_size, round_robin_size, file_list);
}

void store_wholegraph_handle_to_file(int64_t wholegraph_handle_int_ptr,
                                     size_t memory_offset,
                                     size_t memory_entry_size,
                                     size_t file_entry_size,
                                     std::string const& file_name)
{
  auto handle = PyWholeGraphHandleNB::from_c_handle(reinterpret_cast<wholegraph_handle_t>(
    static_cast<uintptr_t>(wholegraph_handle_int_ptr)));
  handle.to_file(memory_offset, memory_entry_size, file_entry_size, file_name);
}
