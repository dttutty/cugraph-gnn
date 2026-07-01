/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "communicator_binding.hpp"

class PyWholeGraphTensorDescriptionNB {
 public:
  PyWholeGraphTensorDescriptionNB() { wholegraph_initialize_tensor_desc(&tensor_description_); }

  void set_dtype(wholegraph_dtype_t dtype) { tensor_description_.dtype = dtype; }

  void set_shape(nb::sequence shape)
  {
    size_t const dim = nb::len(shape);
    if (dim == 0 || dim >= WHOLEGRAPH_MAX_TENSOR_DIM) {
      throw std::invalid_argument("shape length must be in range [1, 8)");
    }

    tensor_description_.dim = static_cast<int>(dim);
    for (size_t i = 0; i < dim; ++i) {
      tensor_description_.sizes[i] = nb::cast<int64_t>(shape[i]);
    }
  }

  void set_stride(nb::sequence strides)
  {
    if (tensor_description_.dim == 0) {
      throw std::invalid_argument("shape must be set before stride");
    }
    size_t const dim = nb::len(strides);
    if (dim != static_cast<size_t>(tensor_description_.dim)) {
      throw std::invalid_argument("stride length must match tensor dimension");
    }

    for (size_t i = 0; i < dim; ++i) {
      tensor_description_.strides[i] = nb::cast<int64_t>(strides[i]);
    }
  }

  void set_storage_offset(int64_t storage_offset)
  {
    tensor_description_.storage_offset = storage_offset;
  }

  wholegraph_dtype_t dtype() const { return tensor_description_.dtype; }

  int dim() const { return tensor_description_.dim; }

  nb::tuple shape() const
  {
    return int64_tuple_from_values(tensor_description_.sizes, tensor_description_.dim);
  }

  nb::tuple stride() const
  {
    return int64_tuple_from_values(tensor_description_.strides, tensor_description_.dim);
  }

  int64_t storage_offset() const { return tensor_description_.storage_offset; }

  wholegraph_tensor_description_t* c_ptr() { return &tensor_description_; }

  wholegraph_tensor_description_t const* c_ptr() const { return &tensor_description_; }

 private:
  wholegraph_tensor_description_t tensor_description_{};
};


class WrappedLocalTensorNB {
 public:
  WrappedLocalTensorNB() = default;

  WrappedLocalTensorNB(WrappedLocalTensorNB const&)            = delete;
  WrappedLocalTensorNB& operator=(WrappedLocalTensorNB const&) = delete;

  WrappedLocalTensorNB(WrappedLocalTensorNB&& other) noexcept
    : wg_tensor_(std::exchange(other.wg_tensor_, nullptr))
  {
  }

  WrappedLocalTensorNB& operator=(WrappedLocalTensorNB&& other) noexcept
  {
    if (this != &other) {
      destroy();
      wg_tensor_ = std::exchange(other.wg_tensor_, nullptr);
    }
    return *this;
  }

  ~WrappedLocalTensorNB() { destroy(); }

  WrappedLocalTensorNB& wrap_tensor(PyWholeGraphTensorDescriptionNB const& py_desc,
                                    int64_t data_ptr)
  {
    wholegraph_tensor_description_t desc = *py_desc.c_ptr();
    wholegraph_tensor_t new_tensor       = nullptr;
    check_wholegraph_error_code(wholegraph_make_tensor_from_pointer(
      &new_tensor, reinterpret_cast<void*>(static_cast<uintptr_t>(data_ptr)), &desc));
    destroy();
    wg_tensor_ = new_tensor;
    return *this;
  }

  int64_t get_c_handle() const { return reinterpret_cast<int64_t>(wg_tensor_); }

  wholegraph_tensor_t c_handle() const { return wg_tensor_; }

 private:
  void destroy() noexcept
  {
    if (wg_tensor_ != nullptr) {
      static_cast<void>(wholegraph_destroy_tensor(wg_tensor_));
      wg_tensor_ = nullptr;
    }
  }

  wholegraph_tensor_t wg_tensor_ = nullptr;
};

class PyWholeGraphHandleNB {
 public:
  PyWholeGraphHandleNB() = default;

  static PyWholeGraphHandleNB from_c_handle(wholegraph_handle_t wholegraph_handle)
  {
    PyWholeGraphHandleNB handle;
    handle.wholegraph_handle_ = wholegraph_handle;
    return handle;
  }

  int64_t get_c_handle() const { return reinterpret_cast<int64_t>(wholegraph_handle_); }

  PyWholeGraphCommNB get_communicator() const
  {
    wholegraph_comm_t comm = nullptr;
    check_wholegraph_error_code(wholegraph_get_communicator(&comm, wholegraph_handle_));
    return PyWholeGraphCommNB::from_c_handle(comm);
  }

  wholegraph_memory_type_t get_memory_type() const
  {
    return wholegraph_get_memory_type(wholegraph_handle_);
  }

  wholegraph_memory_location_t get_memory_location() const
  {
    return wholegraph_get_memory_location(wholegraph_handle_);
  }

  nb::tuple get_global_flatten_tensor(nb::callable import_dlpack_fn,
                                      wholegraph_dtype_t data_type,
                                      wholegraph_memory_location_t view_from_device,
                                      int view_from_device_id) const;

  nb::tuple get_local_flatten_tensor(nb::callable import_dlpack_fn,
                                     wholegraph_dtype_t data_type,
                                     wholegraph_memory_location_t view_from_device,
                                     int view_from_device_id) const;

  void from_filelist(size_t memory_offset,
                     size_t memory_entry_size,
                     size_t file_entry_size,
                     int round_robin_size,
                     nb::object file_list) const
  {
    std::vector<std::string> filenames = strings_from_file_list(file_list);
    std::vector<const char*> filename_ptrs;
    filename_ptrs.reserve(filenames.size());
    for (auto const& filename : filenames) {
      filename_ptrs.push_back(filename.c_str());
    }

    check_wholegraph_error_code(wholegraph_load_from_file(wholegraph_handle_,
                                                          memory_offset,
                                                          memory_entry_size,
                                                          file_entry_size,
                                                          filename_ptrs.data(),
                                                          static_cast<int>(filename_ptrs.size()),
                                                          round_robin_size));
  }

  void to_file(size_t memory_offset,
               size_t memory_entry_size,
               size_t file_entry_size,
               std::string const& file_name) const
  {
    check_wholegraph_error_code(wholegraph_store_to_file(wholegraph_handle_,
                                                         memory_offset,
                                                         memory_entry_size,
                                                         file_entry_size,
                                                         file_name.c_str()));
  }

  wholegraph_handle_t c_handle() const { return wholegraph_handle_; }

 private:
  wholegraph_handle_t wholegraph_handle_ = nullptr;
};

class PyWholeGraphFlattenDlpackNB {
 public:
  PyWholeGraphFlattenDlpackNB() = default;

  void set_view_device(wholegraph_memory_location_t device_type, int device_id)
  {
    device_type_ = device_type;
    device_id_   = device_id;
  }

  std::pair<size_t, size_t> get_view(PyWholeGraphHandleNB const& handle,
                                     wholegraph_dtype_t data_type,
                                     WholeGraphViewTypeNB view_type,
                                     int target_rank)
  {
    data_type_ = data_type;
    itemsize_  = wholegraph_dtype_get_element_size(data_type);
    if (itemsize_ == 0 || itemsize_ > 8) {
      throw std::invalid_argument("data_type not supported");
    }
    typestr_ = get_type_string(data_type);

    wholegraph_memory_type_t const mem_type = wholegraph_get_memory_type(handle.c_handle());
    wholegraph_memory_location_t const mem_location =
      wholegraph_get_memory_location(handle.c_handle());
    if (device_type_ == WHOLEGRAPH_ML_HOST && mem_location == WHOLEGRAPH_ML_DEVICE) {
      throw std::invalid_argument("Device WholeGraph cannot get view from host.");
    }
    if (mem_type == WHOLEGRAPH_MT_DISTRIBUTED &&
        (view_type == WholeGraphViewTypeNB::VtGlobal ||
         view_type == WholeGraphViewTypeNB::VtRemote)) {
      throw std::invalid_argument("Distributed WholeGraph have no view of Global or Remote");
    }

    wholegraph_comm_t comm = nullptr;
    if (device_type_ == WHOLEGRAPH_ML_HOST && mem_type == WHOLEGRAPH_MT_CONTINUOUS) {
      check_wholegraph_error_code(wholegraph_get_communicator(&comm, handle.c_handle()));
      if (!wholegraph_is_intranode_communicator(comm)) {
        throw std::invalid_argument(
          "Multi-node continuous type wholegraph does not support host_view. Only supports "
          "host_view=false regardless of whether location is host or not.");
      }
    }

    size_t map_size    = 0;
    size_t map_offset  = 0;
    size_t global_size = wholegraph_get_total_size(handle.c_handle());
    if (global_size % itemsize_ != 0) {
      throw std::invalid_argument("global_size is not multiple of elt_size");
    }
    size_t const global_elt_count = global_size / itemsize_;

    if (view_type == WholeGraphViewTypeNB::VtLocal) {
      check_wholegraph_error_code(
        wholegraph_get_local_memory(&data_ptr_, &map_size, &map_offset, handle.c_handle()));
      if (map_size % itemsize_ != 0 || map_offset % itemsize_ != 0) {
        throw std::invalid_argument("map_size or map_offset is not multiple of elt_size");
      }
      shape_[0] = static_cast<int64_t>(map_size / itemsize_);
      return {map_size / itemsize_, map_offset / itemsize_};
    }

    if (view_type == WholeGraphViewTypeNB::VtGlobal) {
      check_wholegraph_error_code(
        wholegraph_get_global_pointer(&data_ptr_, handle.c_handle()));
      shape_[0] = static_cast<int64_t>(global_elt_count);
      return {global_elt_count, 0};
    }

    if (view_type == WholeGraphViewTypeNB::VtRemote) {
      int world_size = 0;
      check_wholegraph_error_code(wholegraph_get_communicator(&comm, handle.c_handle()));
      check_wholegraph_error_code(wholegraph_communicator_get_size(&world_size, comm));
      if (target_rank < 0 || target_rank >= world_size) {
        throw std::out_of_range("target_rank is outside communicator world_size");
      }
      check_wholegraph_error_code(wholegraph_get_rank_memory(
        &data_ptr_, &map_size, &map_offset, target_rank, handle.c_handle()));
      if (map_size % itemsize_ != 0 || map_offset % itemsize_ != 0) {
        throw std::invalid_argument("rank map_size or map_offset is not multiple of elt_size");
      }
      shape_[0] = static_cast<int64_t>(map_size / itemsize_);
      return {map_size / itemsize_, map_offset / itemsize_};
    }

    throw std::invalid_argument("view type should be VtLocal or VtGlobal or VtRemote");
  }

  size_t length() const { return static_cast<size_t>(shape_[0]); }

  uintptr_t ptr() const { return reinterpret_cast<uintptr_t>(data_ptr_); }

  nb::dict cuda_array_interface() const
  {
    nb::dict intf;
    intf["data"]    = nb::make_tuple(ptr(), false);
    intf["shape"]   = nb::make_tuple(shape_[0]);
    intf["strides"] = nb::none();
    intf["typestr"] = typestr_;
    intf["version"] = 2;
    return intf;
  }

  nb::object dlpack(nb::object /* stream */ = nb::none()) const
  {
    auto dlm_tensor             = make_dlmanaged_tensor();
    dlm_tensor->dl_tensor.data  = data_ptr_;
    dlm_tensor->dl_tensor.ndim  = 1;
    dlm_tensor->dl_tensor.shape = const_cast<int64_t*>(shape_.data());
    dlm_tensor->dl_tensor.strides = const_cast<int64_t*>(strides_.data());
    dlm_tensor->dl_tensor.byte_offset = 0;
    auto device = dlpack_device();
    dlm_tensor->dl_tensor.device.device_type = nb::cast<int>(device[0]);
    dlm_tensor->dl_tensor.device.device_id   = nb::cast<int>(device[1]);
    dlm_tensor->dl_tensor.dtype              = dlpack_data_type(data_type_, itemsize_);
    dlm_tensor->manager_ctx                  = retained_python_owner(*this);
    dlm_tensor->deleter                      = dlpack_tensor_deleter;
    return make_dlpack_capsule(std::move(dlm_tensor));
  }

  nb::tuple dlpack_device() const
  {
    if (device_type_ == WHOLEGRAPH_ML_HOST) {
      return nb::make_tuple(static_cast<int>(DLDeviceTypeNB::kDLCPU), 0);
    }
    if (device_type_ == WHOLEGRAPH_ML_DEVICE) {
      return nb::make_tuple(static_cast<int>(DLDeviceTypeNB::kDLCUDA), device_id_);
    }
    throw std::invalid_argument("Invalid device_type");
  }

 private:
  void* data_ptr_ = nullptr;
  wholegraph_dtype_t data_type_ = WHOLEGRAPH_DT_UNKNOWN;
  size_t itemsize_ = 0;
  std::string typestr_;
  std::array<int64_t, 1> shape_{0};
  std::array<int64_t, 1> strides_{1};
  wholegraph_memory_location_t device_type_ = WHOLEGRAPH_ML_HOST;
  int device_id_ = 0;
};


class PyWholeGraphTensorNB {
 public:
  PyWholeGraphTensorNB() = default;

  static PyWholeGraphTensorNB from_c_handle(wholegraph_tensor_t wholegraph_tensor)
  {
    PyWholeGraphTensorNB tensor;
    tensor.wholegraph_tensor_ = wholegraph_tensor;
    tensor.refresh_description();
    return tensor;
  }

  int64_t get_c_handle() const { return reinterpret_cast<int64_t>(wholegraph_tensor_); }

  PyWholeGraphHandleNB get_wholegraph_handle() const
  {
    return PyWholeGraphHandleNB::from_c_handle(
      wholegraph_tensor_get_memory_handle(wholegraph_tensor_));
  }

  wholegraph_dtype_t dtype() const { return tensor_description_.dtype; }

  int dim() const { return tensor_description_.dim; }

  nb::tuple shape() const
  {
    if (dim() == 1) { return nb::make_tuple(tensor_description_.sizes[0]); }
    if (dim() == 2) {
      return nb::make_tuple(tensor_description_.sizes[0], tensor_description_.sizes[1]);
    }
    throw std::invalid_argument("self.dim() must be 1 or 2");
  }

  nb::tuple stride() const
  {
    if (dim() == 1) { return nb::make_tuple(tensor_description_.strides[0]); }
    if (dim() == 2) {
      return nb::make_tuple(tensor_description_.strides[0], tensor_description_.strides[1]);
    }
    throw std::invalid_argument("self.dim() must be 1 or 2");
  }

  int64_t storage_offset() const { return tensor_description_.storage_offset; }

  size_t get_local_entry_count() const
  {
    size_t local_entry_count = 0;
    check_wholegraph_error_code(
      wholegraph_tensor_get_local_entry_count(&local_entry_count, wholegraph_tensor_));
    return local_entry_count;
  }

  size_t get_local_entry_start() const
  {
    size_t local_entry_start = 0;
    check_wholegraph_error_code(
      wholegraph_tensor_get_local_entry_start(&local_entry_start, wholegraph_tensor_));
    return local_entry_start;
  }

  PyWholeGraphTensorNB get_sub_tensor(nb::sequence starts, nb::sequence ends) const
  {
    if (nb::len(starts) != static_cast<size_t>(dim()) ||
        nb::len(ends) != static_cast<size_t>(dim())) {
      throw std::invalid_argument("starts and ends must match tensor dimension");
    }

    std::array<int64_t, 2> start_array{nb::cast<int64_t>(starts[0]), 0};
    std::array<int64_t, 2> end_array{nb::cast<int64_t>(ends[0]), 0};
    if (dim() == 2) {
      start_array[1] = nb::cast<int64_t>(starts[1]);
      end_array[1]   = nb::cast<int64_t>(ends[1]);
    } else if (dim() != 1) {
      throw std::invalid_argument("self.dim() must be 1 or 2");
    }

    wholegraph_tensor_t sub_tensor = nullptr;
    check_wholegraph_error_code(wholegraph_tensor_get_subtensor(
      wholegraph_tensor_, start_array.data(), end_array.data(), &sub_tensor));
    return PyWholeGraphTensorNB::from_c_handle(sub_tensor);
  }

  nb::tuple get_tensor_in_window(nb::object flatten_tensor, int64_t storage_window_offset) const
  {
    if (dim() == 1) {
      int64_t const storage_offset = tensor_description_.storage_offset;
      int64_t const start_index    = std::max<int64_t>(0, storage_offset - storage_window_offset);
      int64_t const end_index =
        std::min<int64_t>(python_shape_at(flatten_tensor, 0),
                          storage_offset + tensor_description_.sizes[0] - storage_window_offset);
      nb::object view = flatten_tensor.attr("__getitem__")(python_slice(start_index, end_index));
      return nb::make_tuple(view, std::max<int64_t>(0, storage_window_offset - storage_offset));
    }

    if (dim() == 2) {
      int64_t const embedding_stride = tensor_description_.strides[0];
      int64_t const storage_offset0 =
        tensor_description_.storage_offset / embedding_stride;
      int64_t const storage_offset1 =
        tensor_description_.storage_offset % embedding_stride;

      if (storage_window_offset % embedding_stride != 0) {
        throw std::invalid_argument("storage_window_offset must align with tensor stride");
      }

      nb::object mat_tensor = flatten_tensor.attr("reshape")(-1, embedding_stride);
      int64_t const vector_start_offset = storage_window_offset / embedding_stride;
      int64_t const start_index0 =
        std::max<int64_t>(0, storage_offset0 - vector_start_offset);
      int64_t const end_index0 =
        std::min<int64_t>(python_shape_at(mat_tensor, 0),
                          storage_offset0 + tensor_description_.sizes[0] -
                            vector_start_offset);
      int64_t const end_index1 = storage_offset1 + tensor_description_.sizes[1];
      if (python_shape_at(mat_tensor, 1) < end_index1) {
        throw std::invalid_argument("flatten tensor shape is smaller than tensor window");
      }

      nb::object rows = python_slice(start_index0, end_index0);
      nb::object cols = python_slice(storage_offset1, end_index1);
      nb::object view = mat_tensor.attr("__getitem__")(nb::make_tuple(rows, cols));
      return nb::make_tuple(
        view, std::max<int64_t>(0, vector_start_offset - storage_offset0));
    }

    throw std::invalid_argument("tensor dim should be 1 or 2");
  }

  nb::tuple get_local_tensor(nb::callable import_dlpack_fn,
                             wholegraph_memory_location_t view_from_device,
                             int view_from_device_id) const
  {
    nb::tuple flatten_and_offset =
      get_wholegraph_handle().get_local_flatten_tensor(import_dlpack_fn,
                                                       tensor_description_.dtype,
                                                       view_from_device,
                                                       view_from_device_id);
    nb::object flatten_tensor = nb::borrow<nb::object>(flatten_and_offset[0]);
    int64_t const element_offset = nb::cast<int64_t>(flatten_and_offset[1]);
    return get_tensor_in_window(flatten_tensor, element_offset);
  }

  nb::object get_global_tensor(nb::callable import_dlpack_fn,
                               wholegraph_memory_location_t view_from_device,
                               int view_from_device_id) const
  {
    nb::tuple flatten_and_offset =
      get_wholegraph_handle().get_global_flatten_tensor(import_dlpack_fn,
                                                        tensor_description_.dtype,
                                                        view_from_device,
                                                        view_from_device_id);
    nb::object global_flatten_tensor = nb::borrow<nb::object>(flatten_and_offset[0]);
    nb::tuple window                 = get_tensor_in_window(global_flatten_tensor, 0);
    return nb::borrow<nb::object>(window[0]);
  }

  void from_filelist(nb::object file_list, int round_robin_size = 0) const
  {
    auto handle = get_wholegraph_handle();
    size_t const elt_size =
      wholegraph_dtype_get_element_size(tensor_description_.dtype);
    size_t const memory_offset = storage_offset() * elt_size;
    size_t const memory_entry_size = elt_size * tensor_description_.strides[0];
    size_t file_entry_size = elt_size;
    if (dim() == 2) {
      file_entry_size = elt_size * tensor_description_.sizes[1];
    } else if (dim() != 1) {
      throw std::invalid_argument("tensor dim must be 1 or 2");
    }
    handle.from_filelist(
      memory_offset, memory_entry_size, file_entry_size, round_robin_size, file_list);
  }

  void to_file(std::string const& filename) const
  {
    auto handle = get_wholegraph_handle();
    size_t const elt_size =
      wholegraph_dtype_get_element_size(tensor_description_.dtype);
    size_t const memory_offset = storage_offset() * elt_size;
    size_t const memory_entry_size = elt_size * tensor_description_.strides[0];
    size_t file_entry_size = elt_size;
    if (dim() == 2) {
      file_entry_size = elt_size * tensor_description_.sizes[1];
    } else if (dim() != 1) {
      throw std::invalid_argument("tensor dim must be 1 or 2");
    }
    handle.to_file(memory_offset, memory_entry_size, file_entry_size, filename);
  }

  wholegraph_tensor_t c_handle() const { return wholegraph_tensor_; }

 private:
  void refresh_description()
  {
    if (wholegraph_tensor_ != nullptr) {
      tensor_description_ = *wholegraph_tensor_get_tensor_description(wholegraph_tensor_);
    }
  }

  wholegraph_tensor_t wholegraph_tensor_ = nullptr;
  wholegraph_tensor_description_t tensor_description_{};
};
