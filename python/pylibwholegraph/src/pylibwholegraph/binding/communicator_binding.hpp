/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "binding_utils.hpp"

class PyWholeGraphUniqueIDNB {
 public:
  PyWholeGraphUniqueIDNB()
  {
    shape_[0]   = static_cast<int64_t>(sizeof(unique_id_.internal));
    strides_[0] = 1;
  }

  size_t length() const { return sizeof(unique_id_.internal); }

  wholegraph_unique_id_t* c_ptr() { return &unique_id_; }

  wholegraph_unique_id_t const& value() const { return unique_id_; }

  nb::object dlpack(nb::object /* stream */ = nb::none()) const
  {
    auto dlm_tensor             = make_dlmanaged_tensor();
    dlm_tensor->dl_tensor.data  = const_cast<char*>(&unique_id_.internal[0]);
    dlm_tensor->dl_tensor.ndim  = 1;
    dlm_tensor->dl_tensor.shape = const_cast<int64_t*>(shape_.data());
    dlm_tensor->dl_tensor.strides = const_cast<int64_t*>(strides_.data());
    dlm_tensor->dl_tensor.byte_offset        = 0;
    dlm_tensor->dl_tensor.device.device_type = static_cast<int>(DLDeviceTypeNB::kDLCPU);
    dlm_tensor->dl_tensor.device.device_id   = 0;
    dlm_tensor->dl_tensor.dtype.code = static_cast<uint8_t>(DLDataTypeCodeNB::kDLInt);
    dlm_tensor->dl_tensor.dtype.bits  = 8;
    dlm_tensor->dl_tensor.dtype.lanes = 1;
    dlm_tensor->manager_ctx           = retained_python_owner(*this);
    dlm_tensor->deleter               = dlpack_tensor_deleter;
    return make_dlpack_capsule(std::move(dlm_tensor));
  }

  nb::tuple dlpack_device() const
  {
    return nb::make_tuple(static_cast<int>(DLDeviceTypeNB::kDLCPU), 0);
  }

 private:
  wholegraph_unique_id_t unique_id_{};
  std::array<int64_t, 1> shape_{};
  std::array<int64_t, 1> strides_{};
};

class PyWholeGraphCommNB {
 public:
  PyWholeGraphCommNB() = default;

  static PyWholeGraphCommNB from_c_handle(wholegraph_comm_t comm_id)
  {
    PyWholeGraphCommNB comm;
    comm.comm_id_ = comm_id;
    return comm;
  }

  int64_t get_c_handle() const { return reinterpret_cast<int64_t>(comm_id_); }

  bool support_type_location(wholegraph_memory_type_t memory_type,
                             wholegraph_memory_location_t memory_location) const
  {
    auto err =
      wholegraph_communicator_support_type_location(comm_id_, memory_type, memory_location);
    return err == WHOLEGRAPH_SUCCESS;
  }

  int get_rank() const
  {
    int rank = -1;
    check_wholegraph_error_code(wholegraph_communicator_get_rank(&rank, comm_id_));
    return rank;
  }

  int get_size() const
  {
    int size = -1;
    check_wholegraph_error_code(wholegraph_communicator_get_size(&size, comm_id_));
    return size;
  }

  nb::tuple get_clique_info() const
  {
    clique_info_t clique_info{};
    check_wholegraph_error_code(wholegraph_communicator_get_clique_info(&clique_info, comm_id_));
    return nb::make_tuple(clique_info.is_in_clique > 0,
                          clique_info.clique_first_rank,
                          clique_info.clique_rank,
                          clique_info.clique_rank_num,
                          clique_info.clique_id,
                          clique_info.clique_num);
  }

  void barrier() const { check_wholegraph_error_code(wholegraph_communicator_barrier(comm_id_)); }

  wholegraph_distributed_backend_t get_distributed_backend() const
  {
    return wholegraph_communicator_get_distributed_backend(comm_id_);
  }

  void set_distributed_backend(wholegraph_distributed_backend_t distributed_backend)
  {
    check_wholegraph_error_code(
      wholegraph_communicator_set_distributed_backend(comm_id_, distributed_backend));
  }

  wholegraph_comm_t c_handle() const { return comm_id_; }

 private:
  wholegraph_comm_t comm_id_ = nullptr;
};
