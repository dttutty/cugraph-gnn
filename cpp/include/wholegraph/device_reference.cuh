/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

#include "global_reference.h"
namespace wholegraph {

template <typename DataTypeT>
class device_reference {
 public:
  __device__ __forceinline__ explicit device_reference(const wholegraph_gref_t& gref)
    : pointer_(static_cast<DataTypeT*>(gref.pointer))
  {
  }
  __device__ device_reference() = delete;

  __device__ __forceinline__ DataTypeT& operator[](size_t index)
  {
    return pointer_[index];
  }

 private:
  DataTypeT* pointer_;
};

}  // namespace wholegraph
