/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "thrust_allocator.hpp"

#include "error.hpp"
#include "wholegraph/integer_utils.hpp"

namespace wholegraph_tensor_ops {

wg_thrust_allocator::~wg_thrust_allocator() { deallocate_all(); }

wg_thrust_allocator::value_type* wg_thrust_allocator::allocate(std::ptrdiff_t mem_size)
{
  static const std::ptrdiff_t kThrustAlignSize = 256;
  mem_size = std::max<std::ptrdiff_t>(kThrustAlignSize, mem_size);
  mem_size = wholegraph::div_rounding_up_unsafe(mem_size, kThrustAlignSize) * kThrustAlignSize;
  void* memory_context = nullptr;
  fns->temporary_fns.create_memory_context_fn(&memory_context, fns->temporary_fns.global_context);
  wholegraph_tensor_description_t tensor_description;
  wholegraph_initialize_tensor_desc(&tensor_description);
  tensor_description.dim      = 1;
  tensor_description.dtype    = WHOLEGRAPH_DT_INT64;
  tensor_description.sizes[0] = mem_size / sizeof(int64_t);
  auto* ptr                   = static_cast<value_type*>(fns->temporary_fns.malloc_fn(
    &tensor_description, WHOLEGRAPH_MA_DEVICE, memory_context, fns->temporary_fns.global_context));
  mem_ptr_to_context_map.emplace(ptr, memory_context);
  return ptr;
}

void wg_thrust_allocator::deallocate(value_type* p, size_t /*mem_size*/)
{
  auto it = mem_ptr_to_context_map.find(p);
  WHOLEGRAPH_CHECK_NOTHROW(it != mem_ptr_to_context_map.end());
  fns->temporary_fns.free_fn(it->second, fns->temporary_fns.global_context);
  fns->temporary_fns.destroy_memory_context_fn(it->second, fns->temporary_fns.global_context);
  mem_ptr_to_context_map.erase(p);
}

void wg_thrust_allocator::deallocate_all()
{
  while (!mem_ptr_to_context_map.empty()) {
    auto it = mem_ptr_to_context_map.begin();
    fns->temporary_fns.free_fn(it->second, fns->temporary_fns.global_context);
    fns->temporary_fns.destroy_memory_context_fn(it->second, fns->temporary_fns.global_context);
    mem_ptr_to_context_map.erase(it->first);
  }
}

}  // namespace wholegraph_tensor_ops
