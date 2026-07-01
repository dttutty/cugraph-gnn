/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "wholegraph/wholegraph_tensor.h"

#include <atomic>
#include <cstdlib>

#include "logger.hpp"

#ifdef WG_TENSOR_COUNT_DEBUG
static std::atomic<int64_t> wg_tensor_count;
static void inc_tensor_count() { wg_tensor_count.fetch_add(1); }
static void dec_tensor_count() { wg_tensor_count.fetch_add(-1); }
static int64_t get_tensor_count() { return wg_tensor_count.load(); }
#else
static void inc_tensor_count() {}
static void dec_tensor_count() {}
static int64_t get_tensor_count() { return 0; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct wholegraph_tensor_ {
  union {
    wholegraph_handle_t wholegraph_handle;
    void* storage_ptr;
  };
  wholegraph_tensor_description_t tensor_description;
  wholegraph_tensor_t root_tensor;
  bool is_wholegraph;
  bool own_handle;
};

int64_t get_wholegraph_tensor_count() { return get_tensor_count(); }

wholegraph_error_code_t wholegraph_create_tensor(
  wholegraph_tensor_t* p_wholegraph_tensor,
  wholegraph_tensor_description_t* tensor_description,
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  size_t* tensor_entry_partition)
{
  if (p_wholegraph_tensor == nullptr) {
    WHOLEGRAPH_ERROR("p_wholegraph_tensor is nullptr");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description == nullptr) {
    WHOLEGRAPH_ERROR("tensor_description is nullptr");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description->dim <= 0 || tensor_description->dim > 2) {
    WHOLEGRAPH_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description->storage_offset != 0) {
    WHOLEGRAPH_ERROR("tensor_description->storage_offset=%ld", tensor_description->storage_offset);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEGRAPH_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEGRAPH_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEGRAPH_DT_COUNT) {
    WHOLEGRAPH_ERROR("tensor_description is unknown");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  size_t elt_count   = wholegraph_get_memory_element_count_from_tensor(tensor_description);
  size_t elt_size    = wholegraph_dtype_get_element_size(tensor_description->dtype);
  size_t malloc_size = elt_count * elt_size;
  size_t granularity = elt_size * tensor_description->strides[0];

  auto* wholegraph_tensor = static_cast<wholegraph_tensor_*>(malloc(sizeof(wholegraph_tensor_)));

  wholegraph_tensor->tensor_description = *tensor_description;
  wholegraph_tensor->own_handle         = true;
  wholegraph_tensor->is_wholegraph     = true;
  wholegraph_tensor->root_tensor        = wholegraph_tensor;
  *p_wholegraph_tensor                  = wholegraph_tensor;
  auto ret_code = wholegraph_malloc(&wholegraph_tensor->wholegraph_handle,
                                     malloc_size,
                                     comm,
                                     memory_type,
                                     memory_location,
                                     granularity,
                                     tensor_entry_partition);
  inc_tensor_count();
  if (ret_code != WHOLEGRAPH_SUCCESS) { free(wholegraph_tensor); }
  return ret_code;
}

wholegraph_error_code_t wholegraph_destroy_tensor(wholegraph_tensor_t wholegraph_tensor)
{
  if (wholegraph_tensor->own_handle) {
    if (wholegraph_tensor->is_wholegraph) {
      WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_free(wholegraph_tensor->wholegraph_handle));
    } else {
      free(wholegraph_tensor->storage_ptr);
    }
  }
  dec_tensor_count();
  free(wholegraph_tensor);
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_make_tensor_from_pointer(
  wholegraph_tensor_t* p_wholegraph_tensor,
  void* storage_ptr,
  wholegraph_tensor_description_t* tensor_description)
{
  if (storage_ptr == nullptr || tensor_description->dim == 0) {
    auto* wholegraph_tensor =
      static_cast<wholegraph_tensor_*>(malloc(sizeof(wholegraph_tensor_)));
    wholegraph_tensor->storage_ptr        = storage_ptr;
    wholegraph_tensor->tensor_description = *tensor_description;
    wholegraph_tensor->own_handle         = false;
    wholegraph_tensor->is_wholegraph     = false;
    wholegraph_tensor->root_tensor        = wholegraph_tensor;
    *p_wholegraph_tensor                  = wholegraph_tensor;
    inc_tensor_count();
    return WHOLEGRAPH_SUCCESS;
  }

  if (p_wholegraph_tensor == nullptr || tensor_description == nullptr) {
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description->dim < 0) {
    WHOLEGRAPH_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEGRAPH_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEGRAPH_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEGRAPH_DT_COUNT) {
    WHOLEGRAPH_ERROR("tensor_description is unknown");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto* wholegraph_tensor = static_cast<wholegraph_tensor_*>(malloc(sizeof(wholegraph_tensor_)));
  wholegraph_tensor->storage_ptr        = storage_ptr;
  wholegraph_tensor->tensor_description = *tensor_description;
  wholegraph_tensor->own_handle         = false;
  wholegraph_tensor->is_wholegraph     = false;
  wholegraph_tensor->root_tensor        = wholegraph_tensor;
  *p_wholegraph_tensor                  = wholegraph_tensor;
  inc_tensor_count();
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_make_tensor_from_handle(
  wholegraph_tensor_t* p_wholegraph_tensor,
  wholegraph_handle_t wholegraph_handle,
  wholegraph_tensor_description_t* tensor_description)
{
  if (wholegraph_handle == nullptr || p_wholegraph_tensor == nullptr ||
      tensor_description == nullptr) {
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description->dim <= 0 || tensor_description->dim > 2) {
    WHOLEGRAPH_ERROR("tensor_description->dim=%d", tensor_description->dim);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  int const dim = tensor_description->dim;
  if (tensor_description->strides[dim - 1] != 1) {
    WHOLEGRAPH_ERROR("tensor_description->strides[dim - 1]", tensor_description->strides[dim - 1]);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description->dtype <= WHOLEGRAPH_DT_UNKNOWN ||
      tensor_description->dtype >= WHOLEGRAPH_DT_COUNT) {
    WHOLEGRAPH_ERROR("tensor_description is unknown");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  auto* wholegraph_tensor = static_cast<wholegraph_tensor_*>(malloc(sizeof(wholegraph_tensor_)));
  wholegraph_tensor->wholegraph_handle = wholegraph_handle;
  wholegraph_tensor->tensor_description = *tensor_description;
  wholegraph_tensor->own_handle         = false;
  wholegraph_tensor->is_wholegraph     = true;
  wholegraph_tensor->root_tensor        = wholegraph_tensor;
  *p_wholegraph_tensor                  = wholegraph_tensor;
  inc_tensor_count();
  return WHOLEGRAPH_SUCCESS;
}

bool wholegraph_tensor_has_handle(wholegraph_tensor_t wholegraph_tensor)
{
  return wholegraph_tensor->is_wholegraph;
}

wholegraph_handle_t wholegraph_tensor_get_memory_handle(wholegraph_tensor_t wholegraph_tensor)
{
  if (wholegraph_tensor->is_wholegraph) { return wholegraph_tensor->wholegraph_handle; }
  return nullptr;
}

wholegraph_tensor_description_t* wholegraph_tensor_get_tensor_description(
  wholegraph_tensor_t wholegraph_tensor)
{
  return &wholegraph_tensor->tensor_description;
}

wholegraph_error_code_t wholegraph_tensor_get_global_reference(
  wholegraph_tensor_t wholegraph_tensor, wholegraph_gref_t* wholegraph_gref)
{
  if (wholegraph_gref == nullptr || wholegraph_tensor == nullptr) {
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (wholegraph_tensor->is_wholegraph) {
    return wholegraph_get_global_reference(wholegraph_gref,
                                            wholegraph_tensor->wholegraph_handle);
  }
  *wholegraph_gref =
    wholegraph_create_continuous_global_reference(wholegraph_tensor->storage_ptr);
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_tensor_map_local_tensor(
  wholegraph_tensor_t wholegraph_tensor, wholegraph_tensor_t* local_tensor)
{
  // NOTE: wholegraph_tensor should NOT skip entry from front, but can skip from tail.
  if (local_tensor == nullptr || wholegraph_tensor == nullptr) {
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (!wholegraph_tensor->is_wholegraph) { return WHOLEGRAPH_INVALID_VALUE; }
  auto* wg_desc = wholegraph_tensor_get_tensor_description(wholegraph_tensor);
  if (wg_desc->dim != 1 && wg_desc->dim != 2) { return WHOLEGRAPH_INVALID_VALUE; }
  if (wg_desc->dim == 1 && wg_desc->storage_offset != 0) { return WHOLEGRAPH_INVALID_VALUE; }
  if (wg_desc->dim == 2 && wg_desc->storage_offset + wg_desc->sizes[1] > wg_desc->strides[0]) {
    return WHOLEGRAPH_INVALID_VALUE;
  }

  wholegraph_comm_t wg_comm;
  int world_rank;

  void* local_ptr;
  size_t local_size, local_offset;
  auto* handle = wholegraph_tensor_get_memory_handle(wholegraph_tensor);
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(&wg_comm, handle));
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_rank(&world_rank, wg_comm));

  size_t total_handle_memory_size = wholegraph_get_total_size(handle);
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_get_local_memory(&local_ptr, &local_size, &local_offset, handle));
  size_t const element_size = wholegraph_dtype_get_element_size(wg_desc->dtype);
  size_t const gran_size    = wg_desc->dim == 1 ? element_size : element_size * wg_desc->strides[0];
  local_size = std::min<size_t>(local_size, wg_desc->sizes[0] * gran_size - local_offset);
  if (local_size % gran_size != 0) return WHOLEGRAPH_LOGIC_ERROR;
  wholegraph_tensor_description_t local_desc = *wg_desc;
  local_desc.sizes[0]                         = local_size / gran_size;
  WHOLEGRAPH_RETURN_ON_FAIL(
    wholegraph_make_tensor_from_pointer(local_tensor, local_ptr, &local_desc));

  return WHOLEGRAPH_SUCCESS;
}

void* wholegraph_tensor_get_data_pointer(wholegraph_tensor_t wholegraph_tensor)
{
  char* data_ptr = nullptr;
  if (wholegraph_tensor->is_wholegraph &&
      wholegraph_get_memory_type(wholegraph_tensor->wholegraph_handle) !=
        WHOLEGRAPH_MT_CONTINUOUS) {
    return nullptr;
  }
  if (!wholegraph_tensor->is_wholegraph) {
    data_ptr = static_cast<char*>(wholegraph_tensor->storage_ptr);
  } else {
    if (wholegraph_get_global_pointer(reinterpret_cast<void**>(&data_ptr),
                                       wholegraph_tensor->wholegraph_handle) !=
        WHOLEGRAPH_SUCCESS) {
      return nullptr;
    }
  }
  return data_ptr +
         wholegraph_dtype_get_element_size(wholegraph_tensor->tensor_description.dtype) *
           wholegraph_tensor->tensor_description.storage_offset;
}

wholegraph_error_code_t wholegraph_tensor_get_entry_offsets(
  size_t* entry_offsets, wholegraph_tensor_t wholegraph_tensor)
{
  wholegraph_tensor_t root_tensor = wholegraph_tensor_get_root(wholegraph_tensor);
  WHOLEGRAPH_CHECK_NOTHROW(
    (root_tensor->tensor_description.dim == 1 || root_tensor->tensor_description.dim == 2));
  if (wholegraph_tensor->is_wholegraph) {
    size_t embedding_stride = 1;
    size_t const element_size =
      wholegraph_dtype_get_element_size(wholegraph_tensor->tensor_description.dtype);
    if (root_tensor->tensor_description.dim == 2) {
      embedding_stride = root_tensor->tensor_description.strides[0];
    }

    int world_size;
    wholegraph_comm_t comm;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(
      &comm, wholegraph_tensor_get_memory_handle(wholegraph_tensor)));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&world_size, comm));

    wholegraph_get_rank_partition_offsets(
      entry_offsets, wholegraph_tensor_get_memory_handle(wholegraph_tensor));
    for (int i = 0; i < world_size + 1; i++) {
      WHOLEGRAPH_CHECK_NOTHROW(entry_offsets[i] % (embedding_stride * element_size) == 0);
      entry_offsets[i] /= (embedding_stride * element_size);
    }
    return WHOLEGRAPH_SUCCESS;
  }
  entry_offsets[0] = 0;
  entry_offsets[1] = root_tensor->tensor_description.sizes[0];
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_tensor_get_entry_partition_sizes(
  size_t* entry_partition, wholegraph_tensor_t wholegraph_tensor)
{
  wholegraph_tensor_t root_tensor = wholegraph_tensor_get_root(wholegraph_tensor);
  WHOLEGRAPH_CHECK_NOTHROW(
    (root_tensor->tensor_description.dim == 1 || root_tensor->tensor_description.dim == 2));
  if (wholegraph_tensor->is_wholegraph) {
    size_t embedding_stride = 1;
    size_t const element_size =
      wholegraph_dtype_get_element_size(wholegraph_tensor->tensor_description.dtype);
    if (root_tensor->tensor_description.dim == 2) {
      embedding_stride = root_tensor->tensor_description.strides[0];
    }

    int world_size;
    wholegraph_comm_t comm;
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_get_communicator(
      &comm, wholegraph_tensor_get_memory_handle(wholegraph_tensor)));
    WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_communicator_get_size(&world_size, comm));

    wholegraph_get_rank_partition_sizes(entry_partition,
                                         wholegraph_tensor_get_memory_handle(wholegraph_tensor));
    for (int i = 0; i < world_size; i++) {
      WHOLEGRAPH_CHECK_NOTHROW(entry_partition[i] % (embedding_stride * element_size) == 0);
      entry_partition[i] /= (embedding_stride * element_size);
    }
    return WHOLEGRAPH_SUCCESS;
  }
  entry_partition[0] = root_tensor->tensor_description.sizes[0];
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_tensor_get_local_entry_count(
  size_t* local_entry_count, wholegraph_tensor_t wholegraph_tensor)
{
  wholegraph_tensor_t root_tensor = wholegraph_tensor_get_root(wholegraph_tensor);
  WHOLEGRAPH_CHECK_NOTHROW(
    (root_tensor->tensor_description.dim == 1 || root_tensor->tensor_description.dim == 2));
  if (wholegraph_tensor->is_wholegraph) {
    size_t embedding_stride = 1;
    size_t const element_size =
      wholegraph_dtype_get_element_size(wholegraph_tensor->tensor_description.dtype);
    if (root_tensor->tensor_description.dim == 2) {
      embedding_stride = root_tensor->tensor_description.strides[0];
    }

    size_t entry_cnt;
    wholegraph_get_local_size(&entry_cnt,
                               wholegraph_tensor_get_memory_handle(wholegraph_tensor));
    WHOLEGRAPH_CHECK_NOTHROW(entry_cnt % (embedding_stride * element_size) == 0);
    entry_cnt /= (embedding_stride * element_size);
    *local_entry_count = entry_cnt;
    return WHOLEGRAPH_SUCCESS;
  }
  *local_entry_count = root_tensor->tensor_description.sizes[0];
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_tensor_get_local_entry_start(
  size_t* local_entry_start, wholegraph_tensor_t wholegraph_tensor)
{
  wholegraph_tensor_t root_tensor = wholegraph_tensor_get_root(wholegraph_tensor);
  WHOLEGRAPH_CHECK_NOTHROW(
    (root_tensor->tensor_description.dim == 1 || root_tensor->tensor_description.dim == 2));
  if (wholegraph_tensor->is_wholegraph) {
    size_t embedding_stride = 1;
    size_t const element_size =
      wholegraph_dtype_get_element_size(wholegraph_tensor->tensor_description.dtype);
    if (root_tensor->tensor_description.dim == 2) {
      embedding_stride = root_tensor->tensor_description.strides[0];
    }
    size_t entry_start;
    wholegraph_get_local_offset(&entry_start,
                                 wholegraph_tensor_get_memory_handle(wholegraph_tensor));
    WHOLEGRAPH_CHECK_NOTHROW(entry_start % (embedding_stride * element_size) == 0);
    entry_start /= (embedding_stride * element_size);
    *local_entry_start = entry_start;
    return WHOLEGRAPH_SUCCESS;
  }
  *local_entry_start = 0;
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t wholegraph_tensor_get_subtensor(
  wholegraph_tensor_t wholegraph_tensor,
  int64_t* starts,
  int64_t* ends,
  wholegraph_tensor_t* p_sub_wholegraph_tensor)
{
  if (p_sub_wholegraph_tensor == nullptr || wholegraph_tensor == nullptr || starts == nullptr ||
      ends == nullptr) {
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (wholegraph_tensor->tensor_description.dim > 2) { return WHOLEGRAPH_NOT_IMPLEMENTED; }
  int const dim      = wholegraph_tensor->tensor_description.dim;
  int64_t offsets[2] = {0, 0};
  if (dim == 1) {
    offsets[0] = wholegraph_tensor->tensor_description.storage_offset;
  } else {
    offsets[0] = wholegraph_tensor->tensor_description.storage_offset /
                 wholegraph_tensor->tensor_description.strides[0];
    offsets[1] = wholegraph_tensor->tensor_description.storage_offset %
                 wholegraph_tensor->tensor_description.strides[0];
  }
  int64_t new_size[2] = {0, 0};
  int64_t new_offset  = wholegraph_tensor->tensor_description.storage_offset;
  for (int i = 0; i < dim; i++) {
    int64_t starts_i = starts[i];
    int64_t ends_i   = ends[i];
    if (starts[i] == -1) starts_i = 0;
    if (ends[i] == -1) ends_i = wholegraph_tensor->tensor_description.sizes[i];
    if (ends_i <= starts_i) return WHOLEGRAPH_INVALID_INPUT;
    if (starts_i >= wholegraph_tensor->tensor_description.sizes[i])
      return WHOLEGRAPH_INVALID_INPUT;
    if (ends_i <= 0) return WHOLEGRAPH_INVALID_INPUT;
    new_offset += wholegraph_tensor->tensor_description.strides[i] * starts_i;
    new_size[i] = ends_i - starts_i;
  }
  auto* sub_wholegraph_tensor =
    static_cast<wholegraph_tensor_*>(malloc(sizeof(wholegraph_tensor_)));
  *sub_wholegraph_tensor                                   = *wholegraph_tensor;
  sub_wholegraph_tensor->own_handle                        = false;
  sub_wholegraph_tensor->tensor_description.storage_offset = new_offset;
  sub_wholegraph_tensor->tensor_description.dim            = dim;
  sub_wholegraph_tensor->tensor_description.dtype =
    sub_wholegraph_tensor->tensor_description.dtype;
  for (int i = 0; i < dim; i++) {
    sub_wholegraph_tensor->tensor_description.sizes[i] = new_size[i];
    sub_wholegraph_tensor->tensor_description.strides[i] =
      wholegraph_tensor->tensor_description.strides[i];
  }
  *p_sub_wholegraph_tensor = sub_wholegraph_tensor;
  inc_tensor_count();

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_tensor_t wholegraph_tensor_get_root(wholegraph_tensor_t wholegraph_tensor)
{
  return wholegraph_tensor->root_tensor;
}

#ifdef __cplusplus
}
#endif
