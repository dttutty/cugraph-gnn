/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholegraph/tensor_description.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t wholegraph_dtype_get_element_size(wholegraph_dtype_t dtype)
{
  switch (dtype) {
    case WHOLEGRAPH_DT_UNKNOWN: return 0;
    case WHOLEGRAPH_DT_INT8: return 1;
    case WHOLEGRAPH_DT_INT16:
    case WHOLEGRAPH_DT_BF16:
    case WHOLEGRAPH_DT_HALF: return 2;
    case WHOLEGRAPH_DT_INT:
    case WHOLEGRAPH_DT_FLOAT: return 4;
    case WHOLEGRAPH_DT_INT64:
    case WHOLEGRAPH_DT_DOUBLE: return 8;
    default: return -1;
  }
}

bool wholegraph_dtype_is_floating_number(wholegraph_dtype_t dtype)
{
  if (dtype == WHOLEGRAPH_DT_FLOAT || dtype == WHOLEGRAPH_DT_HALF ||
      dtype == WHOLEGRAPH_DT_DOUBLE || dtype == WHOLEGRAPH_DT_BF16)
    return true;
  return false;
}

bool wholegraph_dtype_is_integer_number(wholegraph_dtype_t dtype)
{
  if (dtype == WHOLEGRAPH_DT_INT || dtype == WHOLEGRAPH_DT_INT64 ||
      dtype == WHOLEGRAPH_DT_INT16 || dtype == WHOLEGRAPH_DT_INT8)
    return true;
  return false;
}

wholegraph_array_description_t wholegraph_create_array_desc(int64_t size,
                                                              int64_t storage_offset,
                                                              wholegraph_dtype_t dtype)
{
  wholegraph_array_description_t wg_array_desc;
  wg_array_desc.size           = size;
  wg_array_desc.storage_offset = storage_offset;
  wg_array_desc.dtype          = dtype;
  return wg_array_desc;
}

wholegraph_matrix_description_t wholegraph_create_matrix_desc(int64_t sizes[2],
                                                                int64_t stride,
                                                                int64_t storage_offset,
                                                                wholegraph_dtype_t dtype)
{
  wholegraph_matrix_description_t wg_matrix_desc;
  wg_matrix_desc.sizes[0]       = sizes[0];
  wg_matrix_desc.sizes[1]       = sizes[1];
  wg_matrix_desc.stride         = stride;
  wg_matrix_desc.storage_offset = storage_offset;
  wg_matrix_desc.dtype          = dtype;
  return wg_matrix_desc;
}

void wholegraph_initialize_tensor_desc(wholegraph_tensor_description_t* p_tensor_description)
{
  p_tensor_description->dim = 0;
  for (int i = 0; i < WHOLEGRAPH_MAX_TENSOR_DIM; i++) {
    p_tensor_description->sizes[i]   = 1;
    p_tensor_description->strides[i] = 1;
  }
  p_tensor_description->storage_offset = 0;
  p_tensor_description->dtype          = WHOLEGRAPH_DT_UNKNOWN;
}

void wholegraph_copy_array_desc_to_matrix(wholegraph_matrix_description_t* p_matrix_description,
                                           wholegraph_array_description_t* p_array_description)
{
  p_matrix_description->sizes[0]       = p_array_description->size;
  p_matrix_description->sizes[1]       = 1;
  p_matrix_description->dtype          = p_array_description->dtype;
  p_matrix_description->stride         = 1;
  p_matrix_description->storage_offset = p_array_description->storage_offset;
}

void wholegraph_copy_array_desc_to_tensor(wholegraph_tensor_description_t* p_tensor_description,
                                           wholegraph_array_description_t* p_array_description)
{
  wholegraph_initialize_tensor_desc(p_tensor_description);
  p_tensor_description->dim            = 1;
  p_tensor_description->sizes[0]       = p_array_description->size;
  p_tensor_description->strides[0]     = 1;
  p_tensor_description->dtype          = p_array_description->dtype;
  p_tensor_description->storage_offset = p_array_description->storage_offset;
}

void wholegraph_copy_matrix_desc_to_tensor(wholegraph_tensor_description_t* p_tensor_description,
                                            wholegraph_matrix_description_t* p_matrix_description)
{
  wholegraph_initialize_tensor_desc(p_tensor_description);
  p_tensor_description->dim            = 2;
  p_tensor_description->sizes[0]       = p_matrix_description->sizes[0];
  p_tensor_description->sizes[1]       = p_matrix_description->sizes[1];
  p_tensor_description->strides[0]     = p_matrix_description->stride;
  p_tensor_description->strides[1]     = 1;
  p_tensor_description->dtype          = p_matrix_description->dtype;
  p_tensor_description->storage_offset = p_matrix_description->storage_offset;
}

bool wholegraph_convert_tensor_desc_to_array(
  wholegraph_array_description_t* p_array_description,
  wholegraph_tensor_description_t* p_tensor_description)
{
  if (p_tensor_description->dtype <= WHOLEGRAPH_DT_UNKNOWN ||
      p_tensor_description->dtype >= WHOLEGRAPH_DT_COUNT)
    return false;
  if (p_tensor_description->dim != 1) return false;
  if (p_tensor_description->strides[0] != 1) return false;
  p_array_description->dtype          = p_tensor_description->dtype;
  p_array_description->storage_offset = p_tensor_description->storage_offset;
  p_array_description->size           = p_tensor_description->sizes[0];
  return true;
}

bool wholegraph_convert_tensor_desc_to_matrix(
  wholegraph_matrix_description_t* p_matrix_description,
  wholegraph_tensor_description_t* p_tensor_description)
{
  if (p_tensor_description->dtype <= WHOLEGRAPH_DT_UNKNOWN ||
      p_tensor_description->dtype >= WHOLEGRAPH_DT_COUNT)
    return false;
  if (p_tensor_description->dim > 2 || p_tensor_description->dim <= 0) return false;
  if (p_tensor_description->dim == 2 && p_tensor_description->strides[1] != 1) return false;
  p_matrix_description->dtype          = p_tensor_description->dtype;
  p_matrix_description->storage_offset = p_tensor_description->storage_offset;
  p_matrix_description->sizes[0]       = p_tensor_description->sizes[0];
  if (p_tensor_description->dim == 2) {
    p_matrix_description->sizes[1] = p_tensor_description->sizes[1];
    p_matrix_description->stride   = p_tensor_description->strides[0];
  } else {
    p_matrix_description->sizes[1] = 1;
    p_matrix_description->stride   = 1;
  }
  return true;
}

int64_t wholegraph_get_memory_element_count_from_array(
  wholegraph_array_description_t* p_array_description)
{
  return p_array_description->size;
}

int64_t wholegraph_get_memory_size_from_array(wholegraph_array_description_t* p_array_description)
{
  return wholegraph_get_memory_element_count_from_array(p_array_description) *
         wholegraph_dtype_get_element_size(p_array_description->dtype);
}

int64_t wholegraph_get_memory_element_count_from_matrix(
  wholegraph_matrix_description_t* p_matrix_description)
{
  return p_matrix_description->sizes[0] * p_matrix_description->stride;
}

int64_t wholegraph_get_memory_size_from_matrix(
  wholegraph_matrix_description_t* p_matrix_description)
{
  return wholegraph_get_memory_element_count_from_matrix(p_matrix_description) *
         wholegraph_dtype_get_element_size(p_matrix_description->dtype);
}

int64_t wholegraph_get_memory_element_count_from_tensor(
  wholegraph_tensor_description_t* p_tensor_description)
{
  if (p_tensor_description->dim == 0) return 1;
  if (p_tensor_description->dim < 0 || p_tensor_description->dim >= WHOLEGRAPH_MAX_TENSOR_DIM)
    return -1;
  return p_tensor_description->strides[0] * p_tensor_description->sizes[0];
}

int64_t wholegraph_get_memory_size_from_tensor(
  wholegraph_tensor_description_t* p_tensor_description)
{
  return wholegraph_get_memory_element_count_from_tensor(p_tensor_description) *
         wholegraph_dtype_get_element_size(p_tensor_description->dtype);
}

bool wholegraph_squeeze_tensor(wholegraph_tensor_description_t* p_tensor_description, int dim)
{
  if (p_tensor_description == nullptr) return false;
  if (dim < 0 || dim >= p_tensor_description->dim) return false;
  if (p_tensor_description->sizes[dim] != 1) return false;
  if (dim != p_tensor_description->dim - 1 &&
      p_tensor_description->strides[dim] != p_tensor_description->strides[dim + 1]) {
    return false;
  }
  for (int idx = dim; idx < p_tensor_description->dim - 1; idx++) {
    p_tensor_description->sizes[idx]   = p_tensor_description->sizes[idx + 1];
    p_tensor_description->strides[idx] = p_tensor_description->strides[idx + 1];
  }
  p_tensor_description->dim--;
  return true;
}

bool wholegraph_unsqueeze_tensor(wholegraph_tensor_description_t* p_tensor_description, int dim)
{
  if (p_tensor_description == nullptr) return false;
  if (dim < 0 || dim > p_tensor_description->dim) return false;
  int idx             = p_tensor_description->dim;
  int64_t last_stride = p_tensor_description->strides[p_tensor_description->dim - 1];
  for (; idx > dim; idx--) {
    p_tensor_description->sizes[idx] = p_tensor_description->sizes[idx - 1];
    last_stride = p_tensor_description->strides[idx] = p_tensor_description->strides[idx - 1];
  }
  p_tensor_description->sizes[dim]   = 1;
  p_tensor_description->strides[dim] = last_stride;
  p_tensor_description->dim++;
  return true;
}

#ifdef __cplusplus
}
#endif
