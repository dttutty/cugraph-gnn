/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum wholegraph_dtype_t
 * @brief defines WholeGraph data type for tensors
 */
enum wholegraph_dtype_t {
  WHOLEGRAPH_DT_UNKNOWN = 0, /*!< Unknown type */
  WHOLEGRAPH_DT_FLOAT,       /*!< 32-bit float type */
  WHOLEGRAPH_DT_HALF,        /*!< 16-bit half float type */
  WHOLEGRAPH_DT_DOUBLE,      /*!< 64-bit double type */
  WHOLEGRAPH_DT_BF16,        /*!< 16-bit bfloat type */
  WHOLEGRAPH_DT_INT,         /*!< 32-bit signed integer type */
  WHOLEGRAPH_DT_INT64,       /*!< 64-bit signed integer type */
  WHOLEGRAPH_DT_INT16,       /*!< 16-bit signed integer type */
  WHOLEGRAPH_DT_INT8,        /*!< 8-bit signed integer type */
  WHOLEGRAPH_DT_COUNT,       /*!< total count if types */
};

/**
 * Get element size of wholegraph_dtype_t
 * @param dtype : wholegraph_dtype_t
 * @return : element size of dtype, -1 on invalid dtype.
 */
size_t wholegraph_dtype_get_element_size(wholegraph_dtype_t dtype);

/**
 * Check if dtype is floating number
 * @param dtype : wholegraph_dtype_t
 * @return : True if dtype is WHOLEGRAPH_DT_FLOAT, WHOLEGRAPH_DT_HALF, WHOLEGRAPH_DT_DOUBLE or
 * WHOLEGRAPH_DT_BF16. False otherwise.
 */
bool wholegraph_dtype_is_floating_number(wholegraph_dtype_t dtype);

/**
 * Check if dtype is integer number
 * @param dtype : wholegraph_dtype_t
 * @return : True if dtype is WHOLEGRAPH_DT_INT, WHOLEGRAPH_DT_INT64, WHOLEGRAPH_DT_INT16 or
 * WHOLEGRAPH_DT_INT8, False otherwise.
 */
bool wholegraph_dtype_is_integer_number(wholegraph_dtype_t dtype);

/**
 * @struct wholegraph_array_description_t
 * @brief wrapper for array in WholeGraph
 */
struct wholegraph_array_description_t {
  int64_t size;              /*!< size of the array in elements. */
  int64_t storage_offset;    /*!< offset in number of elements, NOT in bytes. */
  wholegraph_dtype_t dtype; /*!< data type of the array */
};

/**
 * @struct wholegraph_matrix_description_t
 * @brief wrapper for matrix in WholeGraph
 */
struct wholegraph_matrix_description_t {
  int64_t sizes[2];          /*!< sizes[0] is row of the matrix, sizes[1] is column of the matrix */
  int64_t stride;            /*!< stride of first dimension, in number of elements */
  int64_t storage_offset;    /*!< offset in number of elements, NOT in bytes. */
  wholegraph_dtype_t dtype; /*!< data type of the matrix */
};

#define WHOLEGRAPH_MAX_TENSOR_DIM (8)

/**
 * @struct wholegraph_tensor_description_t
 * @brief Tensor description in WholeGraph, dimension 0 is the slowest changed dimension
 */
struct wholegraph_tensor_description_t {
  int64_t sizes[WHOLEGRAPH_MAX_TENSOR_DIM]; /*!< size of each dimension of the tensor, in number of
                                                elements */
  int64_t strides[WHOLEGRAPH_MAX_TENSOR_DIM]; /*!< stride of the tensor, in number of elements */
  int64_t storage_offset;                      /*!< offset in number of elements, NOT in bytes. */
  int dim;                                     /*!< dim of the tensor */
  wholegraph_dtype_t dtype;                   /*!< data type of the tensor */
};

/*!
 * Create wholegraph_array_description_t object
 * @param size : array size in number of elements
 * @param storage_offset : storage offset in number of elements
 * @param dtype : data type of array elements
 * @return created wholegraph_array_description_t
 */
wholegraph_array_description_t wholegraph_create_array_desc(int64_t size,
                                                              int64_t storage_offset,
                                                              wholegraph_dtype_t dtype);

/*!
 * Create wholegraph_matrix_description_t object
 * @param sizes : matrix sizes array, counted in number of elements, sizes[1] changes fastest.
 * @param stride : stride of first dimension(slower changed dimension), stride is counted in number
 * of elements
 * @param storage_offset : storage offset in number of elements
 * @param dtype : data type of matrix elements
 * @return created wholegraph_matrix_description_t
 */
wholegraph_matrix_description_t wholegraph_create_matrix_desc(int64_t sizes[2],
                                                                int64_t stride,
                                                                int64_t storage_offset,
                                                                wholegraph_dtype_t dtype);

/*!
 * Initialize wholegraph_tensor_description_t, set sizes and strides to all ones, and set
 * storage_offset to 0, set dtype to WHOLEGRAPH_DT_UNKNOWN, set dim to 0.
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t.
 */
void wholegraph_initialize_tensor_desc(wholegraph_tensor_description_t* p_tensor_description);

/**
 * Copy array description to tensor description
 * @param p_matrix_description : pointer to wholegraph_matrix_description_t.
 * @param p_array_description : pointer to wholegraph_array_description_t.
 */
void wholegraph_copy_array_desc_to_matrix(wholegraph_matrix_description_t* p_matrix_description,
                                           wholegraph_array_description_t* p_array_description);

/*!
 * Copy array description to tensor description
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t.
 * @param p_array_description : pointer to wholegraph_array_description_t.
 */
void wholegraph_copy_array_desc_to_tensor(wholegraph_tensor_description_t* p_tensor_description,
                                           wholegraph_array_description_t* p_array_description);

/*!
 * Copy matrix description to tensor description
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t.
 * @param p_matrix_description : pointer to wholegraph_matrix_description_t.
 */
void wholegraph_copy_matrix_desc_to_tensor(wholegraph_tensor_description_t* p_tensor_description,
                                            wholegraph_matrix_description_t* p_matrix_description);
/*!
 * Convert tensor description to array description
 * @param p_array_description : pointer to wholegraph_array_description_t.
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t.
 * @return : Return true if convertible else false.
 */
bool wholegraph_convert_tensor_desc_to_array(
  wholegraph_array_description_t* p_array_description,
  wholegraph_tensor_description_t* p_tensor_description);

/*!
 * Convert tensor description to matrix description
 * @param p_matrix_description : pointer to wholegraph_matrix_description_t.
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t.
 * @return : Return true if convertible else false.
 */
bool wholegraph_convert_tensor_desc_to_matrix(
  wholegraph_matrix_description_t* p_matrix_description,
  wholegraph_tensor_description_t* p_tensor_description);

/*!
 * Get total element count from array description.
 * @param p_array_description : pointer to wholegraph_array_description_t.
 * @return : Return element count.
 */
int64_t wholegraph_get_memory_element_count_from_array(
  wholegraph_array_description_t* p_array_description);

/*!
 * Get total memory size from array description.
 * @param p_array_description : pointer to wholegraph_array_description_t.
 * @return : Return memory size.
 */
int64_t wholegraph_get_memory_size_from_array(
  wholegraph_array_description_t* p_array_description);

/*!
 * Get total element count from matrix description.
 * @param p_matrix_description : pointer to wholegraph_matrix_description_t.
 * @return : Return element count.
 */
int64_t wholegraph_get_memory_element_count_from_matrix(
  wholegraph_matrix_description_t* p_matrix_description);

/*!
 * Get total memory size from matrix description.
 * @param p_matrix_description : pointer to wholegraph_matrix_description_t.
 * @return : Return memory size.
 */
int64_t wholegraph_get_memory_size_from_matrix(
  wholegraph_matrix_description_t* p_matrix_description);

/*!
 * Get total element count from tensor description.
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t.
 * @return : Return element count.
 */
int64_t wholegraph_get_memory_element_count_from_tensor(
  wholegraph_tensor_description_t* p_tensor_description);

/*!
 * Get total memory size from tensor description.
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t.
 * @return : Return memory size.
 */
int64_t wholegraph_get_memory_size_from_tensor(
  wholegraph_tensor_description_t* p_tensor_description);

/**
 * Squeeze tensor
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t
 * @param dim : which dim to squeeze
 * @return : true if success else false
 */
bool wholegraph_squeeze_tensor(wholegraph_tensor_description_t* p_tensor_description, int dim);

/**
 * Unsqueeze tensor
 * @param p_tensor_description : pointer to wholegraph_tensor_description_t
 * @param dim : unsqueeze at which dim
 * @return : true if success else false
 */
bool wholegraph_unsqueeze_tensor(wholegraph_tensor_description_t* p_tensor_description, int dim);

#ifdef __cplusplus
}
#endif
