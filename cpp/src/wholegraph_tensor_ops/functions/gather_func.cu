/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gather_scatter_func.h"

#include "cuda_macros.hpp"
#include "error.hpp"

namespace wholegraph_tensor_ops {

wholegraph_error_code_t gather_integer_int32_func(wholegraph_gref_t embedding_gref,
                                                   wholegraph_matrix_description_t embedding_desc,
                                                   void* indices,
                                                   wholegraph_array_description_t indices_desc,
                                                   bool gather_with_sorted_ids,
                                                   void* raw_indices,
                                                   void* output,
                                                   wholegraph_matrix_description_t output_desc,
                                                   cudaStream_t stream,
                                                   int gather_sms);
wholegraph_error_code_t gather_integer_int64_func(wholegraph_gref_t embedding_gref,
                                                   wholegraph_matrix_description_t embedding_desc,
                                                   void* indices,
                                                   wholegraph_array_description_t indices_desc,
                                                   bool gather_with_sorted_ids,
                                                   void* raw_indices,
                                                   void* output,
                                                   wholegraph_matrix_description_t output_desc,
                                                   cudaStream_t stream,
                                                   int gather_sms);
wholegraph_error_code_t gather_floating_int32_func(wholegraph_gref_t embedding_gref,
                                                    wholegraph_matrix_description_t embedding_desc,
                                                    void* indices,
                                                    wholegraph_array_description_t indices_desc,
                                                    bool gather_with_sorted_ids,
                                                    void* raw_indices,
                                                    void* output,
                                                    wholegraph_matrix_description_t output_desc,
                                                    cudaStream_t stream,
                                                    int gather_sms);
wholegraph_error_code_t gather_floating_int64_func(wholegraph_gref_t embedding_gref,
                                                    wholegraph_matrix_description_t embedding_desc,
                                                    void* indices,
                                                    wholegraph_array_description_t indices_desc,
                                                    bool gather_with_sorted_ids,
                                                    void* raw_indices,
                                                    void* output,
                                                    wholegraph_matrix_description_t output_desc,
                                                    cudaStream_t stream,
                                                    int gather_sms);

wholegraph_error_code_t gather_func(wholegraph_gref_t embedding_gref,
                                     wholegraph_matrix_description_t embedding_desc,
                                     void* indices,
                                     wholegraph_array_description_t indices_desc,
                                     void* output,
                                     wholegraph_matrix_description_t output_desc,
                                     cudaStream_t stream,
                                     int gather_sms)
{
  try {
    bool embedding_is_float = wholegraph_dtype_is_floating_number(embedding_desc.dtype);
    WHOLEGRAPH_CHECK(embedding_is_float ||
                      wholegraph_dtype_is_integer_number(embedding_desc.dtype));
    bool output_is_float = wholegraph_dtype_is_floating_number(output_desc.dtype);
    WHOLEGRAPH_CHECK(output_is_float || wholegraph_dtype_is_integer_number(output_desc.dtype));
    WHOLEGRAPH_EXPECTS(
      embedding_is_float == output_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEGRAPH_SUCCESS; }
    wholegraph_error_code_t (*p_gather_func)(wholegraph_gref_t,
                                              wholegraph_matrix_description_t,
                                              void* indices,
                                              wholegraph_array_description_t,
                                              bool,
                                              void*,
                                              void*,
                                              wholegraph_matrix_description_t,
                                              cudaStream_t,
                                              int) = nullptr;
    if (embedding_is_float) {
      if (indices_desc.dtype == WHOLEGRAPH_DT_INT) {
        p_gather_func = gather_floating_int32_func;
      } else {
        p_gather_func = gather_floating_int64_func;
      }
    } else {
      if (indices_desc.dtype == WHOLEGRAPH_DT_INT) {
        p_gather_func = gather_integer_int32_func;
      } else {
        p_gather_func = gather_integer_int64_func;
      }
    }
    return p_gather_func(embedding_gref,
                         embedding_desc,
                         indices,
                         indices_desc,
                         false,
                         nullptr,
                         output,
                         output_desc,
                         stream,
                         gather_sms);
  } catch (const wholegraph::cuda_error& rle) {
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (const wholegraph::logic_error& le) {
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (...) {
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t gather_with_sorted_ids_func(
  wholegraph_gref_t embedding_gref,
  wholegraph_matrix_description_t embedding_desc,
  void* indices,
  wholegraph_array_description_t indices_desc,
  void* raw_indices,
  wholegraph_array_description_t raw_indices_desc,
  void* output,
  wholegraph_matrix_description_t output_desc,
  cudaStream_t stream,
  int gather_sms)
{
  try {
    bool embedding_is_float = wholegraph_dtype_is_floating_number(embedding_desc.dtype);
    WHOLEGRAPH_CHECK(embedding_is_float ||
                      wholegraph_dtype_is_integer_number(embedding_desc.dtype));
    bool output_is_float = wholegraph_dtype_is_floating_number(output_desc.dtype);
    WHOLEGRAPH_CHECK(output_is_float || wholegraph_dtype_is_integer_number(output_desc.dtype));
    WHOLEGRAPH_EXPECTS(
      embedding_is_float == output_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEGRAPH_SUCCESS; }
    WHOLEGRAPH_CHECK(indices_desc.size == raw_indices_desc.size);
    WHOLEGRAPH_CHECK(indices_desc.dtype == raw_indices_desc.dtype);
    wholegraph_error_code_t (*p_gather_func)(wholegraph_gref_t,
                                              wholegraph_matrix_description_t,
                                              void* indices,
                                              wholegraph_array_description_t,
                                              bool,
                                              void*,
                                              void*,
                                              wholegraph_matrix_description_t,
                                              cudaStream_t,
                                              int) = nullptr;
    if (embedding_is_float) {
      if (indices_desc.dtype == WHOLEGRAPH_DT_INT) {
        p_gather_func = gather_floating_int32_func;
      } else {
        p_gather_func = gather_floating_int64_func;
      }
    } else {
      if (indices_desc.dtype == WHOLEGRAPH_DT_INT) {
        p_gather_func = gather_integer_int32_func;
      } else {
        p_gather_func = gather_integer_int64_func;
      }
    }
    return p_gather_func(embedding_gref,
                         embedding_desc,
                         indices,
                         indices_desc,
                         true,
                         raw_indices,
                         output,
                         output_desc,
                         stream,
                         gather_sms);
  } catch (const wholegraph::cuda_error& rle) {
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (const wholegraph::logic_error& le) {
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (...) {
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  return WHOLEGRAPH_SUCCESS;
}

}  // namespace wholegraph_tensor_ops
