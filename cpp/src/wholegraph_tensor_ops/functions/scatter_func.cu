/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gather_scatter_func.h"

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"

namespace wholegraph_tensor_ops {

wholegraph_error_code_t scatter_integer_int32_func(const void* input,
                                                    wholegraph_matrix_description_t input_desc,
                                                    void* indices,
                                                    wholegraph_array_description_t indices_desc,
                                                    wholegraph_gref_t embedding_gref,
                                                    wholegraph_matrix_description_t embedding_desc,
                                                    cudaStream_t stream,
                                                    int scatter_sms);
wholegraph_error_code_t scatter_integer_int64_func(const void* input,
                                                    wholegraph_matrix_description_t input_desc,
                                                    void* indices,
                                                    wholegraph_array_description_t indices_desc,
                                                    wholegraph_gref_t embedding_gref,
                                                    wholegraph_matrix_description_t embedding_desc,
                                                    cudaStream_t stream,
                                                    int scatter_sms);
wholegraph_error_code_t scatter_floating_int32_func(
  const void* input,
  wholegraph_matrix_description_t input_desc,
  void* indices,
  wholegraph_array_description_t indices_desc,
  wholegraph_gref_t embedding_gref,
  wholegraph_matrix_description_t embedding_desc,
  cudaStream_t stream,
  int scatter_sms);
wholegraph_error_code_t scatter_floating_int64_func(
  const void* input,
  wholegraph_matrix_description_t input_desc,
  void* indices,
  wholegraph_array_description_t indices_desc,
  wholegraph_gref_t embedding_gref,
  wholegraph_matrix_description_t embedding_desc,
  cudaStream_t stream,
  int scatter_sms);

wholegraph_error_code_t scatter_func(const void* input,
                                      wholegraph_matrix_description_t input_desc,
                                      void* indices,
                                      wholegraph_array_description_t indices_desc,
                                      wholegraph_gref_t embedding_gref,
                                      wholegraph_matrix_description_t embedding_desc,
                                      cudaStream_t stream,
                                      int scatter_sms)
{
  try {
    bool embedding_is_float = wholegraph_dtype_is_floating_number(embedding_desc.dtype);
    WHOLEGRAPH_CHECK(embedding_is_float ||
                      wholegraph_dtype_is_integer_number(embedding_desc.dtype));
    bool input_is_float = wholegraph_dtype_is_floating_number(input_desc.dtype);
    WHOLEGRAPH_CHECK(input_is_float || wholegraph_dtype_is_integer_number(input_desc.dtype));
    WHOLEGRAPH_EXPECTS(
      embedding_is_float == input_is_float,
      "embedding and output should be same number type, e.g. floating number or integer number.");
    if (indices_desc.size == 0) { return WHOLEGRAPH_SUCCESS; }
    wholegraph_error_code_t (*p_scatter_func)(const void*,
                                               wholegraph_matrix_description_t,
                                               void*,
                                               wholegraph_array_description_t,
                                               wholegraph_gref_t,
                                               wholegraph_matrix_description_t,
                                               cudaStream_t,
                                               int) = nullptr;
    if (embedding_is_float) {
      if (indices_desc.dtype == WHOLEGRAPH_DT_INT) {
        p_scatter_func = scatter_floating_int32_func;
      } else {
        p_scatter_func = scatter_floating_int64_func;
      }
    } else {
      if (indices_desc.dtype == WHOLEGRAPH_DT_INT) {
        p_scatter_func = scatter_integer_int32_func;
      } else {
        p_scatter_func = scatter_integer_int64_func;
      }
    }
    return p_scatter_func(input,
                          input_desc,
                          indices,
                          indices_desc,
                          embedding_gref,
                          embedding_desc,
                          stream,
                          scatter_sms);
  } catch (const wholegraph::cuda_error& wle) {
    WHOLEGRAPH_ERROR("scatter CUDA LOGIC Error %s\n", wle.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (const wholegraph::logic_error& le) {
    WHOLEGRAPH_ERROR("scatter LOGIC Error %s\n", le.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (...) {
    return WHOLEGRAPH_UNKNOW_ERROR;
  }
  return WHOLEGRAPH_SUCCESS;
}

}  // namespace wholegraph_tensor_ops
