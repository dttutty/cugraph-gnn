/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gather_scatter_func.cuh"

#include <wholegraph/wholegraph.h>

#include "logger.hpp"
#include "wholegraph_tensor_ops/register.hpp"

namespace wholegraph_tensor_ops {

template <typename InputT, typename EmbeddingT>
void scatter_floating_int32_temp_func(const void* input,
                                      wholegraph_matrix_description_t input_desc,
                                      void* indices,
                                      int64_t indice_count,
                                      wholegraph_gref_t embedding_gref,
                                      wholegraph_matrix_description_t embedding_desc,
                                      cudaStream_t stream,
                                      int scatter_sms)
{
  scatter_temp_func<InputT, int32_t, EmbeddingT>(
    input, input_desc, indices, indice_count, embedding_gref, embedding_desc, stream, scatter_sms);
}

REGISTER_DISPATCH_TWO_TYPES(ScatterFuncFloatingInt32,
                            scatter_floating_int32_temp_func,
                            ALLFLOAT,
                            ALLFLOAT)

wholegraph_error_code_t scatter_floating_int32_func(
  const void* input,
  wholegraph_matrix_description_t input_desc,
  void* indices,
  wholegraph_array_description_t indices_desc,
  wholegraph_gref_t embedding_gref,
  wholegraph_matrix_description_t embedding_desc,
  cudaStream_t stream,
  int scatter_sms)
{
  try {
    WHOLEGRAPH_CHECK(wholegraph_dtype_is_floating_number(embedding_desc.dtype));
    WHOLEGRAPH_CHECK(wholegraph_dtype_is_floating_number(input_desc.dtype));
    WHOLEGRAPH_CHECK(indices_desc.dtype == WHOLEGRAPH_DT_INT);
    DISPATCH_TWO_TYPES(
      input_desc.dtype,
      embedding_desc.dtype,
      ScatterFuncFloatingInt32,
      input,
      input_desc,
      static_cast<char*>(indices) +
        indices_desc.storage_offset * wholegraph_dtype_get_element_size(indices_desc.dtype),
      indices_desc.size,
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
