/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "gather_scatter_func.cuh"

#include <wholegraph/wholegraph.h>

#include "logger.hpp"
#include "wholegraph_tensor_ops/register.hpp"

#define WHOLEGRAPH_DEFINE_GATHER_FUNC_IMPL(DISPATCH_NAME,                                      \
                                           TEMP_FUNC,                                          \
                                           PUBLIC_FUNC,                                        \
                                           INDEX_T,                                            \
                                           INDEX_DTYPE,                                        \
                                           TYPE_PREDICATE,                                     \
                                           TYPE_SET,                                           \
                                           LOGIC_ERROR_MESSAGE)                                \
  template <typename EmbeddingT, typename OutputT>                                             \
  void TEMP_FUNC(wholegraph_gref_t embedding_gref,                                             \
                 wholegraph_matrix_description_t embedding_desc,                               \
                 void* indices,                                                               \
                 int64_t indice_count,                                                        \
                 bool gather_with_sorted_ids,                                                  \
                 void* raw_indices,                                                           \
                 void* output,                                                                \
                 wholegraph_matrix_description_t output_desc,                                  \
                 cudaStream_t stream,                                                         \
                 int gather_sms)                                                              \
  {                                                                                            \
    gather_temp_func<EmbeddingT, INDEX_T, OutputT>(embedding_gref,                             \
                                                   embedding_desc,                             \
                                                   indices,                                    \
                                                   indice_count,                               \
                                                   gather_with_sorted_ids,                     \
                                                   raw_indices,                                \
                                                   output,                                     \
                                                   output_desc,                                \
                                                   stream,                                     \
                                                   gather_sms);                                \
  }                                                                                            \
                                                                                               \
  REGISTER_DISPATCH_TWO_TYPES(DISPATCH_NAME, TEMP_FUNC, TYPE_SET, TYPE_SET)                    \
                                                                                               \
  wholegraph_error_code_t PUBLIC_FUNC(wholegraph_gref_t embedding_gref,                        \
                                      wholegraph_matrix_description_t embedding_desc,          \
                                      void* indices,                                           \
                                      wholegraph_array_description_t indices_desc,             \
                                      bool gather_with_sorted_ids,                              \
                                      void* raw_indices,                                       \
                                      void* output,                                            \
                                      wholegraph_matrix_description_t output_desc,             \
                                      cudaStream_t stream,                                     \
                                      int gather_sms)                                          \
  {                                                                                            \
    try {                                                                                      \
      WHOLEGRAPH_CHECK(TYPE_PREDICATE(embedding_desc.dtype));                                  \
      WHOLEGRAPH_CHECK(TYPE_PREDICATE(output_desc.dtype));                                     \
      WHOLEGRAPH_CHECK(indices_desc.dtype == INDEX_DTYPE);                                     \
      DISPATCH_TWO_TYPES(embedding_desc.dtype,                                                 \
                         output_desc.dtype,                                                    \
                         DISPATCH_NAME,                                                        \
                         embedding_gref,                                                       \
                         embedding_desc,                                                       \
                         static_cast<char*>(indices) +                                         \
                           indices_desc.storage_offset *                                       \
                             wholegraph_dtype_get_element_size(indices_desc.dtype),            \
                         indices_desc.size,                                                    \
                         gather_with_sorted_ids,                                               \
                         raw_indices,                                                          \
                         output,                                                               \
                         output_desc,                                                          \
                         stream,                                                               \
                         gather_sms);                                                          \
    } catch (const wholegraph::cuda_error& wle) {                                              \
      WHOLEGRAPH_ERROR("gather CUDA LOGIC Error %s\n", wle.what());                           \
      return WHOLEGRAPH_LOGIC_ERROR;                                                          \
    } catch (const wholegraph::logic_error& le) {                                              \
      WHOLEGRAPH_ERROR(LOGIC_ERROR_MESSAGE, le.what());                                        \
      return WHOLEGRAPH_LOGIC_ERROR;                                                          \
    } catch (...) {                                                                            \
      return WHOLEGRAPH_LOGIC_ERROR;                                                          \
    }                                                                                          \
    return WHOLEGRAPH_SUCCESS;                                                                 \
  }
