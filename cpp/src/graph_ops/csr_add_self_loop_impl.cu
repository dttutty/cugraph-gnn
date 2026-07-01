/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "csr_add_self_loop_func.cuh"
#include <wholegraph/wholegraph.h>

#include "wholegraph_tensor_ops/register.hpp"

namespace graph_ops {
wholegraph_error_code_t csr_add_self_loop_impl(
  int* csr_row_ptr,
  wholegraph_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholegraph_array_description_t csr_col_ptr_array_desc,
  int* output_csr_row_ptr,
  wholegraph_array_description_t output_csr_row_ptr_array_desc,
  int* output_csr_col_ptr,
  wholegraph_array_description_t output_csr_col_ptr_array_desc,
  cudaStream_t stream)
{
  try {
    csr_add_self_loop_func(csr_row_ptr,
                           csr_row_ptr_array_desc,
                           csr_col_ptr,
                           csr_col_ptr_array_desc,
                           output_csr_row_ptr,
                           output_csr_row_ptr_array_desc,
                           output_csr_col_ptr,
                           output_csr_col_ptr_array_desc,
                           stream);

  } catch (const wholegraph::cuda_error& rle) {
    // WHOLEGRAPH_FAIL_NOTHROW("%s", rle.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (const wholegraph::logic_error& le) {
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (...) {
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  return WHOLEGRAPH_SUCCESS;
}

}  // namespace graph_ops
