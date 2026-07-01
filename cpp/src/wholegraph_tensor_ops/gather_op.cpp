/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholegraph/wholegraph_tensor_op.h>

#include <wholegraph_tensor_ops/gather_op_impl.h>

#include "error.hpp"
#include "logger.hpp"

wholegraph_error_code_t wholegraph_gather(wholegraph_tensor_t wholegraph_tensor,
                                            wholegraph_tensor_t indices_tensor,
                                            wholegraph_tensor_t output_tensor,
                                            wholegraph_env_func_t* p_env_fns,
                                            void* stream,
                                            int gather_sms)
{
  bool const has_handle                         = wholegraph_tensor_has_handle(wholegraph_tensor);
  wholegraph_memory_type_t memory_type         = WHOLEGRAPH_MT_NONE;
  wholegraph_memory_location_t memory_location = WHOLEGRAPH_ML_NONE;
  if (has_handle) {
    auto memory_handle = wholegraph_tensor_get_memory_handle(wholegraph_tensor);
    memory_type        = wholegraph_get_memory_type(memory_handle);
    memory_location    = wholegraph_get_memory_location(memory_handle);
  }
  wholegraph_matrix_description_t matrix_description;
  auto tensor_description = *wholegraph_tensor_get_tensor_description(wholegraph_tensor);
  if (tensor_description.dim != 1 && tensor_description.dim != 2) {
    WHOLEGRAPH_ERROR("wholegraph_tensor should be 1D or 2D tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (tensor_description.dim == 1) {
    if (!wholegraph_unsqueeze_tensor(&tensor_description, 1)) {
      WHOLEGRAPH_ERROR("Input 1D wholegraph_tensor unsqueeze to 2D failed.");
      return WHOLEGRAPH_LOGIC_ERROR;
    }
  }
  if (!wholegraph_convert_tensor_desc_to_matrix(&matrix_description, &tensor_description)) {
    WHOLEGRAPH_ERROR("Input wholegraph_tensor convert to matrix failed.");
    return WHOLEGRAPH_LOGIC_ERROR;
  }
  if (wholegraph_tensor_get_tensor_description(indices_tensor)->dim != 1) {
    WHOLEGRAPH_ERROR("indices tensor should be 1D tensor");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  wholegraph_tensor_description_t output_tensor_desc =
    *wholegraph_tensor_get_tensor_description(output_tensor);
  if (output_tensor_desc.dim != tensor_description.dim) {
    WHOLEGRAPH_ERROR("output tensor should be same dim as wholegraph_tensor.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (output_tensor_desc.dim == 1) {
    if (!wholegraph_unsqueeze_tensor(&output_tensor_desc, 1)) {
      WHOLEGRAPH_ERROR("Output 1D wholegraph_tensor unsqueeze to 2D failed.");
      return WHOLEGRAPH_LOGIC_ERROR;
    }
  }
  void* indices = wholegraph_tensor_get_data_pointer(indices_tensor);
  void* output  = wholegraph_tensor_get_data_pointer(output_tensor);
  wholegraph_array_description_t indices_desc;
  wholegraph_matrix_description_t output_desc;
  if (!wholegraph_convert_tensor_desc_to_array(
        &indices_desc, wholegraph_tensor_get_tensor_description(indices_tensor))) {
    WHOLEGRAPH_ERROR("Convert indices tensor to array failed.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (!wholegraph_convert_tensor_desc_to_matrix(&output_desc, &output_tensor_desc)) {
    WHOLEGRAPH_ERROR("Convert output tensor to matrix failed.");
    return WHOLEGRAPH_INVALID_INPUT;
  }
  if (has_handle && memory_type == WHOLEGRAPH_MT_DISTRIBUTED) {
    return wholegraph_tensor_ops::wholegraph_gather_distributed(
      wholegraph_tensor_get_memory_handle(wholegraph_tensor),
      matrix_description,
      indices,
      indices_desc,
      output,
      output_desc,
      p_env_fns,
      static_cast<cudaStream_t>(stream),
      gather_sms);
  }

  WHOLEGRAPH_EXPECTS_NOTHROW(!has_handle || memory_type == WHOLEGRAPH_MT_CONTINUOUS,
                              "Memory type not supported.");

  wholegraph_gref_t gref;
  WHOLEGRAPH_RETURN_ON_FAIL(wholegraph_tensor_get_global_reference(wholegraph_tensor, &gref));

  int64_t entry_size =
    tensor_description.sizes[1] * wholegraph_dtype_get_element_size(tensor_description.dtype);
  bool gather_with_sorted_ids =
    (memory_location == WHOLEGRAPH_ML_HOST) && (entry_size <= 512) &&
    (memory_type == WHOLEGRAPH_MT_CONTINUOUS);
  return wholegraph_tensor_ops::wholegraph_gather_mapped(gref,
                                                    matrix_description,
                                                    indices,
                                                    indices_desc,
                                                    output,
                                                    output_desc,
                                                    gather_with_sorted_ids,
                                                    p_env_fns,
                                                    static_cast<cudaStream_t>(stream),
                                                    gather_sms);
}
