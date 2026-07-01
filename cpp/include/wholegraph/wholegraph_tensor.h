/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to WholeGraphTensor
 *
 * An Opaque handle to WholeGraphTensor
 */
typedef struct wholegraph_tensor_* wholegraph_tensor_t;

/**
 * Create WholeGraph Tensor
 * @param wholegraph_tensor : returned WholeGraph Tensor handle
 * @param tensor_description : description of the WholeGraph Tensor, should be 1-D or 2-D
 * continuous tensor without offset.
 * @param comm : WholeGraph Communicator
 * @param memory_type : Memory Type of the underlying WholeGraph
 * @param memory_location : Memory Location of the underlying WholeGraph
 * @param tensor_entry_partition : Tensor entry count of each rank, the length must be world_size.
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_create_tensor(
  wholegraph_tensor_t* wholegraph_tensor,
  wholegraph_tensor_description_t* tensor_description,
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  size_t* tensor_entry_partition = nullptr);

/**
 * Destroy WholeGraph Tensor
 * @param wholegraph_tensor : WholeGraph Tensor to destroy
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_destroy_tensor(wholegraph_tensor_t wholegraph_tensor);

/**
 * Make WholeGraph Tensor from local memory
 * @param wholegraph_tensor : returned WholeGraph Tensor handle
 * @param storage_ptr : pointer to underlying storage memory. Note: storage pointer may be not same
 * as data pointer.
 * @param tensor_description : description of the WholeGraph Tensor, should be 1-D or 2-D
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_make_tensor_from_pointer(
  wholegraph_tensor_t* wholegraph_tensor,
  void* storage_ptr,
  wholegraph_tensor_description_t* tensor_description);

/**
 * Make WholeGraph Tensor from local memory
 * @param wholegraph_tensor : returned WholeGraph Tensor handle
 * @param wholegraph_handle : WholeGraph Handle
 * @param tensor_description : description of the WholeGraph Tensor, should be 1-D or 2-D
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_make_tensor_from_handle(
  wholegraph_tensor_t* wholegraph_tensor,
  wholegraph_handle_t wholegraph_handle,
  wholegraph_tensor_description_t* tensor_description);

/**
 * Check if has WholeGraph Handle, WholeGraph Tensor created by wholegraph_make_tensor has no
 * Handle
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : if has WholeGraph Handle
 */
bool wholegraph_tensor_has_handle(wholegraph_tensor_t wholegraph_tensor);

/**
 * Get WholeGraph handle from WholeGraph Tensor
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : WholeGraph handle
 */
wholegraph_handle_t wholegraph_tensor_get_memory_handle(wholegraph_tensor_t wholegraph_tensor);

/**
 * Get tensor description from WholeGraph Tensor
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : pointer to the underlying wholegraph_tensor_description_t
 */
wholegraph_tensor_description_t* wholegraph_tensor_get_tensor_description(
  wholegraph_tensor_t wholegraph_tensor);

/**
 * Get global reference from WholeGraph Tensor
 * @param wholegraph_tensor : WholeGraph Tensor
 * @param wholegraph_gref : global reference
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_tensor_get_global_reference(
  wholegraph_tensor_t wholegraph_tensor, wholegraph_gref_t* wholegraph_gref);

/**
 * Map local tensor of WholeGraph Tensor.
 * Only support 1D and 2D tensor with WholeGraph Handle.
 * For 1D tensor, storage_offset should be 0
 * For 2D tensor, storage_offset + size[1] should <= stride[0]
 *
 * @param wholegraph_tensor : WholeGraph Tensor.
 * @param local_tensor : returned local tensor, need to be destroyed.
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_tensor_map_local_tensor(
  wholegraph_tensor_t wholegraph_tensor, wholegraph_tensor_t* local_tensor);

/**
 * Get data pointer from WholeGraph Tensor
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : Pointer to first data for CONTINUOUS WholeGraph or not WholeGraph.
 */
void* wholegraph_tensor_get_data_pointer(wholegraph_tensor_t wholegraph_tensor);

/**
 * Get entry offset of each rank from WholeGraph Tensor
 * @param entry_offsets : returned entry offset of each rank
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_tensor_get_entry_offsets(
  size_t* entry_offsets, wholegraph_tensor_t wholegraph_tensor);

/**
 * Get entry count of each rank from WholeGraph Tensor
 * @param entry_partition : returned entry count of each rank
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_tensor_get_entry_partition_sizes(
  size_t* entry_partition, wholegraph_tensor_t wholegraph_tensor);

/**
 * Get entry count of current rank from WholeGraph Tensor
 * @param local_entry_count  : returned entry count of current rank
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_tensor_get_local_entry_count(
  size_t* local_entry_count, wholegraph_tensor_t wholegraph_tensor);

/**
 * Get entry start of current rank from WholeGraph Tensor
 * @param local_entry_start  : returned entry start id of current rank
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_tensor_get_local_entry_start(
  size_t* local_entry_start, wholegraph_tensor_t wholegraph_tensor);

/**
 * Get sub tensor of a WholeGraph Tensor
 * @param wholegraph_tensor : WholeGraph Tensor
 * @param starts : starts of each dim, length should be the dim of wholegraph_tensor.
 * @param ends : ends of each dim, length should be the dim of wholegraph_tensor
 * @param sub_wholegraph_tensor : pointer to returned sub tensor
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_tensor_get_subtensor(
  wholegraph_tensor_t wholegraph_tensor,
  int64_t* starts,
  int64_t* ends,
  wholegraph_tensor_t* sub_wholegraph_tensor);

/**
 * Get root tensor of a WholeGraph Tensor, root means it is not a sub tensor of any WholeGraph
 * Tensor.
 * @param wholegraph_tensor : WholeGraph Tensor
 * @return : the root of current WholeGraph tensor, maybe same as wholegraph_tensor.
 */
wholegraph_tensor_t wholegraph_tensor_get_root(wholegraph_tensor_t wholegraph_tensor);

#define WG_TENSOR_COUNT_DEBUG
int64_t get_wholegraph_tensor_count();

#ifdef __cplusplus
}
#endif
