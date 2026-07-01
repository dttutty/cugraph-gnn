/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholegraph/env_func_ptrs.h>
#include <wholegraph/wholegraph_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to WholeGraph Embedding Cache Policy
 *
 * An Opaque handle to WholeGraph Embedding Cache Policy
 */
typedef struct wholegraph_embedding_cache_policy_* wholegraph_embedding_cache_policy_t;

/**
 * @brief Opaque handle to WholeGraph Embedding Optimizer
 *
 * An Opaque handle to WholeGraph Embedding Optimizer
 */
typedef struct wholegraph_embedding_optimizer_* wholegraph_embedding_optimizer_t;

/**
 * @brief Opaque handle to WholeGraph Embedding
 *
 * An Opaque handle to WholeGraph Embedding
 */
typedef struct wholegraph_embedding_* wholegraph_embedding_t;

/**
 * @enum wholegraph_access_type_t
 * @brief defines access type of WholeGraph Embedding
 */
enum wholegraph_access_type_t {
  WHOLEGRAPH_AT_NONE = 0,  /*!< Not defined */
  WHOLEGRAPH_AT_READONLY,  /*!< Only have readonly access to the WholeGraph */
  WHOLEGRAPH_AT_READWRITE, /*!< May have write access to the WholeGraph */
};

/**
 * @enum wholegraph_optimizer_type_t
 * @brief defines optimizer type for WholeGraph Embedding
 */
enum wholegraph_optimizer_type_t {
  WHOLEGRAPH_OPT_NONE = 0,  /*!< No optimizer needed */
  WHOLEGRAPH_OPT_SGD,       /*!< Use SGD optimizer */
  WHOLEGRAPH_OPT_LAZY_ADAM, /*!< Use Lazy Adam optimizer */
  WHOLEGRAPH_OPT_RMSPROP,   /*!< Use RMSProp optimizer */
  WHOLEGRAPH_OPT_ADAGRAD,   /*!< Use AdaGrad optimizer */
};

/**
 * Create Optimizer
 * @param optimizer : Returned wholegraph_embedding_optimizer_t
 * @param optimizer_type : Optimizer type
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_create_embedding_optimizer(
  wholegraph_embedding_optimizer_t* optimizer, wholegraph_optimizer_type_t optimizer_type);

/**
 * Set parameter for optimizer.
 * @param optimizer : Optimizer to set parameter
 * @param parameter_name : parameter name
 * @param value : parameter value
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_optimizer_set_parameter(
  wholegraph_embedding_optimizer_t optimizer, const char* parameter_name, void* value);

/**
 * Destroy optimizer
 * @param optimizer : optimizer to destroy.
 */
void wholegraph_destroy_embedding_optimizer(wholegraph_embedding_optimizer_t optimizer);

/**
 * Create WholeGraph Embedding Cache Policy
 * @param cache_policy : Returned wholegraph_embedding_cache_policy_t
 * @param cache_level_comm : At which level to cache the full embedding. In most cases it should be
 * same as wholegraph_embedding_t's comm. If access_type is WHOLEGRAPH_AT_READONLY, it can be
 * different for multiple readonly caches. E.g. for a multi-node WHOLEGRAPH_MT_DISTRIBUTED
 * WHOLEGRAPH_AT_READONLY embedding, it can have an intra-node WHOLEGRAPH_MT_CONTINUOUS cache or a
 * multi-node WHOLEGRAPH_MT_DISTRIBUTED cache.
 * @param memory_type : Memory Type of the underlying WholeGraph for cache
 * @param memory_location : Memory Location of the underlying WholeGraph for cache
 * @param access_type : ReadOnly or ReadWrite
 * @param cache_ratio : suggested cache ratio, values should be in range [1.0 / 512, 1.0]
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_create_embedding_cache_policy(
  wholegraph_embedding_cache_policy_t* cache_policy,
  wholegraph_comm_t cache_level_comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  wholegraph_access_type_t access_type,
  float cache_ratio);

/**
 * Destroy WholeGraph Embedding Cache Policy
 * @param cache_policy : WholeGraph Embedding Cache Policy to destroy.
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_destroy_embedding_cache_policy(
  wholegraph_embedding_cache_policy_t cache_policy);

/**
 * Create WholeGraph Embedding
 * @param wholegraph_embedding : Returned wholegraph_embedding_t
 * @param embedding_tensor_description : Description of the embedding, sizes and dtype used, stride
 * and storage_offset ignored. Must be matrix
 * @param comm : WholeGraph Communicator
 * @param memory_type : Memory Type of the underlying WholeGraph
 * @param memory_location : Memory Location of the underlying WholeGraph
 * @param cache_policy : Cache policy for this embedding, if don't use cache, use nullptr
 * @param embedding_entry_partition: Embedding entry count of each rank, the length must be
 * world_size
 * @param user_defined_sms : User-defined sms number for raw embedding gather/scatter
 * @param round_robin_size : continuous embedding size in each rank under round-robin shard mode
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_create_embedding(
  wholegraph_embedding_t* wholegraph_embedding,
  wholegraph_tensor_description_t* embedding_tensor_description,
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  wholegraph_embedding_cache_policy_t cache_policy,
  size_t* embedding_entry_partition = nullptr,
  int user_defined_sms              = -1,
  int round_robin_size              = 0);

/**
 * Destroy WholeGraph Embedding
 * @param wholegraph_embedding : WholeGraph Embedding to destroy
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_destroy_embedding(
  wholegraph_embedding_t wholegraph_embedding);

/**
 * Get WholeGraph Tensor from WholeGraph Embedding.
 * @param wholegraph_embedding : WholeGraph Embedding
 * @return : WholeGraph Tensor
 */
wholegraph_tensor_t wholegraph_embedding_get_embedding_tensor(
  wholegraph_embedding_t wholegraph_embedding);

/**
 * Set Optimizer for WholeGraph Embedding
 * @param wholegraph_embedding : WholeGraph Embedding
 * @param optimizer : Optimizer to be set
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_embedding_set_optimizer(
  wholegraph_embedding_t wholegraph_embedding, wholegraph_embedding_optimizer_t optimizer);

/**
 * Gather from WholeGraph Embedding
 * @param wholegraph_embedding : WholeGraph Embedding
 * @param indices : indices to gather
 * @param output : output tensor
 * @param adjust_cache : if we should adjust cache in this gather
 * @param p_env_fns : env fns
 * @param stream_int : CUDA stream to use
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_embedding_gather(wholegraph_embedding_t wholegraph_embedding,
                                                      wholegraph_tensor_t indices,
                                                      wholegraph_tensor_t output,
                                                      bool adjust_cache,
                                                      wholegraph_env_func_t* p_env_fns,
                                                      int64_t stream_int);

/**
 * Gather backward for WholeGraph Embedding
 * @param wholegraph_embedding : WholeGraph Embedding
 * @param indices : indices to gather
 * @param grads : gradient of output tensor
 * @param adjust_cache : if we should adjust cache in this gather
 * @param lr : learning rate of current step.
 * @param p_env_fns : env fns
 * @param stream_int : CUDA stream to use
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_embedding_gather_gradient_apply(
  wholegraph_embedding_t wholegraph_embedding,
  wholegraph_tensor_t indices,
  wholegraph_tensor_t grads,
  bool adjust_cache,
  float lr,
  wholegraph_env_func_t* p_env_fns,
  int64_t stream_int);

/**
 * Get optimizer internal state names
 * @param wholegraph_embedding : WholeGraph Embedding
 * @return : nullptr terminated names.
 */
const char* const* wholegraph_embedding_get_optimizer_state_names(
  wholegraph_embedding_t wholegraph_embedding);

/**
 * Get optimizer internal state
 * @param wholegraph_embedding : WholeGraph Embedding
 * @param name : state name
 * @return : internal state, nullptr for not exist.
 */
wholegraph_tensor_t wholegraph_embedding_get_optimizer_state(
  wholegraph_embedding_t wholegraph_embedding, const char* name);

/**
 * Writeback all cache WholeGraph Embedding
 * @param wholegraph_embedding : WholeGraph Embedding
 * @param stream_int : CUDA stream to use.
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_embedding_writeback_cache(
  wholegraph_embedding_t wholegraph_embedding, int64_t stream_int);

/**
 * Drop all cache in WholeGraph Embedding
 * @param wholegraph_embedding : WholeGraph Embedding
 * @param stream_int : CUDA stream to use.
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_embedding_drop_all_cache(
  wholegraph_embedding_t wholegraph_embedding, int64_t stream_int);

#ifdef __cplusplus
}
#endif
