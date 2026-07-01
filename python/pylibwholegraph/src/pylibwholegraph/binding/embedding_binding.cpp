/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "embedding_binding.hpp"

WholeGraphCachePolicyNB create_non_cache_policy() { return WholeGraphCachePolicyNB{}; }

PyWholeGraphEmbeddingNB create_embedding(
  PyWholeGraphTensorDescriptionNB const& tensor_description,
  PyWholeGraphCommNB const& comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location,
  WholeGraphCachePolicyNB const& cache_policy,
  nb::object embedding_entry_partition,
  int user_defined_sms,
  int round_robin_size)
{
  wholegraph_tensor_description_t desc = *tensor_description.c_ptr();
  wholegraph_embedding_t embedding     = nullptr;
  std::vector<size_t> partition        = size_vector_from_iterable(embedding_entry_partition);
  check_wholegraph_error_code(wholegraph_create_embedding(
    &embedding,
    &desc,
    comm.c_handle(),
    memory_type,
    memory_location,
    cache_policy.c_handle(),
    partition.empty() ? nullptr : partition.data(),
    user_defined_sms,
    round_robin_size));
  return PyWholeGraphEmbeddingNB::from_c_handle(embedding);
}
