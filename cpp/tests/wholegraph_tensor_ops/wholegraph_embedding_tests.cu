/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <wholegraph/embedding.h>

#include "../wholegraph/wholegraph_test_utils.hpp"
#include "embedding_test_utils.hpp"
#include "wholegraph/env_func_ptrs.hpp"

struct EmbeddingTestParams {
  EmbeddingTestParams()
  {
    const int64_t kDefaultEmbeddingEntryCount = 4000001;
    const int64_t kDefaultEmbeddingDim        = 127;
    const int64_t kDefaultGatherIndiceCount   = 100005;
    int64_t embedding_sizes[2]                = {kDefaultEmbeddingEntryCount, kDefaultEmbeddingDim};
    embedding_description                     = wholegraph_create_matrix_desc(
      &embedding_sizes[0], kDefaultEmbeddingDim, 0, WHOLEGRAPH_DT_FLOAT);
    indice_description =
      wholegraph_create_array_desc(kDefaultGatherIndiceCount, 0, WHOLEGRAPH_DT_INT64);
    int64_t output_sizes[2] = {kDefaultGatherIndiceCount, kDefaultEmbeddingDim};
    output_description      = wholegraph_create_matrix_desc(
      &output_sizes[0], kDefaultEmbeddingDim, 0, WHOLEGRAPH_DT_FLOAT);
  }
  bool is_large_test()
  {
    int64_t embedding_table_mem_size =
      wholegraph_get_memory_element_count_from_matrix(&embedding_description) *
      wholegraph_dtype_get_element_size(embedding_description.dtype);
    if (embedding_table_mem_size > 2LL * 1024 * 1024 * 1024) return true;
    return false;
  }
  EmbeddingTestParams& set_entry_count(int64_t entry_count)
  {
    embedding_description.sizes[0] = entry_count;
    return *this;
  }
  EmbeddingTestParams& set_embedding_dim(int embedding_dim)
  {
    embedding_description.sizes[1] = embedding_dim;
    output_description.sizes[1]    = embedding_dim;
    embedding_description.stride   = embedding_dim;
    if (output_description.stride < embedding_dim) output_description.stride = embedding_dim;
    return *this;
  }
  EmbeddingTestParams& set_embedding_stride(int stride)
  {
    embedding_description.stride = stride;
    return *this;
  }
  EmbeddingTestParams& set_embedding_dtype(wholegraph_dtype_t dtype)
  {
    embedding_description.dtype = dtype;
    return *this;
  }
  EmbeddingTestParams& set_indice_count(int indice_count)
  {
    indice_description.size     = indice_count;
    output_description.sizes[0] = indice_count;
    return *this;
  }
  EmbeddingTestParams& set_indice_dtype(wholegraph_dtype_t dtype)
  {
    indice_description.dtype = dtype;
    return *this;
  }
  EmbeddingTestParams& set_output_stride(int stride)
  {
    output_description.stride = stride;
    return *this;
  }
  EmbeddingTestParams& set_output_dtype(wholegraph_dtype_t dtype)
  {
    output_description.dtype = dtype;
    return *this;
  }
  EmbeddingTestParams& set_memory_type(wholegraph_memory_type_t mt)
  {
    memory_type = mt;
    return *this;
  }
  EmbeddingTestParams& set_memory_location(wholegraph_memory_location_t ml)
  {
    memory_location = ml;
    return *this;
  }
  EmbeddingTestParams& set_cache_memory_type(wholegraph_memory_type_t cmt)
  {
    cache_memory_type = cmt;
    return *this;
  }
  EmbeddingTestParams& set_cache_memory_location(wholegraph_memory_location_t cml)
  {
    cache_memory_location = cml;
    return *this;
  }
  EmbeddingTestParams& set_cache_ratio(float ratio)
  {
    cache_ratio = ratio;
    return *this;
  }
  wholegraph_embedding_cache_policy_t get_cache_policy(wholegraph_comm_t comm)
  {
    wholegraph_embedding_cache_policy_t cache_policy = nullptr;
    if (cache_type == 0) return nullptr;
    EXPECT_EQ(wholegraph_create_embedding_cache_policy(
                &cache_policy,
                comm,
                cache_memory_type,
                cache_memory_location,
                cache_type == 1 ? WHOLEGRAPH_AT_READWRITE : WHOLEGRAPH_AT_READONLY,
                cache_ratio),
              WHOLEGRAPH_SUCCESS);
    return cache_policy;
  }
  int get_rank_partition_method() const { return rank_partition_method; }
  EmbeddingTestParams& non_cache()
  {
    cache_type = 0;
    return *this;
  }
  EmbeddingTestParams& device_cache()
  {
    cache_type = 1;
    return *this;
  }
  EmbeddingTestParams& local_cache()
  {
    cache_type = 2;
    return *this;
  }
  EmbeddingTestParams& set_cache_group_count(int count)
  {
    cache_group_count = count;
    return *this;
  }
  EmbeddingTestParams& use_random_partition()
  {
    rank_partition_method = 1;
    return *this;
  }
  wholegraph_array_description_t indice_description;
  wholegraph_matrix_description_t embedding_description;
  wholegraph_matrix_description_t output_description;
  wholegraph_memory_type_t memory_type               = WHOLEGRAPH_MT_CONTINUOUS;
  wholegraph_memory_location_t memory_location       = WHOLEGRAPH_ML_HOST;
  wholegraph_memory_type_t cache_memory_type         = WHOLEGRAPH_MT_CONTINUOUS;
  wholegraph_memory_location_t cache_memory_location = WHOLEGRAPH_ML_DEVICE;
  float cache_ratio                                   = 0.2;
  int cache_type            = 0;  // 0: no cache, 1: device cache, 2: local cache
  int cache_group_count     = 1;
  int rank_partition_method = 0;  // 0-default, 1-random
};

class WholeGraphEmbeddingParameterTests : public ::testing::TestWithParam<EmbeddingTestParams> {};

TEST_P(WholeGraphEmbeddingParameterTests, EmbeddingGatherTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  if (dev_count % params.cache_group_count != 0) {
    GTEST_SKIP() << "skipping test due to not enough GPUs group count=" << params.cache_group_count
                 << ", but GPU count=" << dev_count;
  }
  if (dev_count == 1 && params.is_large_test()) {
    GTEST_SKIP() << "skipping large test on single gpu";
  }
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(dev_count, [&params, &pipes](int world_rank, int world_size) {
    EXPECT_EQ(wholegraph_init(0), WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(cudaSetDevice(world_rank), cudaSuccess);
    wholegraph_comm_t wg_comm    = create_communicator_by_pipes(pipes, world_rank, world_size);
    wholegraph_comm_t cache_comm = wg_comm;

    if (wholegraph_communicator_support_type_location(
          wg_comm, params.memory_type, params.memory_location) != WHOLEGRAPH_SUCCESS) {
      EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
      WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
      GTEST_SKIP_("Skip due to not supported.");
      return;
    }

    if (params.cache_type == 2) {
      cache_comm =
        create_group_communicator_by_pipes(pipes, world_rank, world_size, params.cache_group_count);
    }

    if ((params.cache_type == 1 ||
         params.cache_type == 2 && params.cache_group_count < world_size) &&
        wholegraph_communicator_support_type_location(
          wg_comm, params.cache_memory_type, params.cache_memory_location) != WHOLEGRAPH_SUCCESS) {
      EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
      WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
      GTEST_SKIP_("Skip due to cache memory type/location not supported.");
      return;
    }

    void *dev_indices = nullptr, *dev_gather_buffer = nullptr, *dev_reference_buffer = nullptr;
    void *host_indices = nullptr, *host_gather_buffer = nullptr, *host_reference_buffer = nullptr;
    size_t gather_buffer_size = wholegraph_get_memory_size_from_matrix(&params.output_description);
    size_t indices_buffer_size = wholegraph_get_memory_size_from_array(&params.indice_description);

    cudaStream_t stream;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    EXPECT_EQ(cudaMallocHost(&host_indices, indices_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_indices, indices_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_gather_buffer, gather_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_reference_buffer, gather_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMallocHost(&host_gather_buffer, gather_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMallocHost(&host_reference_buffer, gather_buffer_size), cudaSuccess);

    wholegraph_tensor_t indices_tensor, output_tensor;
    wholegraph_tensor_description_t indices_tensor_desc, output_tensor_desc;
    wholegraph_copy_array_desc_to_tensor(&indices_tensor_desc, &params.indice_description);
    wholegraph_copy_matrix_desc_to_tensor(&output_tensor_desc, &params.output_description);
    EXPECT_EQ(
      wholegraph_make_tensor_from_pointer(&indices_tensor, dev_indices, &indices_tensor_desc),
      WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(
      wholegraph_make_tensor_from_pointer(&output_tensor, dev_gather_buffer, &output_tensor_desc),
      WHOLEGRAPH_SUCCESS);

    wholegraph_embedding_cache_policy_t cache_policy = params.get_cache_policy(cache_comm);

    wholegraph_embedding_t wg_embedding;
    wholegraph_tensor_description_t embedding_tensor_description;
    wholegraph_copy_matrix_desc_to_tensor(&embedding_tensor_description,
                                           &params.embedding_description);
    std::vector<size_t> rank_partition(world_size);
    wholegraph_tensor_ops::testing::host_random_partition(
      rank_partition.data(), embedding_tensor_description.sizes[0], world_size);
    size_t* rank_partition_ptr = nullptr;
    if (params.get_rank_partition_method() == 1) { rank_partition_ptr = rank_partition.data(); }
    EXPECT_EQ(wholegraph_create_embedding(&wg_embedding,
                                           &embedding_tensor_description,
                                           wg_comm,
                                           params.memory_type,
                                           params.memory_location,
                                           cache_policy,
                                           rank_partition_ptr),
              WHOLEGRAPH_SUCCESS);

    wholegraph_tensor_t embedding_tensor =
      wholegraph_embedding_get_embedding_tensor(wg_embedding);
    wholegraph_handle_t embedding_handle = wholegraph_tensor_get_memory_handle(embedding_tensor);

    wholegraph_matrix_description_t embedding_matrix_desc;
    EXPECT_TRUE(wholegraph_convert_tensor_desc_to_matrix(
      &embedding_matrix_desc, wholegraph_tensor_get_tensor_description(embedding_tensor)));
    wholegraph_tensor_ops::testing::device_random_init_local_embedding_table(
      embedding_handle, embedding_matrix_desc, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    wholegraph_communicator_barrier(wg_comm);

    for (int i = 0; i < 10; i++) {
      wholegraph_tensor_ops::testing::host_random_init_indices(
        host_indices, params.indice_description, params.embedding_description.sizes[0]);
      EXPECT_EQ(cudaMemcpyAsync(dev_indices,
                                host_indices,
                                wholegraph_get_memory_size_from_array(&params.indice_description),
                                cudaMemcpyHostToDevice,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholegraph_communicator_barrier(wg_comm);
      EXPECT_EQ(wholegraph_embedding_gather(wg_embedding,
                                             indices_tensor,
                                             output_tensor,
                                             i % 2 == 0,
                                             wholegraph::get_default_env_func(),
                                             (int64_t)stream),
                WHOLEGRAPH_SUCCESS);

      wholegraph_tensor_ops::testing::device_get_expected_embedding(dev_reference_buffer,
                                                              params.output_description,
                                                              params.embedding_description.dtype,
                                                              dev_indices,
                                                              params.indice_description,
                                                              wholegraph::get_default_env_func(),
                                                              stream);
      EXPECT_EQ(cudaMemcpyAsync(host_gather_buffer,
                                dev_gather_buffer,
                                wholegraph_get_memory_size_from_matrix(&params.output_description),
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpyAsync(host_reference_buffer,
                                dev_reference_buffer,
                                wholegraph_get_memory_size_from_matrix(&params.output_description),
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

      wholegraph_tensor_ops::testing::host_check_embedding_same(host_gather_buffer,
                                                          params.output_description,
                                                          host_reference_buffer,
                                                          params.output_description);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholegraph_communicator_barrier(wg_comm);
    }

    EXPECT_EQ(wholegraph_destroy_embedding_cache_policy(cache_policy), WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(wholegraph_destroy_tensor(indices_tensor), WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(wholegraph_destroy_tensor(output_tensor), WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(cudaFreeHost(host_indices), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_indices), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_gather_buffer), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_reference_buffer), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(host_gather_buffer), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(host_reference_buffer), cudaSuccess);

    EXPECT_EQ(wholegraph_destroy_embedding(wg_embedding), WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
    WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
  });
}

INSTANTIATE_TEST_SUITE_P(
  CachedEmbeddingGatherTest,
  WholeGraphEmbeddingParameterTests,
  ::testing::Values(

    EmbeddingTestParams()
      .local_cache()
      .set_entry_count((1LL << 22LL) + 131)
      .set_embedding_dim(256)
      .set_cache_group_count(2)
      .set_cache_ratio(0.1),
    EmbeddingTestParams()
      .local_cache()
      .set_entry_count((1LL << 22LL) + 131)
      .set_embedding_dim(256)
      .set_cache_group_count(4)
      .set_cache_ratio(0.05),
    EmbeddingTestParams()
      .local_cache()
      .set_entry_count((1LL << 22LL) + 131)
      .set_embedding_dim(256)
      .set_cache_group_count(8)
      .set_cache_ratio(0.02),
#if 1
    EmbeddingTestParams().non_cache(),
    EmbeddingTestParams().non_cache().set_memory_location(WHOLEGRAPH_ML_DEVICE),
    EmbeddingTestParams()
      .non_cache()
      .set_memory_location(WHOLEGRAPH_ML_DEVICE)
      .use_random_partition(),
    EmbeddingTestParams()
      .non_cache()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .use_random_partition(),
    EmbeddingTestParams()
      .non_cache()
      .set_memory_location(WHOLEGRAPH_ML_DEVICE)
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .use_random_partition(),
    EmbeddingTestParams().device_cache(),
    EmbeddingTestParams().device_cache().set_cache_memory_type(WHOLEGRAPH_MT_DISTRIBUTED),
    EmbeddingTestParams().local_cache(),
    EmbeddingTestParams().local_cache().set_cache_memory_location(WHOLEGRAPH_ML_HOST),
    EmbeddingTestParams()
      .local_cache()
      .set_memory_location(WHOLEGRAPH_ML_DEVICE)
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED),

    EmbeddingTestParams().device_cache().set_cache_ratio(0.002),
    EmbeddingTestParams().local_cache().set_cache_ratio(0.002),

    EmbeddingTestParams()
      .device_cache()
      .set_cache_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_cache_ratio(0.002),
    EmbeddingTestParams()
      .local_cache()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_cache_ratio(0.002),

    EmbeddingTestParams().non_cache().set_output_dtype(WHOLEGRAPH_DT_HALF),
    EmbeddingTestParams().device_cache().set_output_dtype(WHOLEGRAPH_DT_HALF),
    EmbeddingTestParams().local_cache().set_output_dtype(WHOLEGRAPH_DT_HALF),

    EmbeddingTestParams().non_cache().set_indice_dtype(WHOLEGRAPH_DT_INT),
    EmbeddingTestParams().device_cache().set_indice_dtype(WHOLEGRAPH_DT_INT),
    EmbeddingTestParams().local_cache().set_indice_dtype(WHOLEGRAPH_DT_INT),

    EmbeddingTestParams().non_cache().set_embedding_dtype(WHOLEGRAPH_DT_HALF),
    EmbeddingTestParams().device_cache().set_embedding_dtype(WHOLEGRAPH_DT_HALF),
    EmbeddingTestParams().local_cache().set_embedding_dtype(WHOLEGRAPH_DT_HALF),

    EmbeddingTestParams()
      .non_cache()
      .set_memory_location(WHOLEGRAPH_ML_DEVICE)
      .set_output_dtype(WHOLEGRAPH_DT_HALF),
    EmbeddingTestParams()
      .device_cache()
      .set_cache_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_output_dtype(WHOLEGRAPH_DT_HALF),
    EmbeddingTestParams()
      .local_cache()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_output_dtype(WHOLEGRAPH_DT_HALF),

    EmbeddingTestParams().non_cache().set_embedding_dim(131),
    EmbeddingTestParams().device_cache().set_embedding_dim(131),
    EmbeddingTestParams().local_cache().set_embedding_dim(131),

    EmbeddingTestParams().non_cache().set_embedding_dim(11).set_output_stride(11),
    EmbeddingTestParams().device_cache().set_embedding_dim(11).set_output_stride(11),
    EmbeddingTestParams().local_cache().set_embedding_dim(11).set_output_stride(11),

    EmbeddingTestParams().non_cache().set_embedding_dim(1157).set_entry_count(300000),
    EmbeddingTestParams().device_cache().set_embedding_dim(1157).set_entry_count(300000),
    EmbeddingTestParams().local_cache().set_embedding_dim(1157).set_entry_count(300000),

    EmbeddingTestParams().non_cache().set_output_stride(131),
    EmbeddingTestParams().device_cache().set_output_stride(131),
    EmbeddingTestParams().local_cache().set_output_stride(131),
    // large tests
    EmbeddingTestParams()
      .non_cache()
      .set_entry_count((1LL << 32LL) + 127)
      .set_embedding_dim(3)
      .set_embedding_stride(3),
    EmbeddingTestParams()
      .device_cache()
      .set_entry_count((1LL << 32LL) + 127)
      .set_embedding_dim(3)
      .set_embedding_stride(3),
    EmbeddingTestParams()
      .local_cache()
      .set_entry_count((1LL << 32LL) + 127)
      .set_embedding_dim(3)
      .set_embedding_stride(3),
    EmbeddingTestParams()
      .non_cache()
      .set_entry_count((1LL << 31LL) - 127)
      .set_embedding_dim(5)
      .set_embedding_stride(5)
      .set_indice_dtype(WHOLEGRAPH_DT_INT),
    EmbeddingTestParams()
      .device_cache()
      .set_entry_count((1LL << 31LL) - 127)
      .set_indice_dtype(WHOLEGRAPH_DT_INT)
      .set_embedding_dim(5)
      .set_embedding_stride(5),
    EmbeddingTestParams()
      .local_cache()
      .set_entry_count((1LL << 31LL) - 127)
      .set_indice_dtype(WHOLEGRAPH_DT_INT)
      .set_embedding_dim(5)
      .set_embedding_stride(5),

    EmbeddingTestParams().non_cache().set_entry_count((1LL << 20LL) + 131).set_embedding_dim(1024),
    EmbeddingTestParams()
      .device_cache()
      .set_entry_count((1LL << 20LL) + 131)
      .set_embedding_dim(1024),
    EmbeddingTestParams()
      .local_cache()
      .set_entry_count((1LL << 20LL) + 131)
      .set_embedding_dim(1024),

    EmbeddingTestParams().non_cache().set_entry_count((1LL << 23LL) + 127).set_embedding_dim(1025),
    EmbeddingTestParams()
      .device_cache()
      .set_entry_count((1LL << 23LL) + 127)
      .set_embedding_dim(1025),
    EmbeddingTestParams()
      .local_cache()
      .set_entry_count((1LL << 23LL) + 127)
      .set_embedding_dim(1025),

    EmbeddingTestParams()
      .non_cache()
      .set_entry_count((1LL << 22LL) + 131)
      .set_embedding_dim(11)
      .set_embedding_stride(12),
    EmbeddingTestParams()
      .device_cache()
      .set_entry_count((1LL << 22LL) + 131)
      .set_embedding_dim(11)
      .set_embedding_stride(12),
    EmbeddingTestParams()
      .local_cache()
      .set_entry_count((1LL << 22LL) + 131)
      .set_embedding_dim(11)
      .set_embedding_stride(12),

#endif
    EmbeddingTestParams()));
