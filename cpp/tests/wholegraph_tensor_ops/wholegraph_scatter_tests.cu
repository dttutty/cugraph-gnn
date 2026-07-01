/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>
#include <wholegraph/wholegraph_tensor_op.h>

#include "parallel_utils.hpp"
#include "wholegraph/communicator.hpp"
#include "wholegraph/env_func_ptrs.hpp"

#include "../wholegraph/wholegraph_test_utils.hpp"
#include "embedding_test_utils.hpp"

typedef struct WholeGraphScatterTestParam {
  wholegraph_matrix_description_t get_embedding_desc() const
  {
    int64_t matrix_sizes[2] = {embedding_entry_count, embedding_dim};
    return wholegraph_create_matrix_desc(
      matrix_sizes, embedding_stride, embedding_storage_offset, embedding_type);
  }
  wholegraph_array_description_t get_indices_desc() const
  {
    return wholegraph_create_array_desc(indices_count, indices_storage_offset, indices_type);
  }
  wholegraph_matrix_description_t get_input_desc() const
  {
    int64_t input_sizes[2] = {indices_count, embedding_dim};
    return wholegraph_create_matrix_desc(
      input_sizes, input_stride, input_storage_offset, input_type);
  }
  int64_t get_embedding_granularity() const
  {
    return embedding_stride * wholegraph_dtype_get_element_size(embedding_type);
  }
  int get_rank_partition_method() const { return rank_partition_method; }

  WholeGraphScatterTestParam& set_memory_type(wholegraph_memory_type_t new_memory_type)
  {
    memory_type = new_memory_type;
    return *this;
  }
  WholeGraphScatterTestParam& set_memory_location(
    wholegraph_memory_location_t new_memory_location)
  {
    memory_location = new_memory_location;
    return *this;
  }
  WholeGraphScatterTestParam& set_entry_count(int64_t entry_count)
  {
    embedding_entry_count = entry_count;
    return *this;
  }
  WholeGraphScatterTestParam& set_embedding_dim(int64_t new_embedding_dim)
  {
    embedding_dim = new_embedding_dim;
    if (embedding_stride < embedding_dim) embedding_stride = embedding_dim;
    if (input_stride < embedding_dim) input_stride = embedding_dim;
    return *this;
  }
  WholeGraphScatterTestParam& set_embedding_stride(int64_t new_embedding_stride)
  {
    embedding_stride = new_embedding_stride;
    return *this;
  }
  WholeGraphScatterTestParam& set_input_stride(int64_t new_input_stride)
  {
    input_stride = new_input_stride;
    return *this;
  }
  WholeGraphScatterTestParam& set_indices_count(int64_t new_indices_count)
  {
    indices_count = new_indices_count;
    return *this;
  }
  WholeGraphScatterTestParam& set_embedding_type(wholegraph_dtype_t new_embedding_type)
  {
    embedding_type = new_embedding_type;
    return *this;
  }
  WholeGraphScatterTestParam& set_indices_type(wholegraph_dtype_t new_indices_type)
  {
    indices_type = new_indices_type;
    return *this;
  }
  WholeGraphScatterTestParam& set_input_type(wholegraph_dtype_t new_input_type)
  {
    input_type = new_input_type;
    return *this;
  }
  WholeGraphScatterTestParam& set_distributed_backend(
    wholegraph_distributed_backend_t new_distributed_backend)
  {
    distributed_backend = new_distributed_backend;
    return *this;
  }
  WholeGraphScatterTestParam& use_random_partition()
  {
    rank_partition_method = 1;
    return *this;
  }
  wholegraph_memory_type_t memory_type                 = WHOLEGRAPH_MT_CONTINUOUS;
  wholegraph_memory_location_t memory_location         = WHOLEGRAPH_ML_DEVICE;
  int64_t embedding_entry_count                         = 1000000LL;
  int64_t embedding_dim                                 = 32;
  int64_t embedding_stride                              = 32;
  int64_t indices_count                                 = 100000;
  int64_t input_stride                                  = 32;
  wholegraph_dtype_t embedding_type                    = WHOLEGRAPH_DT_FLOAT;
  wholegraph_dtype_t indices_type                      = WHOLEGRAPH_DT_INT;
  wholegraph_dtype_t input_type                        = WHOLEGRAPH_DT_FLOAT;
  int64_t embedding_storage_offset                      = 0;
  int64_t indices_storage_offset                        = 0;
  int64_t input_storage_offset                          = 0;
  wholegraph_distributed_backend_t distributed_backend = WHOLEGRAPH_DB_NCCL;
  int rank_partition_method                             = 0;  // 0-default, 1-random
} WholeGraphScatterTestParam;

class WholeGraphScatterParameterTests
  : public ::testing::TestWithParam<WholeGraphScatterTestParam> {};

TEST_P(WholeGraphScatterParameterTests, ScatterTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(dev_count, [&params, &pipes](int world_rank, int world_size) {
    EXPECT_EQ(wholegraph_init(0), WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(cudaSetDevice(world_rank), cudaSuccess);

    wholegraph_comm_t wg_comm = create_communicator_by_pipes(pipes, world_rank, world_size);
    if (wholegraph_communicator_support_type_location(
          wg_comm, params.memory_type, params.memory_location) != WHOLEGRAPH_SUCCESS) {
      EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
      WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
      if (world_rank == 0) GTEST_SKIP_("Skip due to not supported.");
      return;
    }

    wholegraph_handle_t embedding_handle;
    auto embedding_desc         = params.get_embedding_desc();
    auto indices_desc           = params.get_indices_desc();
    auto input_desc             = params.get_input_desc();
    size_t embedding_entry_size = params.get_embedding_granularity();
    std::vector<size_t> rank_partition(world_size);
    wholegraph_tensor_ops::testing::host_random_partition(
      rank_partition.data(), embedding_desc.sizes[0], world_size);
    size_t* rank_partition_ptr = nullptr;
    if (params.get_rank_partition_method() == 1) { rank_partition_ptr = rank_partition.data(); }
    EXPECT_EQ(wholegraph_malloc(&embedding_handle,
                                 wholegraph_get_memory_size_from_matrix(&embedding_desc),
                                 wg_comm,
                                 params.memory_type,
                                 params.memory_location,
                                 embedding_entry_size,
                                 rank_partition_ptr),
              WHOLEGRAPH_SUCCESS);

    cudaStream_t stream;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    void *host_indices = nullptr, *dev_indices = nullptr, *dev_input_buffer = nullptr,
         *dev_gather_buffer  = nullptr;
    void *host_gather_buffer = nullptr, *host_input_buffer = nullptr;
    size_t scatter_buffer_size = wholegraph_get_memory_size_from_matrix(&input_desc);
    size_t indices_buffer_size = wholegraph_get_memory_size_from_array(&indices_desc);

    EXPECT_EQ(cudaMallocHost(&host_indices, indices_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_indices, indices_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_input_buffer, scatter_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_gather_buffer, scatter_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMallocHost(&host_input_buffer, scatter_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMallocHost(&host_gather_buffer, scatter_buffer_size), cudaSuccess);

    wholegraph_tensor_ops::testing::host_random_init_indices(
      host_indices, indices_desc, embedding_desc.sizes[0]);
    EXPECT_EQ(cudaMemcpyAsync(dev_indices,
                              host_indices,
                              wholegraph_get_memory_size_from_array(&indices_desc),
                              cudaMemcpyHostToDevice,
                              stream),
              cudaSuccess);
    wholegraph_tensor_ops::testing::device_get_expected_embedding(dev_input_buffer,
                                                            input_desc,
                                                            embedding_desc.dtype,
                                                            dev_indices,
                                                            indices_desc,
                                                            wholegraph::get_default_env_func(),
                                                            stream);

    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    wholegraph_communicator_barrier(wg_comm);

    wholegraph_tensor_t embedding_tensor;
    wholegraph_tensor_description_t embedding_tensor_desc;
    wholegraph_copy_matrix_desc_to_tensor(&embedding_tensor_desc, &embedding_desc);
    EXPECT_EQ(wholegraph_make_tensor_from_handle(
                &embedding_tensor, embedding_handle, &embedding_tensor_desc),
              WHOLEGRAPH_SUCCESS);

    wholegraph_tensor_t indices_tensor, input_tensor;
    wholegraph_tensor_description_t indices_tensor_desc, input_tensor_desc;
    wholegraph_copy_array_desc_to_tensor(&indices_tensor_desc, &indices_desc);
    wholegraph_copy_matrix_desc_to_tensor(&input_tensor_desc, &input_desc);
    EXPECT_EQ(
      wholegraph_make_tensor_from_pointer(&indices_tensor, dev_indices, &indices_tensor_desc),
      WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(
      wholegraph_make_tensor_from_pointer(&input_tensor, dev_input_buffer, &input_tensor_desc),
      WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(wholegraph_scatter(input_tensor,
                                  indices_tensor,
                                  embedding_tensor,
                                  wholegraph::get_default_env_func(),
                                  stream),
              WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(wholegraph_destroy_tensor(input_tensor), WHOLEGRAPH_SUCCESS);

    wholegraph_communicator_barrier(wg_comm);

    wholegraph_tensor_t gathered_tensor;
    EXPECT_EQ(
      wholegraph_make_tensor_from_pointer(&gathered_tensor, dev_gather_buffer, &input_tensor_desc),
      WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(wholegraph_gather(embedding_tensor,
                                 indices_tensor,
                                 gathered_tensor,
                                 wholegraph::get_default_env_func(),
                                 stream),
              WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(cudaMemcpyAsync(host_gather_buffer,
                              dev_gather_buffer,
                              wholegraph_get_memory_size_from_matrix(&input_desc),
                              cudaMemcpyDeviceToHost,
                              stream),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpyAsync(host_input_buffer,
                              dev_input_buffer,
                              wholegraph_get_memory_size_from_matrix(&input_desc),
                              cudaMemcpyDeviceToHost,
                              stream),
              cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(wholegraph_destroy_tensor(gathered_tensor), WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(wholegraph_destroy_tensor(indices_tensor), WHOLEGRAPH_SUCCESS);

    wholegraph_tensor_ops::testing::host_check_embedding_same(
      host_gather_buffer, input_desc, host_input_buffer, input_desc);

    EXPECT_EQ(cudaFreeHost(host_indices), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_indices), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_input_buffer), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_gather_buffer), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(host_input_buffer), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(host_gather_buffer), cudaSuccess);

    EXPECT_EQ(wholegraph_destroy_tensor(embedding_tensor), WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(wholegraph_free(embedding_handle), WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
    WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
  });
}

INSTANTIATE_TEST_SUITE_P(
  WholeGraphScatterOpTests,
  WholeGraphScatterParameterTests,
  ::testing::Values(
#if 1
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_indices_count(0),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_indices_count(0),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED).set_indices_count(0),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_memory_location(WHOLEGRAPH_ML_HOST),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_memory_location(WHOLEGRAPH_ML_HOST),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_memory_location(WHOLEGRAPH_ML_HOST),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).use_random_partition(),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .use_random_partition(),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_memory_location(WHOLEGRAPH_ML_HOST)
      .use_random_partition(),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_dim(128),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_dim(128),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_dim(128),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_dim(11)
      .set_indices_count(100005),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_dim(11)
      .set_indices_count(100005),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_dim(11)
      .set_indices_count(100005),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_dim(127),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_dim(127),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_dim(127),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_dim(129),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_dim(129),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_dim(129),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_dim(513),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_dim(513),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_dim(513),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_input_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_input_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_input_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_type(WHOLEGRAPH_DT_HALF)
      .set_input_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_type(WHOLEGRAPH_DT_HALF)
      .set_input_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_type(WHOLEGRAPH_DT_HALF)
      .set_input_type(WHOLEGRAPH_DT_HALF),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_indices_type(WHOLEGRAPH_DT_INT64),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_indices_type(WHOLEGRAPH_DT_INT64),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_indices_type(WHOLEGRAPH_DT_INT64),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_stride(33),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_embedding_stride(33),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_stride(33),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_input_stride(33),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_CONTINUOUS).set_input_stride(33),
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED).set_input_stride(33),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_type(WHOLEGRAPH_DT_HALF)
      .set_embedding_stride(33),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
      .set_embedding_type(WHOLEGRAPH_DT_HALF)
      .set_embedding_stride(33),
    WholeGraphScatterTestParam()
      .set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      .set_embedding_type(WHOLEGRAPH_DT_HALF)
      .set_embedding_stride(33),
#endif
    WholeGraphScatterTestParam().set_memory_type(WHOLEGRAPH_MT_DISTRIBUTED)
      ));
