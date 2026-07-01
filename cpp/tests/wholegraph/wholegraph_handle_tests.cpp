/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include "parallel_utils.hpp"
#include "wholegraph/communicator.hpp"
#include "wholegraph/initialize.hpp"
#include "wholegraph/memory_handle.hpp"

#include "wholegraph_test_utils.hpp"

class WholeGraphHandleCreateDestroyParameterTests
  : public ::testing::TestWithParam<
      std::tuple<size_t, wholegraph_memory_type_t, wholegraph_memory_location_t, size_t>> {};

TEST_P(WholeGraphHandleCreateDestroyParameterTests, CreateDestroyTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEGRAPH_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(
    nproc,
    [&pipes, &params](int rank, int world_size) {
      EXPECT_EQ(wholegraph_init(0), WHOLEGRAPH_SUCCESS);

      EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

      wholegraph_comm_t wg_comm = create_communicator_by_pipes(pipes, rank, world_size);

      if (wholegraph_communicator_support_type_location(
            wg_comm, std::get<1>(params), std::get<2>(params)) != WHOLEGRAPH_SUCCESS) {
        EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);
        EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
        WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
        if (rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      wholegraph_handle_t handle1;
      EXPECT_EQ(wholegraph::create_wholegraph(&handle1,
                                                std::get<0>(params),
                                                wg_comm,
                                                std::get<1>(params),
                                                std::get<2>(params),
                                                std::get<2>(params)),
                WHOLEGRAPH_SUCCESS);

      EXPECT_EQ(wholegraph::destroy_wholegraph(handle1), WHOLEGRAPH_SUCCESS);

      EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);

      EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);

      WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
  ClosePipes(&pipes);
}

INSTANTIATE_TEST_SUITE_P(
  WholeGraphHandleTests,
  WholeGraphHandleCreateDestroyParameterTests,
  ::testing::Values(
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_DEVICE, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_HOST, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_DEVICE, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_HOST, 128UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_DISTRIBUTED, WHOLEGRAPH_ML_DEVICE, 128UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_DISTRIBUTED, WHOLEGRAPH_ML_HOST, 128UL),

    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_HOST, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_HOST, 63UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_DISTRIBUTED, WHOLEGRAPH_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_DISTRIBUTED, WHOLEGRAPH_ML_HOST, 63UL),

    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_HOST, 128UL)));

class WholeGraphHandleMultiCreateParameterTests
  : public ::testing::TestWithParam<
      std::tuple<wholegraph_memory_type_t, wholegraph_memory_location_t>> {};

TEST_P(WholeGraphHandleMultiCreateParameterTests, CreateDestroyTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEGRAPH_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(
    nproc,
    [&pipes, &params](int rank, int world_size) {
      EXPECT_EQ(wholegraph_init(0), WHOLEGRAPH_SUCCESS);

      EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

      wholegraph_comm_t wg_comm = create_communicator_by_pipes(pipes, rank, world_size);

      if (wholegraph_communicator_support_type_location(
            wg_comm, std::get<0>(params), std::get<1>(params)) != WHOLEGRAPH_SUCCESS) {
        EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);
        EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
        WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
        if (rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      size_t total_size  = 1024UL * 1024UL * 32;
      size_t granularity = 128;

      wholegraph_handle_t handle1, handle2, handle3, handle4, handle5;
      EXPECT_EQ(
        wholegraph::create_wholegraph(
          &handle1, total_size, wg_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEGRAPH_SUCCESS);
      // handle1: 0
      EXPECT_EQ(handle1->handle_id, 0);

      EXPECT_EQ(
        wholegraph::create_wholegraph(
          &handle2, total_size, wg_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEGRAPH_SUCCESS);
      // handle1: 0, handle2: 1
      EXPECT_EQ(handle2->handle_id, 1);

      EXPECT_EQ(
        wholegraph::create_wholegraph(
          &handle3, total_size, wg_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEGRAPH_SUCCESS);
      // handle1: 0, handle2: 1, handle3: 2
      EXPECT_EQ(handle3->handle_id, 2);
      EXPECT_EQ(wg_comm->wholegraph_map.size(), 3);

      EXPECT_EQ(wholegraph::destroy_wholegraph(handle2), WHOLEGRAPH_SUCCESS);
      // handle1: 0, handle3: 2
      EXPECT_EQ(wg_comm->wholegraph_map.size(), 2);

      EXPECT_EQ(
        wholegraph::create_wholegraph(
          &handle4, total_size, wg_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEGRAPH_SUCCESS);
      // handle1: 0, handle4: 1, handle3: 2
      EXPECT_EQ(handle4->handle_id, 1);

      EXPECT_EQ(wholegraph::destroy_wholegraph(handle1), WHOLEGRAPH_SUCCESS);
      // handle4: 1, handle3: 2
      EXPECT_EQ(wg_comm->wholegraph_map.size(), 2);

      EXPECT_EQ(wholegraph::destroy_wholegraph(handle3), WHOLEGRAPH_SUCCESS);
      // handle4: 1
      EXPECT_EQ(wg_comm->wholegraph_map.size(), 1);

      EXPECT_EQ(
        wholegraph::create_wholegraph(
          &handle5, total_size, wg_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEGRAPH_SUCCESS);
      // handle5: 0, handle4: 1
      EXPECT_EQ(handle5->handle_id, 0);

      EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);

      EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);

      WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
  ClosePipes(&pipes);
}

#if 1
INSTANTIATE_TEST_SUITE_P(
  WholeGraphHandleTests,
  WholeGraphHandleMultiCreateParameterTests,
  ::testing::Values(std::make_tuple(WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_HOST),
                    std::make_tuple(WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_DEVICE),
                    std::make_tuple(WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_HOST),
                    std::make_tuple(WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_DEVICE),
                    std::make_tuple(WHOLEGRAPH_MT_DISTRIBUTED, WHOLEGRAPH_ML_HOST),
                    std::make_tuple(WHOLEGRAPH_MT_DISTRIBUTED, WHOLEGRAPH_ML_DEVICE)));
#endif
