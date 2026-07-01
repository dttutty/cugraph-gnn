/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include "parallel_utils.hpp"
#include "wholegraph/communicator.hpp"

#include "wholegraph_test_utils.hpp"

TEST(WholeGraphCommTest, SimpleCreateDestroyCommunicator)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEGRAPH_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    wholegraph_comm_t wg_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wg_comm1->comm_id, 0);
    EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);

    WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
  });
}

TEST(WholeGraphCommTest, CommunicatorFunctions)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEGRAPH_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    wholegraph_comm_t wg_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wg_comm1->comm_id, 0);
    int comm_rank = -1;
    EXPECT_EQ(wholegraph::communicator_get_rank(&comm_rank, wg_comm1), WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(comm_rank, rank);
    int comm_size = 0;
    EXPECT_EQ(wholegraph::communicator_get_size(&comm_size, wg_comm1), WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(comm_size, world_size);
    EXPECT_EQ(wholegraph::is_intranode_communicator(wg_comm1), true);
    EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);

    WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
  });
}

TEST(WholeGraphCommTest, MultipleCreateDestroyCommunicator)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEGRAPH_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    wholegraph_comm_t wg_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wg_comm1->comm_id, 0);
    wholegraph_comm_t wg_comm2 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wg_comm2->comm_id, 1);
    EXPECT_EQ(wholegraph::destroy_communicator(wg_comm1), WHOLEGRAPH_SUCCESS);
    wholegraph_comm_t wg_comm3 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wg_comm3->comm_id, 0);
    EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);
    wholegraph_comm_t wg_comm4 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wg_comm4->comm_id, 0);
    EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);

    WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
  });
}
