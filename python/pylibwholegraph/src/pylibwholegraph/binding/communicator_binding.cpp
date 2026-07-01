/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "communicator_binding.hpp"

PyWholeGraphUniqueIDNB create_unique_id()
{
  PyWholeGraphUniqueIDNB unique_id;
  check_wholegraph_error_code(wholegraph_create_unique_id(unique_id.c_ptr()));
  return unique_id;
}

PyWholeGraphCommNB create_communicator(PyWholeGraphUniqueIDNB const& unique_id,
                                       int world_rank,
                                       int world_size)
{
  wholegraph_comm_t comm = nullptr;
  check_wholegraph_error_code(
    wholegraph_create_communicator(&comm, unique_id.value(), world_rank, world_size));
  return PyWholeGraphCommNB::from_c_handle(comm);
}

void destroy_communicator(PyWholeGraphCommNB const& comm)
{
  check_wholegraph_error_code(wholegraph_destroy_communicator(comm.c_handle()));
}

PyWholeGraphCommNB split_communicator(PyWholeGraphCommNB const& comm, int color, int key)
{
  wholegraph_comm_t new_comm = nullptr;
  check_wholegraph_error_code(
    wholegraph_split_communicator(&new_comm, comm.c_handle(), color, key));
  return PyWholeGraphCommNB::from_c_handle(new_comm);
}

void communicator_set_distributed_backend(
  PyWholeGraphCommNB& comm, wholegraph_distributed_backend_t distributed_backend)
{
  comm.set_distributed_backend(distributed_backend);
}

size_t equal_partition_plan(size_t entry_count, int world_size)
{
  size_t per_rank_count = 0;
  check_wholegraph_error_code(
    wholegraph_equal_entry_partition_plan(&per_rank_count, entry_count, world_size));
  return per_rank_count;
}
