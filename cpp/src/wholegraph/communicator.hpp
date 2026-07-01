/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>

#include "cuda_macros.hpp"

namespace wholegraph {

class nccl_comms;

}

struct wholegraph_comm_ {
  wholegraph_comm_(ncclComm_t nccl_comm, int num_ranks, int rank, cudaStream_t stream);
  ~wholegraph_comm_();

  void barrier() const;

  void abort() const;

  void allreduce(const void* sendbuff,
                 void* recvbuff,
                 size_t count,
                 wholegraph_dtype_t datatype,
                 ncclRedOp_t op,
                 cudaStream_t stream) const;

  void host_allreduce(const void* sendbuff,
                      void* recvbuff,
                      size_t count,
                      wholegraph_dtype_t datatype,
                      ncclRedOp_t op) const;

  void bcast(
    void* buff, size_t count, wholegraph_dtype_t datatype, int root, cudaStream_t stream) const;

  void bcast(const void* sendbuff,
             void* recvbuff,
             size_t count,
             wholegraph_dtype_t datatype,
             int root,
             cudaStream_t stream) const;

  void host_bcast(const void* sendbuff,
                  void* recvbuff,
                  size_t count,
                  wholegraph_dtype_t datatype,
                  int root) const;

  void host_bcast(void* buff, size_t count, wholegraph_dtype_t datatype, int root) const;

  void reduce(const void* sendbuff,
              void* recvbuff,
              size_t count,
              wholegraph_dtype_t datatype,
              ncclRedOp_t op,
              int root,
              cudaStream_t stream) const;

  void host_reduce(const void* sendbuff,
                   void* recvbuff,
                   size_t count,
                   wholegraph_dtype_t datatype,
                   ncclRedOp_t op,
                   int root) const;

  void allgather(const void* sendbuff,
                 void* recvbuff,
                 size_t sendcount,
                 wholegraph_dtype_t datatype,
                 cudaStream_t stream) const;

  void host_allgather(const void* sendbuff,
                      void* recvbuff,
                      size_t sendcount,
                      wholegraph_dtype_t datatype) const;

  void allgatherv(const void* sendbuf,
                  void* recvbuf,
                  const size_t* recvcounts,
                  const size_t* displs,
                  wholegraph_dtype_t datatype,
                  cudaStream_t stream) const;

  void host_allgatherv(const void* sendbuf,
                       void* recvbuf,
                       const size_t* recvcounts,
                       const size_t* displs,
                       wholegraph_dtype_t datatype) const;

  void gather(const void* sendbuff,
              void* recvbuff,
              size_t sendcount,
              wholegraph_dtype_t datatype,
              int root,
              cudaStream_t stream) const;

  void host_gather(const void* sendbuff,
                   void* recvbuff,
                   size_t sendcount,
                   wholegraph_dtype_t datatype,
                   int root) const;

  void gatherv(const void* sendbuff,
               void* recvbuff,
               size_t sendcount,
               const size_t* recvcounts,
               const size_t* displs,
               wholegraph_dtype_t datatype,
               int root,
               cudaStream_t stream) const;

  void reducescatter(const void* sendbuff,
                     void* recvbuff,
                     size_t recvcount,
                     wholegraph_dtype_t datatype,
                     ncclRedOp_t op,
                     cudaStream_t stream) const;

  void alltoall(const void* sendbuff,
                void* recvbuff,
                size_t sendcount,
                wholegraph_dtype_t datatype,
                cudaStream_t stream) const;

  void host_alltoall(const void* sendbuff,
                     void* recvbuff,
                     size_t sendcount,
                     wholegraph_dtype_t datatype) const;

  void alltoallv(const void* sendbuff,
                 void* recvbuff,
                 const size_t* sendcounts,
                 const size_t* senddispls,
                 const size_t* recvcounts,
                 const size_t* recvdispls,
                 wholegraph_dtype_t datatype,
                 cudaStream_t stream) const;

  wholegraph_error_code_t sync_stream(cudaStream_t stream) const;

  wholegraph_error_code_t sync_stream() const;

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_send(const void* send_buf, size_t send_size, int dest, cudaStream_t stream) const;

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_recv(void* recv_buf, size_t recv_size, int source, cudaStream_t stream) const;

  void device_sendrecv(const void* sendbuf,
                       size_t sendsize,
                       int dest,
                       void* recvbuf,
                       size_t recvsize,
                       int source,
                       cudaStream_t stream) const;

  void device_multicast_sendrecv(const void* sendbuf,
                                 std::vector<size_t> const& sendsizes,
                                 std::vector<size_t> const& sendoffsets,
                                 std::vector<int> const& dests,
                                 void* recvbuf,
                                 std::vector<size_t> const& recvsizes,
                                 std::vector<size_t> const& recvoffsets,
                                 std::vector<int> const& sources,
                                 cudaStream_t stream) const;

  bool is_intranode() const;

  bool is_intra_mnnvl() const;
  bool support_type_location(wholegraph_memory_type_t memory_type,
                             wholegraph_memory_location_t memory_location) const;

  void group_start() const;

  void group_end() const;

  wholegraph::nccl_comms* raft_nccl_comm;
  cudaStream_t comm_stream = nullptr;
  cudaEvent_t cuda_event   = nullptr;
  ncclComm_t raw_nccl_comm = nullptr;

  int world_rank = 0;
  int world_size = 1;

  int intra_node_first_rank     = -1;
  int intra_node_rank           = -1;
  int intra_node_rank_num       = 0;
  int intra_node_first_rank_pid = -1;

  clique_info_t clique_info;

  int comm_id = -1;

  int dev_id            = -1;
  int local_gpu_ids[16] = {0};
  bool support_mnnvl    = false;

  size_t alloc_granularity = 2 * 1024 * 1024UL;

  std::mutex mu;
  std::map<int, wholegraph_handle_t> wholegraph_map;
  wholegraph_distributed_backend_t distributed_backend = WHOLEGRAPH_DB_NCCL;
} __attribute__((aligned(64)));

template <typename TypeT>
inline bool wg_comm_check_all_same(wholegraph_comm_t comm, const TypeT& t)
{
  std::unique_ptr<TypeT[]> t_array(new TypeT[comm->world_size]());
  comm->host_allgather(&t, t_array.get(), sizeof(TypeT), WHOLEGRAPH_DT_INT8);
  for (int r = 0; r < comm->world_size; r++) {
    if (t_array.get()[r] != t) return false;
  }
  return true;
}

template <>
inline bool wg_comm_check_all_same(wholegraph_comm_t comm, const std::string& str)
{
  size_t str_len = str.size();
  if (!wg_comm_check_all_same(comm, str_len)) return false;
  std::string cat_str;
  cat_str.resize(str_len * comm->world_size, '\0');
  comm->host_allgather(
    str.data(), const_cast<char*>(cat_str.c_str()), str_len, WHOLEGRAPH_DT_INT8);
  for (int r = 0; r < comm->world_size; r++) {
    if (std::strncmp(str.data(), cat_str.data() + r * str_len, str_len) != 0) return false;
  }
  return true;
}

#define WG_COMM_CHECK_ALL_SAME(comm, data)                                                         \
  do {                                                                                             \
    if (!wg_comm_check_all_same(comm, data)) { WHOLEGRAPH_FATAL("COMM_CHECK_ALL_SAME failed."); } \
  } while (0)

namespace wholegraph {

wholegraph_error_code_t create_unique_id(wholegraph_unique_id_t* unique_id) noexcept;

wholegraph_error_code_t create_communicator(wholegraph_comm_t* comm,
                                             wholegraph_unique_id_t unique_id,
                                             int rank,
                                             int size) noexcept;

wholegraph_error_code_t split_communicator(wholegraph_comm_t* new_comm,
                                            wholegraph_comm_t parent_comm,
                                            int color,
                                            int key) noexcept;

wholegraph_error_code_t destroy_communicator_locked(wholegraph_comm_t comm) noexcept;

wholegraph_error_code_t destroy_communicator(wholegraph_comm_t comm) noexcept;

wholegraph_error_code_t communicator_support_type_location(
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location) noexcept;

wholegraph_error_code_t destroy_all_communicators() noexcept;

wholegraph_error_code_t communicator_get_rank(int* rank, wholegraph_comm_t comm) noexcept;

wholegraph_error_code_t communicator_get_size(int* size, wholegraph_comm_t comm) noexcept;

wholegraph_error_code_t communicator_get_local_size(int* local_size,
                                                     wholegraph_comm_t comm) noexcept;

wholegraph_error_code_t communicator_get_clique_info(clique_info_t* clique_info,
                                                      wholegraph_comm_t comm) noexcept;

void communicator_barrier(wholegraph_comm_t comm);

bool is_intranode_communicator(wholegraph_comm_t comm) noexcept;

bool is_intra_mnnvl_communicator(wholegraph_comm_t comm) noexcept;

std::string get_temporary_directory_path(wholegraph_comm_t comm);

std::string get_shm_prefix(wholegraph_comm_t comm);

wholegraph_error_code_t communicator_set_distributed_backend(
  wholegraph_comm_t comm, wholegraph_distributed_backend_t distributed_backend) noexcept;

wholegraph_distributed_backend_t communicator_get_distributed_backend(
  wholegraph_comm_t comm) noexcept;

}  // namespace wholegraph
