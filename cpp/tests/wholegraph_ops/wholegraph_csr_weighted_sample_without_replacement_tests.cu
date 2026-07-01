/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <random>

#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph_op.h>
#include <wholegraph/wholegraph.h>

#include "parallel_utils.hpp"
#include "wholegraph/communicator.hpp"
#include "wholegraph/env_func_ptrs.hpp"
#include "wholegraph/initialize.hpp"

#include "../wholegraph/wholegraph_test_utils.hpp"
#include "graph_sampling_test_utils.hpp"

typedef struct WholeGraphCSRWeightedSampleWithoutReplacementTestParam {
  wholegraph_array_description_t get_csr_row_ptr_desc() const
  {
    return wholegraph_create_array_desc(graph_node_count + 1, 0, csr_row_ptr_dtype);
  }

  wholegraph_array_description_t get_csr_col_ptr_desc() const
  {
    return wholegraph_create_array_desc(graph_edge_count, 0, csr_col_ptr_dtype);
  }

  wholegraph_array_description_t get_csr_weight_ptr_desc() const
  {
    return wholegraph_create_array_desc(graph_edge_count, 0, csr_weight_ptr_dtype);
  }

  wholegraph_array_description_t get_center_node_desc() const
  {
    return wholegraph_create_array_desc(center_node_count, 0, center_node_dtype);
  }

  wholegraph_array_description_t get_output_sample_offset_desc() const
  {
    return wholegraph_create_array_desc(center_node_count + 1, 0, output_sample_offset_dtype);
  }

  int64_t get_graph_node_count() const { return graph_node_count; }
  int64_t get_graph_edge_count() const { return graph_edge_count; }
  int64_t get_max_sample_count() const { return max_sample_count; }

  WholeGraphCSRWeightedSampleWithoutReplacementTestParam& set_memory_type(
    wholegraph_memory_type_t new_memory_type)
  {
    memory_type = new_memory_type;
    return *this;
  };
  WholeGraphCSRWeightedSampleWithoutReplacementTestParam& set_memory_location(
    wholegraph_memory_location_t new_memory_location)
  {
    memory_location = new_memory_location;
    return *this;
  };

  WholeGraphCSRWeightedSampleWithoutReplacementTestParam& set_max_sample_count(int new_sample_count)
  {
    max_sample_count = new_sample_count;
    return *this;
  }

  WholeGraphCSRWeightedSampleWithoutReplacementTestParam& set_center_node_count(
    int new_center_node_count)
  {
    center_node_count = new_center_node_count;
    return *this;
  }
  WholeGraphCSRWeightedSampleWithoutReplacementTestParam& set_graph_node_count(
    int new_graph_node_count)
  {
    graph_node_count = new_graph_node_count;
    return *this;
  }
  WholeGraphCSRWeightedSampleWithoutReplacementTestParam& set_graph_edge_couont(
    int new_graph_edge_count)
  {
    graph_edge_count = new_graph_edge_count;
    return *this;
  }

  WholeGraphCSRWeightedSampleWithoutReplacementTestParam& set_center_node_type(
    wholegraph_dtype_t new_center_node_dtype)
  {
    center_node_dtype = new_center_node_dtype;
    return *this;
  }

  wholegraph_memory_type_t memory_type                 = WHOLEGRAPH_MT_CONTINUOUS;
  wholegraph_memory_location_t memory_location         = WHOLEGRAPH_ML_DEVICE;
  int64_t max_sample_count                              = 10;
  int64_t center_node_count                             = 512;
  int64_t graph_node_count                              = 9703LL;
  int64_t graph_edge_count                              = 104323L;
  wholegraph_dtype_t csr_row_ptr_dtype                 = WHOLEGRAPH_DT_INT64;
  wholegraph_dtype_t csr_col_ptr_dtype                 = WHOLEGRAPH_DT_INT;
  wholegraph_dtype_t csr_weight_ptr_dtype              = WHOLEGRAPH_DT_FLOAT;
  wholegraph_dtype_t center_node_dtype                 = WHOLEGRAPH_DT_INT;
  wholegraph_dtype_t output_sample_offset_dtype        = WHOLEGRAPH_DT_INT;
  wholegraph_dtype_t output_dest_node_dtype            = center_node_dtype;
  wholegraph_dtype_t output_center_node_local_id_dtype = WHOLEGRAPH_DT_INT;
  wholegraph_dtype_t output_globla_edge_id_dtype       = WHOLEGRAPH_DT_INT64;

} WholeGraphCSRWeightedSampleWithoutReplacementTestParam;

class WholeGraphCSRWeightedSampleWithoutReplacementParameterTests
  : public ::testing::TestWithParam<WholeGraphCSRWeightedSampleWithoutReplacementTestParam> {};

TEST_P(WholeGraphCSRWeightedSampleWithoutReplacementParameterTests, WeightedSampleTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  auto graph_node_count          = params.get_graph_node_count();
  auto graph_edge_count          = params.get_graph_edge_count();
  auto graph_csr_row_ptr_desc    = params.get_csr_row_ptr_desc();
  auto graph_csr_col_ptr_desc    = params.get_csr_col_ptr_desc();
  auto graph_csr_weight_ptr_desc = params.get_csr_weight_ptr_desc();

  void* host_csr_row_ptr =
    (void*)malloc(wholegraph_get_memory_size_from_array(&graph_csr_row_ptr_desc));
  void* host_csr_col_ptr =
    (void*)malloc(wholegraph_get_memory_size_from_array(&graph_csr_col_ptr_desc));
  void* host_csr_weight_ptr =
    (void*)malloc(wholegraph_get_memory_size_from_array(&graph_csr_weight_ptr_desc));
  wholegraph_ops::testing::gen_csr_graph(graph_node_count,
                                         graph_edge_count,
                                         host_csr_row_ptr,
                                         graph_csr_row_ptr_desc,
                                         host_csr_col_ptr,
                                         graph_csr_col_ptr_desc,
                                         host_csr_weight_ptr,
                                         graph_csr_weight_ptr_desc);

  MultiProcessRun(
    dev_count,
    [&params, &pipes, host_csr_row_ptr, host_csr_col_ptr, host_csr_weight_ptr](int world_rank,
                                                                               int world_size) {
      thread_local std::random_device rd;
      thread_local std::mt19937 gen(rd());
      thread_local std::uniform_int_distribution<unsigned long long> distrib;
      unsigned long long random_seed = distrib(gen);

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

      auto csr_row_ptr_desc          = params.get_csr_row_ptr_desc();
      auto csr_col_ptr_desc          = params.get_csr_col_ptr_desc();
      auto csr_weight_ptr_desc       = params.get_csr_weight_ptr_desc();
      auto center_node_desc          = params.get_center_node_desc();
      auto output_sample_offset_desc = params.get_output_sample_offset_desc();
      auto max_sample_count          = params.get_max_sample_count();
      int64_t graph_node_count       = params.get_graph_node_count();
      int64_t graph_edge_count       = params.get_graph_edge_count();

      size_t center_node_size = wholegraph_get_memory_size_from_array(&center_node_desc);
      size_t output_sample_offset_size =
        wholegraph_get_memory_size_from_array(&output_sample_offset_desc);

      cudaStream_t stream;
      EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

      void *host_ref_output_sample_offset, *host_ref_output_dest_nodes,
        *host_ref_output_center_nodes_local_id, *host_ref_output_global_edge_id;

      void *host_center_nodes, *host_output_sample_offset, *host_output_dest_nodes,
        *host_output_center_nodes_local_id, *host_output_global_edge_id;
      void *dev_center_nodes, *dev_output_sample_offset;

      wholegraph_handle_t csr_row_ptr_memory_handle;
      wholegraph_handle_t csr_col_ptr_memory_handle;
      wholegraph_handle_t csr_weight_ptr_memory_handle;

      EXPECT_EQ(wholegraph_malloc(&csr_row_ptr_memory_handle,
                                   wholegraph_get_memory_size_from_array(&csr_row_ptr_desc),
                                   wg_comm,
                                   params.memory_type,
                                   params.memory_location,
                                   wholegraph_dtype_get_element_size(csr_row_ptr_desc.dtype)),
                WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_malloc(&csr_col_ptr_memory_handle,
                                   wholegraph_get_memory_size_from_array(&csr_col_ptr_desc),
                                   wg_comm,
                                   params.memory_type,
                                   params.memory_location,
                                   wholegraph_dtype_get_element_size(csr_col_ptr_desc.dtype)),
                WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_malloc(&csr_weight_ptr_memory_handle,
                                   wholegraph_get_memory_size_from_array(&csr_weight_ptr_desc),
                                   wg_comm,
                                   params.memory_type,
                                   params.memory_location,
                                   wholegraph_dtype_get_element_size(csr_weight_ptr_desc.dtype)),
                WHOLEGRAPH_SUCCESS);

      wholegraph_ops::testing::copy_host_array_to_wholegraph(
        host_csr_row_ptr, csr_row_ptr_memory_handle, csr_row_ptr_desc, stream);
      wholegraph_ops::testing::copy_host_array_to_wholegraph(
        host_csr_col_ptr, csr_col_ptr_memory_handle, csr_col_ptr_desc, stream);
      wholegraph_ops::testing::copy_host_array_to_wholegraph(
        host_csr_weight_ptr, csr_weight_ptr_memory_handle, csr_weight_ptr_desc, stream);

      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholegraph_communicator_barrier(wg_comm);

      EXPECT_EQ(cudaSetDevice(world_rank), cudaSuccess);
      EXPECT_EQ(cudaMallocHost(&host_center_nodes, center_node_size), cudaSuccess);
      EXPECT_EQ(cudaMallocHost(&host_output_sample_offset, output_sample_offset_size), cudaSuccess);

      EXPECT_EQ(cudaMalloc(&dev_center_nodes, center_node_size), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&dev_output_sample_offset, output_sample_offset_size), cudaSuccess);

      wholegraph_ops::testing::host_random_init_array(
        host_center_nodes, center_node_desc, 0, graph_node_count - 1);
      EXPECT_EQ(cudaMemcpyAsync(dev_center_nodes,
                                host_center_nodes,
                                wholegraph_get_memory_size_from_array(&center_node_desc),
                                cudaMemcpyHostToDevice,
                                stream),
                cudaSuccess);

      wholegraph_tensor_t wg_csr_row_ptr_tensor, wg_csr_col_ptr_tensor, wg_csr_weight_ptr_tensor;
      wholegraph_tensor_description_t wg_csr_row_ptr_tensor_desc, wg_csr_col_ptr_tensor_desc,
        wg_csr_weight_ptr_tensor_desc;
      wholegraph_copy_array_desc_to_tensor(&wg_csr_row_ptr_tensor_desc, &csr_row_ptr_desc);
      wholegraph_copy_array_desc_to_tensor(&wg_csr_col_ptr_tensor_desc, &csr_col_ptr_desc);
      wholegraph_copy_array_desc_to_tensor(&wg_csr_weight_ptr_tensor_desc, &csr_weight_ptr_desc);
      EXPECT_EQ(wholegraph_make_tensor_from_handle(
                  &wg_csr_row_ptr_tensor, csr_row_ptr_memory_handle, &wg_csr_row_ptr_tensor_desc),
                WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_make_tensor_from_handle(
                  &wg_csr_col_ptr_tensor, csr_col_ptr_memory_handle, &wg_csr_col_ptr_tensor_desc),
                WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(
        wholegraph_make_tensor_from_handle(
          &wg_csr_weight_ptr_tensor, csr_weight_ptr_memory_handle, &wg_csr_weight_ptr_tensor_desc),
        WHOLEGRAPH_SUCCESS);

      wholegraph_tensor_t center_nodes_tensor, output_sample_offset_tensor;
      wholegraph_tensor_description_t center_nodes_tensor_desc, output_sample_offset_tensor_desc;
      wholegraph_copy_array_desc_to_tensor(&center_nodes_tensor_desc, &center_node_desc);
      wholegraph_copy_array_desc_to_tensor(&output_sample_offset_tensor_desc,
                                            &output_sample_offset_desc);
      EXPECT_EQ(wholegraph_make_tensor_from_pointer(
                  &center_nodes_tensor, dev_center_nodes, &center_nodes_tensor_desc),
                WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_make_tensor_from_pointer(&output_sample_offset_tensor,
                                                     dev_output_sample_offset,
                                                     &output_sample_offset_tensor_desc),
                WHOLEGRAPH_SUCCESS);

      wholegraph_env_func_t* default_env_func = wholegraph::get_default_env_func();
      wholegraph::default_memory_context_t output_dest_mem_ctx, output_center_localid_mem_ctx,
        output_edge_gid_mem_ctx;

      EXPECT_EQ(wholegraph_csr_weighted_sample_without_replacement(wg_csr_row_ptr_tensor,
                                                                   wg_csr_col_ptr_tensor,
                                                                   wg_csr_weight_ptr_tensor,
                                                                   center_nodes_tensor,
                                                                   max_sample_count,
                                                                   output_sample_offset_tensor,
                                                                   &output_dest_mem_ctx,
                                                                   &output_center_localid_mem_ctx,
                                                                   &output_edge_gid_mem_ctx,
                                                                   random_seed,
                                                                   default_env_func,
                                                                   stream),
                WHOLEGRAPH_SUCCESS);

      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholegraph_communicator_barrier(wg_comm);

      EXPECT_EQ(output_dest_mem_ctx.desc.dim, 1);
      EXPECT_EQ(output_center_localid_mem_ctx.desc.dim, 1);
      EXPECT_EQ(output_edge_gid_mem_ctx.desc.dim, 1);

      EXPECT_EQ(output_dest_mem_ctx.desc.dtype, csr_col_ptr_desc.dtype);
      EXPECT_EQ(output_center_localid_mem_ctx.desc.dtype, WHOLEGRAPH_DT_INT);
      EXPECT_EQ(output_edge_gid_mem_ctx.desc.dtype, WHOLEGRAPH_DT_INT64);

      EXPECT_EQ(output_dest_mem_ctx.desc.sizes[0], output_center_localid_mem_ctx.desc.sizes[0]);
      EXPECT_EQ(output_dest_mem_ctx.desc.sizes[0], output_edge_gid_mem_ctx.desc.sizes[0]);

      int64_t total_sample_count = output_dest_mem_ctx.desc.sizes[0];

      host_output_dest_nodes =
        malloc(total_sample_count * wholegraph_dtype_get_element_size(csr_col_ptr_desc.dtype));
      host_output_center_nodes_local_id = malloc(total_sample_count * sizeof(int));
      host_output_global_edge_id        = malloc(total_sample_count * sizeof(int64_t));

      EXPECT_EQ(cudaMemcpyAsync(host_output_sample_offset,
                                dev_output_sample_offset,
                                output_sample_offset_size,
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpyAsync(
                  host_output_dest_nodes,
                  output_dest_mem_ctx.ptr,
                  total_sample_count * wholegraph_dtype_get_element_size(csr_col_ptr_desc.dtype),
                  cudaMemcpyDeviceToHost,
                  stream),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpyAsync(host_output_center_nodes_local_id,
                                output_center_localid_mem_ctx.ptr,
                                total_sample_count * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpyAsync(host_output_global_edge_id,
                                output_edge_gid_mem_ctx.ptr,
                                total_sample_count * sizeof(int64_t),
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);

      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholegraph_communicator_barrier(wg_comm);

      wholegraph_ops::testing::segment_sort_output(
        host_output_sample_offset,
        output_sample_offset_desc,
        host_output_dest_nodes,
        wholegraph_create_array_desc(total_sample_count, 0, csr_col_ptr_desc.dtype),
        host_output_global_edge_id,
        wholegraph_create_array_desc(total_sample_count, 0, WHOLEGRAPH_DT_INT64));

      int host_total_sample_count;

      wholegraph_ops::testing::wholegraph_csr_weighted_sample_without_replacement_cpu(
        host_csr_row_ptr,
        csr_row_ptr_desc,
        host_csr_col_ptr,
        csr_col_ptr_desc,
        host_csr_weight_ptr,
        csr_weight_ptr_desc,
        host_center_nodes,
        center_node_desc,
        max_sample_count,
        &host_ref_output_sample_offset,
        output_sample_offset_desc,
        &host_ref_output_dest_nodes,
        &host_ref_output_center_nodes_local_id,
        &host_ref_output_global_edge_id,
        &host_total_sample_count,
        random_seed);

      EXPECT_EQ(total_sample_count, host_total_sample_count);
      wholegraph_ops::testing::segment_sort_output(
        host_ref_output_sample_offset,
        output_sample_offset_desc,
        host_ref_output_dest_nodes,
        wholegraph_create_array_desc(host_total_sample_count, 0, csr_col_ptr_desc.dtype),
        host_ref_output_global_edge_id,
        wholegraph_create_array_desc(host_total_sample_count, 0, WHOLEGRAPH_DT_INT64));
      wholegraph_ops::testing::host_check_two_array_same(host_output_sample_offset,
                                                         output_sample_offset_desc,
                                                         host_ref_output_sample_offset,
                                                         output_sample_offset_desc);
      wholegraph_ops::testing::host_check_two_array_same(
        host_output_dest_nodes,
        wholegraph_create_array_desc(host_total_sample_count, 0, csr_col_ptr_desc.dtype),
        host_ref_output_dest_nodes,
        wholegraph_create_array_desc(host_total_sample_count, 0, csr_col_ptr_desc.dtype));

      wholegraph_ops::testing::host_check_two_array_same(
        host_output_center_nodes_local_id,
        wholegraph_create_array_desc(host_total_sample_count, 0, WHOLEGRAPH_DT_INT),
        host_ref_output_center_nodes_local_id,
        wholegraph_create_array_desc(host_total_sample_count, 0, WHOLEGRAPH_DT_INT));

      wholegraph_ops::testing::host_check_two_array_same(
        host_output_global_edge_id,
        wholegraph_create_array_desc(host_total_sample_count, 0, WHOLEGRAPH_DT_INT64),
        host_ref_output_global_edge_id,
        wholegraph_create_array_desc(host_total_sample_count, 0, WHOLEGRAPH_DT_INT64));

      (default_env_func->output_fns).free_fn(&output_dest_mem_ctx, nullptr);
      (default_env_func->output_fns).free_fn(&output_center_localid_mem_ctx, nullptr);
      (default_env_func->output_fns).free_fn(&output_edge_gid_mem_ctx, nullptr);

      if (host_ref_output_sample_offset != nullptr) free(host_ref_output_sample_offset);
      if (host_ref_output_dest_nodes != nullptr) free(host_ref_output_dest_nodes);
      if (host_ref_output_center_nodes_local_id != nullptr)
        free(host_ref_output_center_nodes_local_id);
      if (host_ref_output_global_edge_id != nullptr) free(host_ref_output_global_edge_id);

      EXPECT_EQ(cudaFreeHost(host_center_nodes), cudaSuccess);
      EXPECT_EQ(cudaFreeHost(host_output_sample_offset), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_center_nodes), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_output_sample_offset), cudaSuccess);

      EXPECT_EQ(wholegraph_free(csr_row_ptr_memory_handle), WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_free(csr_col_ptr_memory_handle), WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_free(csr_weight_ptr_memory_handle), WHOLEGRAPH_SUCCESS);

      EXPECT_EQ(wholegraph::destroy_all_communicators(), WHOLEGRAPH_SUCCESS);
      EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
      WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
    },
    true);

  if (host_csr_row_ptr != nullptr) free(host_csr_row_ptr);
  if (host_csr_col_ptr != nullptr) free(host_csr_col_ptr);
  if (host_csr_weight_ptr != nullptr) free(host_csr_weight_ptr);
}

INSTANTIATE_TEST_SUITE_P(WholeGraphCSRWeightedSampleWithoutReplacementOpTests,
                         WholeGraphCSRWeightedSampleWithoutReplacementParameterTests,
                         ::testing::Values(WholeGraphCSRWeightedSampleWithoutReplacementTestParam()
                                             .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS),
                                           WholeGraphCSRWeightedSampleWithoutReplacementTestParam()
                                             .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS),
                                           WholeGraphCSRWeightedSampleWithoutReplacementTestParam()
                                             .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
                                             .set_memory_location(WHOLEGRAPH_ML_HOST),
                                           WholeGraphCSRWeightedSampleWithoutReplacementTestParam()
                                             .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
                                             .set_memory_location(WHOLEGRAPH_ML_HOST),
                                           WholeGraphCSRWeightedSampleWithoutReplacementTestParam()
                                             .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
                                             .set_max_sample_count(10)
                                             .set_center_node_count(35)
                                             .set_graph_node_count(23289)
                                             .set_graph_edge_couont(689403),
                                           WholeGraphCSRWeightedSampleWithoutReplacementTestParam()
                                             .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
                                             .set_max_sample_count(300)
                                             .set_center_node_count(256)
                                             .set_graph_node_count(23200)
                                             .set_graph_edge_couont(68940300),
                                           WholeGraphCSRWeightedSampleWithoutReplacementTestParam()
                                             .set_memory_type(WHOLEGRAPH_MT_CONTINUOUS)
                                             .set_center_node_type(WHOLEGRAPH_DT_INT64)));
