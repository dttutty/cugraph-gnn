/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <wholegraph/tensor_description.h>
#include <wholegraph/wholegraph.h>
#include <wholegraph/wholegraph_tensor.h>

#include "parallel_utils.hpp"

struct MatrixTestParam {
  MatrixTestParam& set_row(int64_t r)
  {
    row = r;
    return *this;
  }
  MatrixTestParam& set_col(int64_t c)
  {
    col = c;
    return *this;
  }
  MatrixTestParam& set_dtype(wholegraph_dtype_t dt)
  {
    dtype = dt;
    return *this;
  }
  int64_t row               = 256LL * 128LL;
  int64_t col               = 256LL;
  wholegraph_dtype_t dtype = WHOLEGRAPH_DT_FLOAT;
};

class WholeGraphMatrixTest : public ::testing::TestWithParam<MatrixTestParam> {};

TEST(WholeGraphMatrixTest, SubTensorTest)
{
  MatrixTestParam params;
  params.set_row(256LL * 128LL).set_col(256LL).set_dtype(WHOLEGRAPH_DT_INT);
  MultiProcessRun(1, [&params](int world_rank, int world_size) {
    EXPECT_EQ(wholegraph_init(0), WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);

    wholegraph_unique_id_t unique_id;
    wholegraph_comm_t wg_comm;
    EXPECT_EQ(wholegraph_create_unique_id(&unique_id), WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(wholegraph_create_communicator(&wg_comm, unique_id, world_rank, world_size),
              WHOLEGRAPH_SUCCESS);

    int64_t sizes[2] = {params.row, params.col};
    wholegraph_matrix_description_t mat_desc =
      wholegraph_create_matrix_desc(sizes, params.col, 0, params.dtype);
    wholegraph_tensor_description_t tensor_desc;
    wholegraph_copy_matrix_desc_to_tensor(&tensor_desc, &mat_desc);

    wholegraph_tensor_t wholegraph_tensor;
    EXPECT_EQ(
      wholegraph_create_tensor(
        &wholegraph_tensor, &tensor_desc, wg_comm, WHOLEGRAPH_MT_CONTINUOUS, WHOLEGRAPH_ML_HOST),
      WHOLEGRAPH_SUCCESS);
    wholegraph_handle_t wg_handle = wholegraph_tensor_get_memory_handle(wholegraph_tensor);
    int* ptr                       = nullptr;
    EXPECT_EQ(wholegraph_get_global_pointer((void**)&ptr, wg_handle), WHOLEGRAPH_SUCCESS);
    for (int64_t i = 0; i < params.row * params.col; i++) {
      ptr[i] = i;
    }
    wholegraph_tensor_t wholegraph_sub_tensor_0, wholegraph_sub_tensor_1;
    wholegraph_tensor_description_t sub_desc_0, sub_desc_1;

    int64_t starts_0[2] = {1, 10};
    int64_t ends_0[2]   = {-1, 100};
    int64_t starts_1[2] = {2, -1};
    int64_t ends_1[2]   = {10000, 80};

    EXPECT_EQ(wholegraph_tensor_get_subtensor(
                wholegraph_tensor, starts_0, ends_0, &wholegraph_sub_tensor_0),
              WHOLEGRAPH_SUCCESS);
    sub_desc_0 = *wholegraph_tensor_get_tensor_description(wholegraph_sub_tensor_0);
    EXPECT_EQ(sub_desc_0.dim, 2);
    EXPECT_EQ(sub_desc_0.dtype, WHOLEGRAPH_DT_INT);
    EXPECT_EQ(sub_desc_0.storage_offset, params.col * 1 + 10);
    EXPECT_EQ(sub_desc_0.sizes[0], params.row - 1);
    EXPECT_EQ(sub_desc_0.sizes[1], 90);
    EXPECT_EQ(sub_desc_0.strides[0], 256);
    EXPECT_EQ(sub_desc_0.strides[1], 1);
    EXPECT_EQ(wholegraph_tensor_get_subtensor(
                wholegraph_sub_tensor_0, starts_1, ends_1, &wholegraph_sub_tensor_1),
              WHOLEGRAPH_SUCCESS);
    sub_desc_1 = *wholegraph_tensor_get_tensor_description(wholegraph_sub_tensor_1);
    EXPECT_EQ(sub_desc_1.dim, 2);
    EXPECT_EQ(sub_desc_1.dtype, WHOLEGRAPH_DT_INT);
    EXPECT_EQ(sub_desc_1.storage_offset, params.col * 3 + 10);
    EXPECT_EQ(sub_desc_1.sizes[0], 10000 - 2);
    EXPECT_EQ(sub_desc_1.sizes[1], 80);
    EXPECT_EQ(sub_desc_1.strides[0], 256);
    EXPECT_EQ(sub_desc_1.strides[1], 1);

    EXPECT_EQ(wholegraph_destroy_tensor(wholegraph_sub_tensor_0), WHOLEGRAPH_SUCCESS);
    EXPECT_EQ(wholegraph_destroy_tensor(wholegraph_sub_tensor_1), WHOLEGRAPH_SUCCESS);

    for (int64_t i = 0; i < params.row * params.col; i++) {
      EXPECT_EQ(ptr[i], i);
    }

    EXPECT_EQ(wholegraph_destroy_tensor(wholegraph_tensor), WHOLEGRAPH_SUCCESS);

    EXPECT_EQ(wholegraph_finalize(), WHOLEGRAPH_SUCCESS);
    WHOLEGRAPH_CHECK(::testing::Test::HasFailure() == false);
  });
}
