/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "initialize.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include "communicator.hpp"
#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"

namespace wholegraph {

static std::mutex mu;
static bool is_wg_init = false;

static const std::string RAFT_NAME  = "wholegraph";
static cudaDeviceProp* device_props = nullptr;

wholegraph_error_code_t init(unsigned int flags, LogLevel log_level) noexcept
{
  try {
    std::unique_lock<std::mutex> lock(mu);
    (void)flags;
    WHOLEGRAPH_EXPECTS(!is_wg_init, "WholeGraph has already been initialized.");
    WG_CU_CHECK(cuInit(0));
    int dev_count = 0;
    WG_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
      WHOLEGRAPH_ERROR("init failed, no CUDA device found");
      return WHOLEGRAPH_CUDA_ERROR;
    }
    device_props = new cudaDeviceProp[dev_count];
    for (int i = 0; i < dev_count; i++) {
      WG_CUDA_CHECK(cudaGetDeviceProperties(device_props + i, i));
    }
    is_wg_init = true;
    wholegraph::set_log_level(log_level);
    return WHOLEGRAPH_SUCCESS;
  } catch (raft::logic_error& logic_error) {
    WHOLEGRAPH_ERROR("init failed, logic_error=%s", logic_error.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (wholegraph::logic_error& wle) {
    WHOLEGRAPH_ERROR("init failed, logic_error=%s", wle.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (wholegraph::cuda_error& wce) {
    WHOLEGRAPH_ERROR("init failed, cuda_error=%s", wce.what());
    return WHOLEGRAPH_CUDA_ERROR;
  } catch (wholegraph::cu_error& wce) {
    WHOLEGRAPH_ERROR("init failed, cu_error=%s", wce.what());
    return WHOLEGRAPH_CUDA_ERROR;
  } catch (...) {
    WHOLEGRAPH_ERROR("init failed, Unknown error.");
    return WHOLEGRAPH_UNKNOW_ERROR;
  }
}

wholegraph_error_code_t finalize() noexcept
{
  std::unique_lock<std::mutex> lock(mu);
  is_wg_init = false;
  WHOLEGRAPH_RETURN_ON_FAIL(destroy_all_communicators());
  delete[] device_props;
  device_props = nullptr;
  return WHOLEGRAPH_SUCCESS;
}

cudaDeviceProp* get_device_prop(int dev_id) noexcept
{
  try {
    if (dev_id == -1) { WG_CUDA_CHECK(cudaGetDevice(&dev_id)); }
    WHOLEGRAPH_CHECK(dev_id >= 0);
    return device_props + dev_id;
  } catch (...) {
    WHOLEGRAPH_ERROR("get_device_prop for dev_id=%d failed.", dev_id);
    return nullptr;
  }
}

}  // namespace wholegraph
