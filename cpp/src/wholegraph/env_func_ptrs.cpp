/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholegraph/env_func_ptrs.hpp>

#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "initialize.hpp"

namespace wholegraph {

void default_create_memory_context_func(void** memory_context, void* /*global_context*/)
{
  auto* default_memory_context = new default_memory_context_t;
  wholegraph_initialize_tensor_desc(&default_memory_context->desc);
  default_memory_context->ptr             = nullptr;
  default_memory_context->allocation_type = WHOLEGRAPH_MA_NONE;
  *memory_context                         = default_memory_context;
}

void default_destroy_memory_context_func(void* memory_context, void* /*global_context*/)
{
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  delete default_memory_context;
}

void* default_malloc_func(wholegraph_tensor_description_t* tensor_description,
                          wholegraph_memory_allocation_type_t memory_allocation_type,
                          void* memory_context,
                          void* /*global_context*/)
{
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  void* ptr                    = nullptr;
  try {
    if (memory_allocation_type == WHOLEGRAPH_MA_HOST) {
      ptr = malloc(wholegraph_get_memory_size_from_tensor(tensor_description));
      if (ptr == nullptr) { WHOLEGRAPH_FAIL_NOTHROW("malloc returned nullptr.\n"); }
    } else if (memory_allocation_type == WHOLEGRAPH_MA_PINNED) {
      WG_CUDA_CHECK(
        cudaMallocHost(&ptr, wholegraph_get_memory_size_from_tensor(tensor_description)));
    } else if (memory_allocation_type == WHOLEGRAPH_MA_DEVICE) {
      WG_CUDA_CHECK(cudaMalloc(&ptr, wholegraph_get_memory_size_from_tensor(tensor_description)));
    } else {
      WHOLEGRAPH_FAIL_NOTHROW("memory_allocation_type incorrect.\n");
    }
  } catch (wholegraph::cuda_error& wce) {
    WHOLEGRAPH_FAIL_NOTHROW("cudaMalloc failed, %s.\n", wce.what());
  }
  default_memory_context->desc            = *tensor_description;
  default_memory_context->ptr             = ptr;
  default_memory_context->allocation_type = memory_allocation_type;
  return ptr;
}

void default_free_func(void* memory_context, void* /*global_context*/)
{
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  auto memory_allocation_type  = default_memory_context->allocation_type;
  if (memory_allocation_type == WHOLEGRAPH_MA_HOST) {
    free(default_memory_context->ptr);
  } else if (memory_allocation_type == WHOLEGRAPH_MA_PINNED) {
    WG_CUDA_CHECK(cudaFreeHost(default_memory_context->ptr));
  } else if (memory_allocation_type == WHOLEGRAPH_MA_DEVICE) {
    WG_CUDA_CHECK(cudaFree(default_memory_context->ptr));
  } else {
    WHOLEGRAPH_FAIL_NOTHROW("memory_allocation_type incorrect.\n");
  }
  wholegraph_initialize_tensor_desc(&default_memory_context->desc);
  default_memory_context->ptr             = nullptr;
  default_memory_context->allocation_type = WHOLEGRAPH_MA_NONE;
}

static wholegraph_env_func_t default_env_func = {
  .temporary_fns =
    {
      .create_memory_context_fn  = default_create_memory_context_func,
      .destroy_memory_context_fn = default_destroy_memory_context_func,
      .malloc_fn                 = default_malloc_func,
      .free_fn                   = default_free_func,
      .global_context            = nullptr,
    },
  .output_fns = {
    .malloc_fn      = default_malloc_func,
    .free_fn        = default_free_func,
    .global_context = nullptr,
  }};

wholegraph_env_func_t* get_default_env_func() { return &default_env_func; }

class SizeClassMemoryPool {
 public:
  SizeClassMemoryPool();
  ~SizeClassMemoryPool();
  void* CachedMalloc(size_t size);
  void CachedFree(void* ptr, size_t size);
  void EmptyCache();
  virtual void* MallocFnImpl(size_t size) = 0;
  virtual void FreeFnImpl(void* ptr)      = 0;

 private:
  static constexpr int kBucketCount = 64;
  std::vector<std::unique_ptr<std::mutex>> mutexes_;
  std::vector<std::queue<void*>> sized_pool_;
};
static size_t GetSizeBucketIndex(size_t size)
{
  if (size == 0) return 0;
  int power           = 0;
  size_t shifted_size = size;
  while (shifted_size) {
    shifted_size >>= 1;
    power++;
  }
  if ((size & (size - 1)) == 0) {
    return power - 1;
  } else {
    return power;
  }
}
SizeClassMemoryPool::SizeClassMemoryPool()
{
  sized_pool_.resize(kBucketCount);
  mutexes_.resize(kBucketCount);
  for (int i = 0; i < kBucketCount; i++) {
    mutexes_[i] = std::make_unique<std::mutex>();
  }
}
SizeClassMemoryPool::~SizeClassMemoryPool() {}
void* SizeClassMemoryPool::CachedMalloc(size_t size)
{
  size_t bucket_index = GetSizeBucketIndex(size);
  std::unique_lock<std::mutex> mlock(*mutexes_[bucket_index]);
  if (!sized_pool_[bucket_index].empty()) {
    void* ptr = sized_pool_[bucket_index].front();
    sized_pool_[bucket_index].pop();
    return ptr;
  } else {
    return MallocFnImpl(1ULL << bucket_index);
  }
  return nullptr;
}
void SizeClassMemoryPool::CachedFree(void* ptr, size_t size)
{
  size_t bucket_index = GetSizeBucketIndex(size);
  std::unique_lock<std::mutex> mlock(*mutexes_[bucket_index]);
  sized_pool_[bucket_index].push(ptr);
}
void SizeClassMemoryPool::EmptyCache()
{
  for (int i = 0; i < kBucketCount; i++) {
    std::unique_lock<std::mutex> mlock(*mutexes_[i]);
    while (!sized_pool_[i].empty()) {
      FreeFnImpl(sized_pool_[i].front());
      sized_pool_[i].pop();
    }
  }
}
class DeviceSizeClassMemoryPool : public SizeClassMemoryPool {
 public:
  explicit DeviceSizeClassMemoryPool(int device_id);
  ~DeviceSizeClassMemoryPool();
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;

 protected:
  int device_id_ = -1;
};
DeviceSizeClassMemoryPool::DeviceSizeClassMemoryPool(int device_id) : device_id_(device_id) {}
DeviceSizeClassMemoryPool::~DeviceSizeClassMemoryPool() {}
void* DeviceSizeClassMemoryPool::MallocFnImpl(size_t size)
{
  int old_dev;
  void* ptr;
  WG_CUDA_CHECK(cudaGetDevice(&old_dev));
  WG_CUDA_CHECK(cudaSetDevice(device_id_));
  WG_CUDA_CHECK(cudaMalloc(&ptr, size));
  WG_CUDA_CHECK(cudaSetDevice(old_dev));
  return ptr;
}
void DeviceSizeClassMemoryPool::FreeFnImpl(void* ptr)
{
  int old_dev;
  WG_CUDA_CHECK(cudaGetDevice(&old_dev));
  WG_CUDA_CHECK(cudaSetDevice(device_id_));
  WG_CUDA_CHECK(cudaFree(ptr));
  WG_CUDA_CHECK(cudaSetDevice(old_dev));
}

class PinnedSizeClassMemoryPool : public SizeClassMemoryPool {
 public:
  PinnedSizeClassMemoryPool()  = default;
  ~PinnedSizeClassMemoryPool() = default;
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;
};
void* PinnedSizeClassMemoryPool::MallocFnImpl(size_t size)
{
  void* ptr;
  WG_CUDA_CHECK(cudaMallocHost(&ptr, size));
  return ptr;
}
void PinnedSizeClassMemoryPool::FreeFnImpl(void* ptr) { WG_CUDA_CHECK(cudaFreeHost(ptr)); }

class HostSizeClassMemoryPool : public SizeClassMemoryPool {
 public:
  HostSizeClassMemoryPool()  = default;
  ~HostSizeClassMemoryPool() = default;
  void* MallocFnImpl(size_t size) override;
  void FreeFnImpl(void* ptr) override;
};
void* HostSizeClassMemoryPool::MallocFnImpl(size_t size) { return malloc(size); }
void HostSizeClassMemoryPool::FreeFnImpl(void* ptr) { free(ptr); }
class CachedAllocator {
 public:
  void* MallocHost(size_t size);
  void* MallocDevice(size_t size);
  void* MallocPinned(size_t size);
  void FreeHost(void* ptr, size_t size);
  void FreeDevice(void* ptr, size_t size);
  void FreePinned(void* ptr, size_t size);
  void DropCaches();
  static CachedAllocator* GetInst();

 private:
  CachedAllocator()
  {
    device_size_class_mem_pools_.resize(kMaxSupportedDeviceCount);
    for (int i = 0; i < kMaxSupportedDeviceCount; i++) {
      device_size_class_mem_pools_[i] = std::make_unique<DeviceSizeClassMemoryPool>(i);
    }
    pinned_size_class_mem_pool_ = std::make_unique<PinnedSizeClassMemoryPool>();
    host_size_class_mem_pool_   = std::make_unique<HostSizeClassMemoryPool>();
  }
  ~CachedAllocator() {}
  CachedAllocator(const CachedAllocator& ca)                  = delete;
  const CachedAllocator& operator=(const CachedAllocator& ca) = delete;

  static CachedAllocator ca_inst_;
  std::vector<std::unique_ptr<DeviceSizeClassMemoryPool>> device_size_class_mem_pools_;
  std::unique_ptr<PinnedSizeClassMemoryPool> pinned_size_class_mem_pool_;
  std::unique_ptr<HostSizeClassMemoryPool> host_size_class_mem_pool_;
  static constexpr int kMaxSupportedDeviceCount = 16;
};

CachedAllocator CachedAllocator::ca_inst_;
CachedAllocator* CachedAllocator::GetInst() { return &ca_inst_; }

void* CachedAllocator::MallocHost(size_t size)
{
  return host_size_class_mem_pool_->CachedMalloc(size);
}
void CachedAllocator::FreeHost(void* ptr, size_t size)
{
  host_size_class_mem_pool_->CachedFree(ptr, size);
}
void* CachedAllocator::MallocDevice(size_t size)
{
  int dev_id;
  WG_CUDA_CHECK(cudaGetDevice(&dev_id));
  return device_size_class_mem_pools_[dev_id]->CachedMalloc(size);
}
void CachedAllocator::FreeDevice(void* ptr, size_t size)
{
  int dev_id;
  WG_CUDA_CHECK(cudaGetDevice(&dev_id));
  device_size_class_mem_pools_[dev_id]->CachedFree(ptr, size);
}
void* CachedAllocator::MallocPinned(size_t size)
{
  return pinned_size_class_mem_pool_->CachedMalloc(size);
}
void CachedAllocator::FreePinned(void* ptr, size_t size)
{
  pinned_size_class_mem_pool_->CachedFree(ptr, size);
}
void CachedAllocator::DropCaches()
{
  for (int i = 0; i < kMaxSupportedDeviceCount; i++) {
    device_size_class_mem_pools_[i]->EmptyCache();
  }
  pinned_size_class_mem_pool_->EmptyCache();
  host_size_class_mem_pool_->EmptyCache();
}

void* cached_malloc_func(wholegraph_tensor_description_t* tensor_description,
                         wholegraph_memory_allocation_type_t memory_allocation_type,
                         void* memory_context,
                         void* /*global_context*/)
{
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  void* ptr                    = nullptr;
  CachedAllocator* cached_inst = CachedAllocator::GetInst();
  int devid;
  WG_CUDA_CHECK(cudaGetDevice((&devid)));
  try {
    if (memory_allocation_type == WHOLEGRAPH_MA_HOST) {
      ptr = cached_inst->MallocHost(wholegraph_get_memory_size_from_tensor(tensor_description));
      if (ptr == nullptr) { WHOLEGRAPH_FAIL_NOTHROW("cached malloc host returned nullptr.\n"); }
    } else if (memory_allocation_type == WHOLEGRAPH_MA_PINNED) {
      ptr = cached_inst->MallocPinned(wholegraph_get_memory_size_from_tensor(tensor_description));
      if (ptr == nullptr) { WHOLEGRAPH_FAIL_NOTHROW("cached malloc pinned returned nullptr.\n"); }
    } else if (memory_allocation_type == WHOLEGRAPH_MA_DEVICE) {
      ptr = cached_inst->MallocDevice(wholegraph_get_memory_size_from_tensor(tensor_description));
      if (ptr == nullptr) { WHOLEGRAPH_FAIL_NOTHROW("cached malloc device returned nullptr.\n"); }
    } else {
      WHOLEGRAPH_FAIL_NOTHROW("memory_allocation_type incorrect.\n");
    }
  } catch (wholegraph::cuda_error& wce) {
    WHOLEGRAPH_FAIL_NOTHROW("cudaMalloc failed, %s.\n", wce.what());
  }
  default_memory_context->desc            = *tensor_description;
  default_memory_context->ptr             = ptr;
  default_memory_context->allocation_type = memory_allocation_type;
  return ptr;
}

void cached_free_func(void* memory_context, void* /*global_context*/)
{
  CachedAllocator* cached_inst = CachedAllocator::GetInst();
  auto* default_memory_context = static_cast<default_memory_context_t*>(memory_context);
  auto memory_allocation_type  = default_memory_context->allocation_type;
  if (memory_allocation_type == WHOLEGRAPH_MA_HOST) {
    cached_inst->FreeHost(default_memory_context->ptr,
                          wholegraph_get_memory_size_from_tensor(&default_memory_context->desc));
  } else if (memory_allocation_type == WHOLEGRAPH_MA_PINNED) {
    cached_inst->FreePinned(default_memory_context->ptr,
                            wholegraph_get_memory_size_from_tensor(&default_memory_context->desc));
  } else if (memory_allocation_type == WHOLEGRAPH_MA_DEVICE) {
    cached_inst->FreeDevice(default_memory_context->ptr,
                            wholegraph_get_memory_size_from_tensor(&default_memory_context->desc));
  } else {
    WHOLEGRAPH_FAIL_NOTHROW("memory_allocation_type incorrect.\n");
  }
  wholegraph_initialize_tensor_desc(&default_memory_context->desc);
  default_memory_context->ptr             = nullptr;
  default_memory_context->allocation_type = WHOLEGRAPH_MA_NONE;
}

static wholegraph_env_func_t cached_env_func = {
  .temporary_fns =
    {
      .create_memory_context_fn  = default_create_memory_context_func,
      .destroy_memory_context_fn = default_destroy_memory_context_func,
      .malloc_fn                 = cached_malloc_func,
      .free_fn                   = cached_free_func,
      .global_context            = nullptr,
    },
  .output_fns = {
    .malloc_fn      = cached_malloc_func,
    .free_fn        = cached_free_func,
    .global_context = nullptr,
  }};

wholegraph_env_func_t* get_cached_env_func() { return &cached_env_func; }

void drop_cached_env_func_cache() { CachedAllocator::GetInst()->DropCaches(); }

}  // namespace wholegraph

#ifdef __cplusplus
extern "C" {
#endif

cudaDeviceProp* get_device_prop(int dev_id) { return wholegraph::get_device_prop(dev_id); }

#ifdef __cplusplus
}
#endif
