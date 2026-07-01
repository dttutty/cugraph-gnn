/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "file_io.h"

#include <atomic>
#include <cstdint>

#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "communicator.hpp"
#include "error.hpp"
#include "integer_utils.hpp"
#include "logger.hpp"

namespace wholegraph {

static bool IsFileExist(const char* filename, int mode) { return access(filename, mode) == 0; }

static size_t StatFileSize(const char* filename)
{
  auto filesize = static_cast<size_t>(-1);
  struct stat statbuf{};
  if (stat(filename, &statbuf) < 0) { return filesize; }
  filesize = statbuf.st_size;
  return filesize;
}

static size_t StatFileBlockSize(const char* filename)
{
  auto blocksize = static_cast<size_t>(-1);
  struct stat statbuf{};
  if (stat(filename, &statbuf) < 0) { return blocksize; }
  blocksize = statbuf.st_blksize;
  return blocksize;
}

static size_t get_handle_partial_size(size_t handle_size,
                                      size_t memory_offset,
                                      size_t memory_entry_stride,
                                      size_t entry_size)
{
  handle_size -= memory_offset;
  size_t tail = handle_size % memory_entry_stride;
  if (tail != 0 && tail < entry_size) {
    WHOLEGRAPH_FAIL_NOTHROW(
      "handle_size=%ld, memory_offset=%ld, memory_entry_stride=%ld, entry_size=%ld, tail=%ld is "
      "not 0"
      " or >= entry_size.",
      handle_size,
      memory_offset,
      memory_entry_stride,
      entry_size,
      tail);
  }
  size_t partial_size = 0;
  if (tail != 0) partial_size = entry_size;
  partial_size += (handle_size / memory_entry_stride) * entry_size;
  return partial_size;
}

/*!
 * Read from file list to local memory of WholeGraph. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeGraph will read the files in parallel
 * and load each part into local memory of each rank. WholeGraph will use round-robin sharding
 * strategy.
 * @param local_ptr : Pointer to local memory of WholeGraph
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeGraph.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeGraph
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wg_rank : WholeGraph rank.
 * @param wg_world_size : WholeGraph world size.
 * @param round_robin_size : continuous embedding size of a rank using round robin shard strategy.
 */
static void read_file_list_to_local_memory_roundrobin(char* local_ptr,
                                                      size_t local_size,
                                                      size_t local_offset,
                                                      size_t entry_size,
                                                      size_t memory_entry_stride,
                                                      size_t memory_offset,
                                                      int file_count,
                                                      const char** file_names,
                                                      const std::vector<size_t>& file_sizes,
                                                      size_t suggested_buffer_size,
                                                      int wg_rank,
                                                      int wg_world_size,
                                                      int round_robin_size)
{
  size_t buffer_size;
  size_t buffer_entry_count = 1;
  if (suggested_buffer_size < entry_size) {
    buffer_size = entry_size;
  } else {
    buffer_entry_count = suggested_buffer_size / entry_size;
    buffer_size        = buffer_entry_count * entry_size;
  }
  std::vector<char> file_read_buffer(buffer_size);

  if (memory_offset >= memory_entry_stride)
    WHOLEGRAPH_ERROR("memory offset %lu should be less than memory entry stride %lu.",
                      memory_offset,
                      memory_entry_stride);
  size_t total_file_sizes = 0;
  for (int i = 0; i < file_count; i++)
    total_file_sizes += file_sizes[i];
  size_t total_file_entry_count = total_file_sizes / entry_size;
  if (round_robin_size <= 0 || round_robin_size > total_file_entry_count / wg_world_size)
    WHOLEGRAPH_ERROR("illegal round_robin_size.");

  size_t local_entry_memory_start_index = wg_rank * round_robin_size;
  size_t local_entry_file_start_index =
    local_entry_memory_start_index - memory_offset / memory_entry_stride;

  int extra_entry       = total_file_entry_count % (wg_world_size * round_robin_size);
  int local_extra_entry = (extra_entry > (wg_rank + 1) * round_robin_size)
                            ? round_robin_size
                            : extra_entry - wg_rank * round_robin_size;
  local_extra_entry     = local_extra_entry > 0 ? local_extra_entry : 0;
  size_t local_entry_count =
    total_file_entry_count / (wg_world_size * round_robin_size) * round_robin_size;
  local_entry_count += local_extra_entry;

  char* local_write_ptr = local_ptr + memory_offset % memory_entry_stride;
  if (wg_rank == 0) {
    local_entry_count -= memory_offset / memory_entry_stride;
    local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
  }

  size_t total_read_entry            = 0;
  size_t next_entry_gap              = local_entry_file_start_index;
  size_t next_continuous_entry_count = round_robin_size > local_entry_count - total_read_entry
                                         ? local_entry_count - total_read_entry
                                         : round_robin_size;
  size_t read_file_begin_entry_off   = 0;
  for (int i = 0; i < file_count; i++) {
    size_t file_entry_count = file_sizes[i] / entry_size;
    if (file_entry_count <= next_entry_gap) {
      next_entry_gap -= file_entry_count;
      continue;
    }
    size_t read_size_from_cur_file = 0;
    read_file_begin_entry_off      = 0;
    //$open file get fp
    FILE* fp = fopen(file_names[i], "rb");
    if (fp == nullptr) { WHOLEGRAPH_ERROR("Open file %s for read failed.", file_names[i]); }
    /*|***read_file_begin_entry_off***|***entry_gap***|***cur_file_read_entry_count***|******|*/
    while (read_file_begin_entry_off < file_entry_count) {
      //$fseek by remain_entry_gap
      if (read_file_begin_entry_off + next_entry_gap >= file_entry_count) {
        next_entry_gap = (read_file_begin_entry_off + next_entry_gap) - file_entry_count;
        break;
      }
      size_t file_read_start_offset = next_entry_gap * entry_size;
      if (fseeko(fp, file_read_start_offset, SEEK_CUR) != 0) {
        WHOLEGRAPH_ERROR("File %s seek to %ld failed.", file_names[i], file_read_start_offset);
      }
      size_t cur_file_read_entry_count;
      if (read_file_begin_entry_off + next_entry_gap + next_continuous_entry_count >
          file_entry_count) {
        cur_file_read_entry_count = file_entry_count - read_file_begin_entry_off - next_entry_gap;
        total_read_entry += cur_file_read_entry_count;
        read_file_begin_entry_off = file_entry_count;
        next_continuous_entry_count -= cur_file_read_entry_count;
        next_entry_gap = 0;
      } else {
        cur_file_read_entry_count = next_continuous_entry_count;
        total_read_entry += cur_file_read_entry_count;
        read_file_begin_entry_off += cur_file_read_entry_count + next_entry_gap;
        next_continuous_entry_count = round_robin_size > local_entry_count - total_read_entry
                                        ? local_entry_count - total_read_entry
                                        : round_robin_size;
        next_entry_gap              = (wg_world_size - 1) * round_robin_size;
      }
      read_size_from_cur_file += cur_file_read_entry_count * entry_size;
      // read cur_file_read_entry_count of embeddings
      size_t cur_file_read_entry = cur_file_read_entry_count;
      while (cur_file_read_entry_count > 0) {
        size_t read_entry_count = std::min(cur_file_read_entry_count, buffer_entry_count);
        int ret                 = fread(file_read_buffer.data(), entry_size, read_entry_count, fp);
        if (ret != read_entry_count) {
          WHOLEGRAPH_ERROR(
            "File %s line %d: reading from file %s, read_entry_count=%ld, entry_size=%ld, "
            "returned %d, error=%s\n",
            __FILE__,
            __LINE__,
            file_names[i],
            read_entry_count,
            entry_size,
            ret,
            strerror(errno));
        }
        if (entry_size != memory_entry_stride) {
          WG_CUDA_CHECK(cudaMemcpy2D(local_write_ptr,
                                     memory_entry_stride,
                                     file_read_buffer.data(),
                                     entry_size,
                                     entry_size,
                                     read_entry_count,
                                     cudaMemcpyDefault));
        } else {
          WG_CUDA_CHECK(cudaMemcpy(local_write_ptr,
                                   file_read_buffer.data(),
                                   read_entry_count * entry_size,
                                   cudaMemcpyDefault));
        }
        local_write_ptr += read_entry_count * memory_entry_stride;
        cur_file_read_entry_count -= read_entry_count;
      }
      if (total_read_entry > local_entry_count) {
        WHOLEGRAPH_ERROR(
          "file read error from rank %d, should read %lu entries, infact %lu entries.",
          wg_rank,
          local_entry_count,
          total_read_entry);
        break;
      } else if (total_read_entry == local_entry_count) {
        break;
      }
    }
    fclose(fp);
    WHOLEGRAPH_INFO("Rank=%d done Reading %ld bytes from file %s size=%ld",
                     wg_rank,
                     read_size_from_cur_file,
                     file_names[i],
                     file_sizes[i]);
    if (total_read_entry == local_entry_count) break;
  }
  WHOLEGRAPH_INFO("Rank=%d done Reading %ld entries, infact read %ld entries",
                   wg_rank,
                   total_read_entry,
                   local_entry_count);
}

/*!
 * Read from file list to local memory of WholeGraph. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeGraph will read the files in parallel
 * and load each part into local memory of each rank.
 * @param local_ptr : Pointer to local memory of WholeGraph
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeGraph.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeGraph
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wg_rank : WholeGraph rank.
 */
static void read_file_list_to_local_memory(char* local_ptr,
                                           size_t local_size,
                                           size_t local_offset,
                                           size_t entry_size,
                                           size_t memory_entry_stride,
                                           size_t memory_offset,
                                           int file_count,
                                           const char** file_names,
                                           const std::vector<size_t>& file_sizes,
                                           size_t suggested_buffer_size,
                                           int wg_rank)
{
  size_t buffer_size;
  size_t buffer_entry_count = 1;
  if (suggested_buffer_size < entry_size) {
    buffer_size = entry_size;
  } else {
    buffer_entry_count = suggested_buffer_size / entry_size;
    buffer_size        = buffer_entry_count * entry_size;
  }
  std::vector<char> file_read_buffer(buffer_size);

  size_t local_entry_memory_start_index = local_offset / memory_entry_stride;
  size_t local_entry_file_start_index =
    local_entry_memory_start_index - memory_offset / memory_entry_stride;
  size_t local_entry_count = local_size / memory_entry_stride;
  char* local_write_ptr    = local_ptr + memory_offset % memory_entry_stride;
  if (wg_rank == 0) {
    local_entry_count -= memory_offset / memory_entry_stride;
    local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
  }
  size_t local_entry_idx = 0;

  size_t file_entry_offset = 0;
  size_t total_read_bytes  = 0;
  for (int i = 0; i < file_count; i++) {
    size_t file_entry_count = file_sizes[i] / entry_size;
    // already outside reading window
    if (file_entry_offset >= local_entry_file_start_index + local_entry_count) break;
    // in reading window
    if (file_entry_offset + file_entry_count > local_entry_file_start_index) {
      size_t file_read_start_offset = 0;
      FILE* fp                      = fopen(file_names[i], "rb");
      if (fp == nullptr) { WHOLEGRAPH_ERROR("Open file %s for read failed.", file_names[i]); }
      // maybe in window end, remove possible tailing data that don't belong to current rank.
      size_t to_read_file_entry_count = std::min(
        file_entry_count, local_entry_file_start_index + local_entry_count - file_entry_offset);
      // if in window begin, remove possible data that belongs to previous rank and skip disk
      // data.
      if (file_entry_offset < local_entry_file_start_index) {
        size_t skip_entry_count = local_entry_file_start_index - file_entry_offset;

        file_read_start_offset = skip_entry_count * entry_size;

        if (fseeko(fp, file_read_start_offset, SEEK_SET) != 0) {
          WHOLEGRAPH_ERROR(
            "File %s seek to %ld failed.", file_names[i], skip_entry_count * entry_size);
        }
        to_read_file_entry_count -= skip_entry_count;
      }
      // now all data in file_entry_count need to be read.
      size_t bytes_to_read    = to_read_file_entry_count * entry_size;
      size_t left_entry_count = to_read_file_entry_count;
      while (left_entry_count > 0) {
        size_t read_entry_count = std::min(left_entry_count, buffer_entry_count);

        int ret = fread(file_read_buffer.data(), entry_size, read_entry_count, fp);
        if (ret != read_entry_count) {
          WHOLEGRAPH_ERROR(
            "File %s line %d: reading from file %s, read_entry_count=%ld, entry_size=%ld, "
            "returned %d, error=%s\n",
            __FILE__,
            __LINE__,
            file_names[i],
            read_entry_count,
            entry_size,
            ret,
            strerror(errno));
        }

        if (entry_size != memory_entry_stride) {
          WG_CUDA_CHECK(cudaMemcpy2D(local_write_ptr,
                                     memory_entry_stride,
                                     file_read_buffer.data(),
                                     entry_size,
                                     entry_size,
                                     read_entry_count,
                                     cudaMemcpyDefault));
        } else {
          WG_CUDA_CHECK(cudaMemcpy(local_write_ptr,
                                   file_read_buffer.data(),
                                   read_entry_count * entry_size,
                                   cudaMemcpyDefault));
        }
        local_write_ptr += read_entry_count * memory_entry_stride;

        left_entry_count -= read_entry_count;
      }
      fclose(fp);
      WHOLEGRAPH_INFO(
        "Rank=%d done Reading %ld bytes from file %s size=%ld, starting from offset=%ld.",
        wg_rank,
        bytes_to_read,
        file_names[i],
        file_sizes[i],
        file_read_start_offset);
      total_read_bytes += bytes_to_read;
    }
    file_entry_offset += file_entry_count;
  }
  WHOLEGRAPH_INFO(
    "Rank=%d done reading total %ld bytes from needed files.", wg_rank, total_read_bytes);
}

/*!
 * Read from file list to local memory of WholeGraph. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeGraph will read the files in parallel
 * and load each part into local memory of each rank. WholeGraph will use round-robin sharding
 * strategy.
 * @param local_ptr : Pointer to local memory of WholeGraph
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeGraph.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeGraph
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wg_rank : WholeGraph rank.
 * @param wg_world_size : WholeGraph world size.
 * @param round_robin_size : continuous embedding size of a rank using round robin shard strategy.
 * @param dev_id : the device bound to the rank.
 */
static void read_file_list_to_local_memory_roundrobin_with_multi_threads(
  char* local_ptr,
  size_t local_size,
  size_t local_offset,
  size_t entry_size,
  size_t memory_entry_stride,
  size_t memory_offset,
  int file_count,
  const char** file_names,
  const std::vector<size_t>& file_sizes,
  size_t suggested_buffer_size,
  int wg_rank,
  int wg_world_size,
  int round_robin_size,
  int dev_id)
{
  int threads_per_rank                 = 1;
  const char* threads_per_rank_env_var = std::getenv("WG_LOAD_THREADS_PER_RANK");
  if (threads_per_rank_env_var != nullptr) {
    try {
      threads_per_rank = std::stoi(threads_per_rank_env_var);
    } catch (const std::invalid_argument& e) {
      threads_per_rank = 1;
      WHOLEGRAPH_WARN(
        "Environment variable WG_LOAD_THREADS_PER_RANK value %s is not valid,use the default  %d",
        threads_per_rank_env_var,
        threads_per_rank);
    }
    if (threads_per_rank < 1 || threads_per_rank > std::thread::hardware_concurrency()) {
      threads_per_rank = 1;
      WHOLEGRAPH_WARN(
        "Environment variable WG_LOAD_THREADS_PER_RANK value %s is not valid,use the default  %d",
        threads_per_rank_env_var,
        threads_per_rank);
    }
  }
  size_t buffer_size;
  size_t buffer_entry_count = 1;
  if (suggested_buffer_size < entry_size) {
    buffer_size = entry_size;
  } else {
    buffer_entry_count = suggested_buffer_size / entry_size;
    buffer_size        = buffer_entry_count * entry_size;
  }

  std::atomic_size_t total_read_entry = 0;

  if (memory_offset >= memory_entry_stride)
    WHOLEGRAPH_ERROR("memory offset %lu should be less than memory entry stride %lu.",
                      memory_offset,
                      memory_entry_stride);
  size_t total_file_sizes = 0;
  for (int i = 0; i < file_count; i++)
    total_file_sizes += file_sizes[i];
  size_t total_file_entry_count = total_file_sizes / entry_size;
  if (round_robin_size <= 0 || round_robin_size > total_file_entry_count / wg_world_size)
    WHOLEGRAPH_ERROR("illegal round_robin_size.");

  char* local_write_ptr = local_ptr + memory_offset % memory_entry_stride;

  size_t local_entry_memory_start_index = wg_rank * round_robin_size;
  size_t local_entry_file_start_index =
    local_entry_memory_start_index - memory_offset / memory_entry_stride;
  int extra_entry       = total_file_entry_count % (wg_world_size * round_robin_size);
  int local_extra_entry = (extra_entry > (wg_rank + 1) * round_robin_size)
                            ? round_robin_size
                            : extra_entry - wg_rank * round_robin_size;
  local_extra_entry     = local_extra_entry > 0 ? local_extra_entry : 0;
  size_t local_entry_count =
    total_file_entry_count / (wg_world_size * round_robin_size) * round_robin_size;

  if (wg_rank == 0) {
    local_entry_count -= memory_offset / memory_entry_stride;
    local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
  }

  int64_t local_round_robin_count = local_entry_count / round_robin_size;

  auto read_file_thread_fun = [=, &total_read_entry](int thread_id, int thread_num) {
    WG_CUDA_CHECK(cudaSetDevice(dev_id));
    std::vector<char> file_read_buffer(buffer_size);

    int64_t round_robin_count_per_thread = (local_round_robin_count + thread_num - 1) / thread_num;
    int64_t round_robin_count_this_thread =
      std::max(0L,
               std::min(round_robin_count_per_thread,
                        local_round_robin_count - round_robin_count_per_thread * thread_id));
    int64_t local_entry_count_this_thread = round_robin_count_this_thread * round_robin_size;
    if (thread_id == thread_num - 1) {
      // last thread
      local_entry_count_this_thread += local_extra_entry;
    }

    if (local_entry_count_this_thread == 0) return;
    int64_t start_round_robin_id_in_local = thread_id * round_robin_count_per_thread;

    if (round_robin_count_this_thread == 0) {
      // last thread
      if (round_robin_count_per_thread != 1) {
        WHOLEGRAPH_ERROR("round_robin_count_per_thread should be 1,but get %d \n",
                          round_robin_count_per_thread);
      }
      start_round_robin_id_in_local = local_round_robin_count;
    }

    size_t local_entry_file_start_index_this_thread =
      local_entry_file_start_index +
      start_round_robin_id_in_local * wg_world_size * round_robin_size;
    char* this_thread_write_ptr =
      local_write_ptr + start_round_robin_id_in_local * round_robin_size * memory_entry_stride;

    size_t total_read_entry_this_thread = 0;
    size_t next_entry_gap               = local_entry_file_start_index_this_thread;
    size_t next_continuous_entry_count =
      round_robin_size > local_entry_count_this_thread - total_read_entry_this_thread
        ? local_entry_count_this_thread - total_read_entry_this_thread
        : round_robin_size;
    size_t read_file_begin_entry_off = 0;
    for (int i = 0; i < file_count; i++) {
      size_t file_entry_count = file_sizes[i] / entry_size;
      if (file_entry_count <= next_entry_gap) {
        next_entry_gap -= file_entry_count;
        continue;
      }
      size_t read_size_from_cur_file = 0;
      read_file_begin_entry_off      = 0;
      //$open file get fp
      FILE* fp = fopen(file_names[i], "rb");
      if (fp == nullptr) { WHOLEGRAPH_ERROR("Open file %s for read failed.", file_names[i]); }
      /*|***read_file_begin_entry_off***|***entry_gap***|***cur_file_read_entry_count***|******|*/

      while (read_file_begin_entry_off < file_entry_count) {
        //$fseek by remain_entry_gap
        if (read_file_begin_entry_off + next_entry_gap >= file_entry_count) {
          next_entry_gap = (read_file_begin_entry_off + next_entry_gap) - file_entry_count;
          break;
        }
        size_t file_read_start_offset = next_entry_gap * entry_size;
        if (fseeko(fp, file_read_start_offset, SEEK_CUR) != 0) {
          WHOLEGRAPH_ERROR("File %s seek to %ld failed.", file_names[i], file_read_start_offset);
        }

        size_t cur_file_read_entry_count;
        if (read_file_begin_entry_off + next_entry_gap + next_continuous_entry_count >
            file_entry_count) {
          cur_file_read_entry_count = file_entry_count - read_file_begin_entry_off - next_entry_gap;
          total_read_entry_this_thread += cur_file_read_entry_count;
          read_file_begin_entry_off = file_entry_count;
          next_continuous_entry_count -= cur_file_read_entry_count;
          next_entry_gap = 0;
        } else {
          cur_file_read_entry_count = next_continuous_entry_count;
          total_read_entry_this_thread += cur_file_read_entry_count;
          read_file_begin_entry_off += cur_file_read_entry_count + next_entry_gap;
          next_continuous_entry_count =
            round_robin_size > local_entry_count_this_thread - total_read_entry_this_thread
              ? local_entry_count_this_thread - total_read_entry_this_thread
              : round_robin_size;
          next_entry_gap = (wg_world_size - 1) * round_robin_size;
        }
        read_size_from_cur_file += cur_file_read_entry_count * entry_size;
        // read cur_file_read_entry_count of embeddings
        size_t cur_file_read_entry = cur_file_read_entry_count;
        while (cur_file_read_entry_count > 0) {
          size_t read_entry_count = std::min(cur_file_read_entry_count, buffer_entry_count);
          int ret = fread(file_read_buffer.data(), entry_size, read_entry_count, fp);
          if (ret != read_entry_count) {
            WHOLEGRAPH_ERROR(
              "File %s line %d: reading from file %s, read_entry_count=%ld, entry_size=%ld, "
              "returned %d, error=%s\n",
              __FILE__,
              __LINE__,
              file_names[i],
              read_entry_count,
              entry_size,
              ret,
              strerror(errno));
          }
          if (entry_size != memory_entry_stride) {
            WG_CUDA_CHECK(cudaMemcpy2D(this_thread_write_ptr,
                                       memory_entry_stride,
                                       file_read_buffer.data(),
                                       entry_size,
                                       entry_size,
                                       read_entry_count,
                                       cudaMemcpyDefault));
          } else {
            WG_CUDA_CHECK(cudaMemcpy(this_thread_write_ptr,
                                     file_read_buffer.data(),
                                     read_entry_count * entry_size,
                                     cudaMemcpyDefault));
          }
          this_thread_write_ptr += read_entry_count * memory_entry_stride;
          cur_file_read_entry_count -= read_entry_count;
        }
        if (total_read_entry_this_thread > local_entry_count_this_thread) {
          WHOLEGRAPH_ERROR(
            "file read error from rank %d, thread_id %d, should read %lu entries, infact %lu "
            "entries.",
            wg_rank,
            thread_id,
            local_entry_count,
            local_entry_count_this_thread);
          break;
        } else if (total_read_entry_this_thread == local_entry_count_this_thread) {
          break;
        }
      }

      fclose(fp);
      WHOLEGRAPH_INFO("Rank=%d thread_id=%d ,done Reading %ld bytes from file %s size=%ld",
                       wg_rank,
                       thread_id,
                       read_size_from_cur_file,
                       file_names[i],
                       file_sizes[i]);

      if (total_read_entry_this_thread == local_entry_count_this_thread) break;
    }
    total_read_entry.fetch_add(total_read_entry_this_thread);
  };

  WHOLEGRAPH_INFO("Rank=%d use %d threads to read file.", wg_rank, threads_per_rank);

  if (threads_per_rank > 1) {
    std::vector<std::thread> read_file_threads;
    read_file_threads.reserve(threads_per_rank);
    for (int i = 0; i < threads_per_rank; i++) {
      read_file_threads.emplace_back(read_file_thread_fun, i, threads_per_rank);
    }

    for (auto&& thread : read_file_threads) {
      thread.join();
    }
  } else {
    read_file_thread_fun(0, 1);
  }

  WHOLEGRAPH_INFO("Rank=%d done Reading %ld entries, infact read %ld entries",
                   wg_rank,
                   total_read_entry.load(),
                   local_entry_count);
};

/*!
 * Read from file list to local memory of WholeGraph. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeGraph will read the files in parallel
 * and load each part into local memory of each rank.
 * @param local_ptr : Pointer to local memory of WholeGraph
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeGraph.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeGraph
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wg_rank : WholeGraph rank.
 * @param wg_world_size : WholeGraph world size.
 * @param dev_id : the device bound to the rank.
 */
static void read_file_list_to_local_memory_with_multi_threads(char* local_ptr,
                                                              size_t local_size,
                                                              size_t local_offset,
                                                              size_t entry_size,
                                                              size_t memory_entry_stride,
                                                              size_t memory_offset,
                                                              int file_count,
                                                              const char** file_names,
                                                              const std::vector<size_t>& file_sizes,
                                                              size_t suggested_buffer_size,
                                                              int wg_rank,
                                                              int wg_world_size,
                                                              int dev_id)
{
  int threads_per_rank                 = 1;
  const char* threads_per_rank_env_var = std::getenv("WG_LOAD_THREADS_PER_RANK");
  if (threads_per_rank_env_var != nullptr) {
    try {
      threads_per_rank = std::stoi(threads_per_rank_env_var);
    } catch (const std::invalid_argument& e) {
      threads_per_rank = 1;
      WHOLEGRAPH_WARN(
        "Environment variable WG_LOAD_THREADS_PER_RANK value %s is not valid,use the default  %d",
        threads_per_rank_env_var,
        threads_per_rank);
    }
    if (threads_per_rank < 1 || threads_per_rank > std::thread::hardware_concurrency()) {
      threads_per_rank = 1;
      WHOLEGRAPH_WARN(
        "Environment variable WG_LOAD_THREADS_PER_RANK value %s is not valid,use the default  %d",
        threads_per_rank_env_var,
        threads_per_rank);
    }
  }
  size_t buffer_size;
  size_t buffer_entry_count = 1;
  if (suggested_buffer_size < entry_size) {
    buffer_size = entry_size;
  } else {
    buffer_entry_count = suggested_buffer_size / entry_size;
    buffer_size        = buffer_entry_count * entry_size;
  }

  size_t local_entry_memory_start_index = local_offset / memory_entry_stride;
  size_t local_entry_file_start_index =
    local_entry_memory_start_index - memory_offset / memory_entry_stride;
  size_t local_entry_count = local_size / memory_entry_stride;
  char* local_write_ptr    = local_ptr + memory_offset % memory_entry_stride;
  if (wg_rank == 0) {
    local_entry_count -= memory_offset / memory_entry_stride;
    local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
  }
  std::atomic_size_t total_read_bytes = 0;

  auto read_file_thread_fun = [=, &total_read_bytes](int thread_id, int thread_num) {
    WG_CUDA_CHECK(cudaSetDevice(dev_id));
    const size_t entry_count_per_thread = (local_entry_count + thread_num - 1) / thread_num;
    const size_t entry_count_this_thread =
      std::min(entry_count_per_thread, local_entry_count - entry_count_per_thread * thread_id);
    const size_t entry_file_start_index_this_thread =
      local_entry_file_start_index + thread_id * entry_count_per_thread;
    char* this_thread_write_ptr =
      local_write_ptr + entry_count_per_thread * thread_id * memory_entry_stride;

    std::vector<char> file_read_buffer(buffer_size);

    if (entry_count_this_thread <= 0) return;
    size_t file_entry_offset     = 0;
    size_t read_size_this_thread = 0;

    for (int i = 0; i < file_count; i++) {
      size_t file_entry_count = file_sizes[i] / entry_size;
      // already outside reading window
      if (file_entry_offset >= (entry_file_start_index_this_thread + entry_count_this_thread))
        break;

      // in reading window
      if (file_entry_offset + file_entry_count > entry_file_start_index_this_thread) {
        size_t file_read_start_offset = 0;
        FILE* fp                      = fopen(file_names[i], "rb");
        if (fp == nullptr) { WHOLEGRAPH_ERROR("Open file %s for read failed.", file_names[i]); }
        // maybe in window end, remove possible tailing data that don't belong to current rank.
        size_t to_read_file_entry_count = std::min(
          file_entry_count,
          entry_file_start_index_this_thread + entry_count_this_thread - file_entry_offset);
        // if in window begin, remove possible data that belongs to previous rank and skip disk
        // data.
        if (file_entry_offset < entry_file_start_index_this_thread) {
          size_t skip_entry_count = entry_file_start_index_this_thread - file_entry_offset;

          file_read_start_offset = skip_entry_count * entry_size;

          if (fseeko(fp, file_read_start_offset, SEEK_SET) != 0) {
            WHOLEGRAPH_ERROR(
              "File %s seek to %ld failed.", file_names[i], skip_entry_count * entry_size);
          }
          to_read_file_entry_count -= skip_entry_count;
        }
        // now all data in file_entry_count need to be read.
        size_t bytes_to_read    = to_read_file_entry_count * entry_size;
        size_t left_entry_count = to_read_file_entry_count;
        while (left_entry_count > 0) {
          size_t read_entry_count = std::min(left_entry_count, buffer_entry_count);

          int ret = fread(file_read_buffer.data(), entry_size, read_entry_count, fp);
          if (ret != read_entry_count) {
            WHOLEGRAPH_ERROR(
              "File %s line %d: reading from file %s, read_entry_count=%ld, entry_size=%ld, "
              "returned %d, error=%s\n",
              __FILE__,
              __LINE__,
              file_names[i],
              read_entry_count,
              entry_size,
              ret,
              strerror(errno));
          }

          if (entry_size != memory_entry_stride) {
            WG_CUDA_CHECK(cudaMemcpy2D(this_thread_write_ptr,
                                       memory_entry_stride,
                                       file_read_buffer.data(),
                                       entry_size,
                                       entry_size,
                                       read_entry_count,
                                       cudaMemcpyDefault));
          } else {
            WHOLEGRAPH_INFO(
              "Rank:%d, threadid:%d, cuda Memcpy : this_thread_write_ptr:%p, "
              "file_read_buffer.data():%p, read_entry_count:%d, entry_size:%d\n",
              wg_rank,
              thread_id,
              this_thread_write_ptr,
              file_read_buffer.data(),
              read_entry_count,
              entry_size);
            WG_CUDA_CHECK(cudaMemcpy(this_thread_write_ptr,
                                     file_read_buffer.data(),
                                     read_entry_count * entry_size,
                                     cudaMemcpyDefault));
          }
          this_thread_write_ptr += read_entry_count * memory_entry_stride;

          left_entry_count -= read_entry_count;
        }
        fclose(fp);
        WHOLEGRAPH_INFO(
          "Rank=%d thread_id=%d done Reading %ld bytes from file %s size=%ld, starting from "
          "offset=%ld.",
          wg_rank,
          thread_id,
          bytes_to_read,
          file_names[i],
          file_sizes[i],
          file_read_start_offset);
        total_read_bytes.fetch_add(bytes_to_read);
        read_size_this_thread += bytes_to_read;
      }

      file_entry_offset += file_entry_count;
    }

    WHOLEGRAPH_INFO("Rank=%d thread_id=%d done Reading %ld bytes from needed files size.",
                     wg_rank,
                     thread_id,
                     read_size_this_thread);
  };
  WHOLEGRAPH_INFO("Rank=%d use %d threads to read file.", wg_rank, threads_per_rank);

  if (threads_per_rank > 1) {
    std::vector<std::thread> read_file_threads;
    read_file_threads.reserve(threads_per_rank);
    for (int i = 0; i < threads_per_rank; i++) {
      read_file_threads.emplace_back(read_file_thread_fun, i, threads_per_rank);
    }

    for (auto&& thread : read_file_threads) {
      thread.join();
    }
  } else {
    read_file_thread_fun(0, 1);
  }
  WHOLEGRAPH_INFO(
    "Rank=%d done reading total %ld bytes from needed files.", wg_rank, total_read_bytes.load());
}

/*!
 * Read from file list to local memory of WholeGraph using DirectIO. Using DirectIO may have better
 * performance by bypassing system cache if it is bottleneck. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeGraph will read the files in parallel
 * and load each part into local memory of each rank. WholeGraph uses round-robin sharding strategy
 * here.
 * @param local_ptr : Pointer to local memory of WholeGraph
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeGraph.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeGraph
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wg_rank : WholeGraph rank.
 * @param wg_world_size : WholeGraph world size.
 * @param round_robin_size : continuous embedding size of a rank using round robin shard strategy.
 */
static void read_file_list_to_local_memory_roundrobin_directio(
  char* local_ptr,
  size_t local_size,
  size_t local_offset,
  size_t entry_size,
  size_t memory_entry_stride,
  size_t memory_offset,
  int file_count,
  const char** file_names,
  const std::vector<size_t>& file_sizes,
  size_t suggested_buffer_size,
  int wg_rank,
  int wg_world_size,
  int round_robin_size)
{
  if (memory_offset + entry_size > memory_entry_stride)
    WHOLEGRAPH_FAIL_NOTHROW("Direct io mode only support reading all entries.");

  static size_t kAlignSize = 16 * 1024 * 1024;
  suggested_buffer_size    = round_up_unsafe<size_t>(suggested_buffer_size, kAlignSize);

  char* block_buffer;
  WHOLEGRAPH_CHECK_NOTHROW(posix_memalign(reinterpret_cast<void**>(&block_buffer),
                                           kAlignSize,
                                           suggested_buffer_size) == 0);

  size_t total_file_sizes = 0;
  for (int i = 0; i < file_count; i++)
    total_file_sizes += file_sizes[i];
  size_t total_file_entry_count = total_file_sizes / entry_size;
  if (round_robin_size <= 0 || round_robin_size > total_file_entry_count / wg_world_size)
    WHOLEGRAPH_ERROR("illegal round_robin_size.");

  size_t local_entry_memory_start_index = wg_rank * round_robin_size;
  size_t local_entry_file_start_index =
    local_entry_memory_start_index - memory_offset / memory_entry_stride;

  int extra_entry       = total_file_entry_count % (wg_world_size * round_robin_size);
  int local_extra_entry = (extra_entry > (wg_rank + 1) * round_robin_size)
                            ? round_robin_size
                            : extra_entry - wg_rank * round_robin_size;
  local_extra_entry     = local_extra_entry > 0 ? local_extra_entry : 0;
  size_t local_entry_count =
    total_file_entry_count / (wg_world_size * round_robin_size) * round_robin_size;
  local_entry_count += local_extra_entry;

  char* local_write_ptr = local_ptr + memory_offset % memory_entry_stride;
  if (wg_rank == 0) {
    local_entry_count -= memory_offset / memory_entry_stride;
    local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
  }

  size_t total_read_entry            = 0;
  size_t next_entry_gap              = local_entry_file_start_index;
  size_t next_continuous_entry_count = round_robin_size > local_entry_count - total_read_entry
                                         ? local_entry_count - total_read_entry
                                         : round_robin_size;
  size_t read_file_begin_entry_off   = 0;
  for (int i = 0; i < file_count; i++) {
    size_t file_entry_count = file_sizes[i] / entry_size;
    if (file_entry_count <= next_entry_gap) {
      next_entry_gap -= file_entry_count;
      continue;
    }

    auto block_size = StatFileBlockSize(file_names[i]);
    if (block_size == 0 || block_size == (size_t)-1 || kAlignSize % block_size != 0) {
      WHOLEGRAPH_FAIL_NOTHROW(
        "block_size=%ld for file %s, but alignment is %ld", block_size, file_names[i], kAlignSize);
    }

    size_t buffer_block_count = suggested_buffer_size / block_size;
    int fd                    = open(file_names[i], O_DIRECT | O_RDONLY);
    if (fd < 0) { WHOLEGRAPH_FAIL_NOTHROW("Open file %s with direct io failed.", file_names[i]); }

    size_t read_size_from_cur_file = 0;
    size_t useful_data_bytes_read  = 0;
    read_file_begin_entry_off      = 0;

    /*|***read_file_begin_entry_off***|***entry_gap***|***cur_file_read_entry_count***|******|*/
    while (read_file_begin_entry_off < file_entry_count) {
      if (read_file_begin_entry_off + next_entry_gap >= file_entry_count) {
        next_entry_gap = (read_file_begin_entry_off + next_entry_gap) - file_entry_count;
        break;
      }
      size_t cur_file_read_entry_count;
      if (read_file_begin_entry_off + next_entry_gap + next_continuous_entry_count >
          file_entry_count) {
        cur_file_read_entry_count = file_entry_count - read_file_begin_entry_off - next_entry_gap;
      } else {
        cur_file_read_entry_count = next_continuous_entry_count;
      }

      // read concerned vars
      size_t cur_read_entry_start = read_file_begin_entry_off + next_entry_gap;
      size_t cur_read_byte_start  = (cur_read_entry_start * entry_size) / block_size * block_size;
      size_t cur_read_byte_end    = (cur_read_entry_start + cur_file_read_entry_count) * entry_size;
      size_t skip_head_size       = cur_read_entry_start * entry_size - cur_read_byte_start;
      // write concerned vars
      char* local_mem_write_entry_for_file =
        local_write_ptr + total_read_entry * memory_entry_stride;
      size_t first_mem_entry_offset = 0;

      while (cur_read_byte_start < cur_read_byte_end) {
        size_t left_size          = cur_read_byte_end - cur_read_byte_start;
        size_t left_block_count   = div_rounding_up_unsafe(left_size, block_size);
        size_t read_block_count   = std::min(left_block_count, buffer_block_count);
        size_t physical_read_size = read_block_count * block_size;
        // physical_data_bytes_read += physical_read_size;
        read_size_from_cur_file += physical_read_size;

        ssize_t pread_size = pread64(fd, block_buffer, physical_read_size, cur_read_byte_start);
        if (pread_size != physical_read_size && cur_read_byte_start + pread_size != file_sizes[i]) {
          WHOLEGRAPH_FAIL_NOTHROW(
            "rank=%d, pread_size=%ld, physical_read_size=%ld, file_block_read_offset=%ld, "
            "file_sizes[i]=%ld, file=%s",
            wg_rank,
            pread_size,
            physical_read_size,
            cur_read_byte_start,
            file_sizes[i],
            file_names[i]);
        }
        physical_read_size    = pread_size;
        size_t drop_tail_size = 0;
        if (cur_read_byte_start + physical_read_size > cur_read_byte_end) {
          drop_tail_size = cur_read_byte_start + physical_read_size - cur_read_byte_end;
        }

        char* useful_data_ptr   = block_buffer + skip_head_size;
        size_t useful_data_size = physical_read_size - skip_head_size - drop_tail_size;
        useful_data_bytes_read += useful_data_size;

        if (first_mem_entry_offset != 0) {
          size_t entry_left_size = entry_size - first_mem_entry_offset;
          WG_CUDA_CHECK_NO_THROW(cudaMemcpy(local_mem_write_entry_for_file + first_mem_entry_offset,
                                            useful_data_ptr,
                                            entry_left_size,
                                            cudaMemcpyDefault));
          local_mem_write_entry_for_file += memory_entry_stride;
          useful_data_ptr += entry_left_size;
          useful_data_size -= entry_left_size;
          entry_left_size = 0;
        }

        size_t full_entry_count = useful_data_size / entry_size;
        size_t full_entry_size  = full_entry_count * entry_size;

        if (full_entry_size > 0) {
          if (entry_size != memory_entry_stride) {
            WG_CUDA_CHECK(cudaMemcpy2D(local_mem_write_entry_for_file,
                                       memory_entry_stride,
                                       useful_data_ptr,
                                       entry_size,
                                       entry_size,
                                       full_entry_count,
                                       cudaMemcpyDefault));
          } else {
            WG_CUDA_CHECK(cudaMemcpy(
              local_mem_write_entry_for_file, useful_data_ptr, full_entry_size, cudaMemcpyDefault));
          }
          local_mem_write_entry_for_file += memory_entry_stride * full_entry_count;
          useful_data_ptr += full_entry_size;
          useful_data_size -= full_entry_size;
        }

        size_t tail_entry_size = useful_data_size % entry_size;
        first_mem_entry_offset = tail_entry_size;
        if (tail_entry_size != 0) {
          // process tail
          WG_CUDA_CHECK_NO_THROW(cudaMemcpy(
            local_mem_write_entry_for_file, useful_data_ptr, tail_entry_size, cudaMemcpyDefault));
        }

        cur_read_byte_start += physical_read_size;
        skip_head_size = 0;
      }

      total_read_entry += cur_file_read_entry_count;
      // read_size_from_cur_file += cur_file_read_entry_count * entry_size;
      if (read_file_begin_entry_off + next_entry_gap + next_continuous_entry_count >
          file_entry_count) {
        read_file_begin_entry_off = file_entry_count;
        next_continuous_entry_count -= cur_file_read_entry_count;
        next_entry_gap = 0;
      } else {
        read_file_begin_entry_off += cur_file_read_entry_count + next_entry_gap;
        next_continuous_entry_count = round_robin_size > local_entry_count - total_read_entry
                                        ? local_entry_count - total_read_entry
                                        : round_robin_size;
        next_entry_gap              = (wg_world_size - 1) * round_robin_size;
      }
      if (total_read_entry > local_entry_count) {
        WHOLEGRAPH_ERROR(
          "file read error from rank %d, should read %lu entries, infact %lu entries.",
          wg_rank,
          local_entry_count,
          total_read_entry);
        break;
      } else if (total_read_entry == local_entry_count) {
        break;
      }
    }
    close(fd);
    WHOLEGRAPH_INFO(
      "Rank=%d done Reading useful %ld bytes by totally reading %ld bytes from file %s size=%ld "
      "using direct IO",
      wg_rank,
      useful_data_bytes_read,
      read_size_from_cur_file,
      file_names[i],
      file_sizes[i]);
    if (total_read_entry == local_entry_count) break;
  }
  WHOLEGRAPH_INFO("Rank=%d done Reading %ld entries, infact read %ld entries",
                   wg_rank,
                   total_read_entry,
                   local_entry_count);
}

/*!
 * Read from file list to local memory of WholeGraph using DirectIO. Using DirectIO may have better
 * performance by bypassing system cache if it is bottleneck. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeGraph will read the files in parallel
 * and load each part into local memory of each rank.
 * @param local_ptr : Pointer to local memory of WholeGraph
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeGraph.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeGraph
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wg_rank : WholeGraph rank.
 */
static void read_file_list_to_local_memory_directio(char* local_ptr,
                                                    size_t local_size,
                                                    size_t local_offset,
                                                    size_t entry_size,
                                                    size_t memory_entry_stride,
                                                    size_t memory_offset,
                                                    int file_count,
                                                    const char** file_names,
                                                    const std::vector<size_t>& file_sizes,
                                                    size_t suggested_buffer_size,
                                                    int wg_rank)
{
  if (memory_offset + entry_size > memory_entry_stride) {
    WHOLEGRAPH_FAIL_NOTHROW("Direct io mode only support reading all entries.");
  }
  size_t local_entry_start_index = local_offset / memory_entry_stride;
  size_t local_entry_count       = local_size / memory_entry_stride;
  char* local_write_ptr          = local_ptr + memory_offset % memory_entry_stride;

  static size_t kAlignSize = 16 * 1024 * 1024;
  suggested_buffer_size    = round_up_unsafe<size_t>(suggested_buffer_size, kAlignSize);

  char* block_buffer;
  WHOLEGRAPH_CHECK_NOTHROW(posix_memalign(reinterpret_cast<void**>(&block_buffer),
                                           kAlignSize,
                                           suggested_buffer_size) == 0);

  size_t file_entry_offset = 0;
  size_t read_entry_count  = 0;
  for (int i = 0; i < file_count; i++) {
    size_t file_entry_count = file_sizes[i] / entry_size;
    // already outside reading window
    if (file_entry_offset >= local_entry_start_index + local_entry_count) break;
    // reading window not reached
    if (file_entry_offset + file_entry_count <= local_entry_start_index) {
      file_entry_offset += file_entry_count;
      continue;
    }
    // in reading window
    auto block_size = StatFileBlockSize(file_names[i]);
    if (block_size == 0 || block_size == (size_t)-1 || kAlignSize % block_size != 0) {
      WHOLEGRAPH_FAIL_NOTHROW(
        "block_size=%ld for file %s, but alignment is %ld", block_size, file_names[i], kAlignSize);
    }
    size_t buffer_block_count = suggested_buffer_size / block_size;
    int fd                    = open(file_names[i], O_DIRECT | O_RDONLY);
    if (fd < 0) { WHOLEGRAPH_FAIL_NOTHROW("Open file %s with direct io failed.", file_names[i]); }

    // maybe in window end, remove possible tailing data that don't belong to current rank.
    size_t to_read_file_entry_count =
      std::min(file_entry_count, local_entry_start_index + local_entry_count - file_entry_offset);

    size_t file_read_end = to_read_file_entry_count * entry_size;
    // if in window begin, remove possible data that belongs to previous rank and skip disk
    // data.
    size_t file_read_start = 0;
    if (file_entry_offset < local_entry_start_index) {
      size_t skip_entry_count = local_entry_start_index - file_entry_offset;
      to_read_file_entry_count -= skip_entry_count;
      file_read_start = skip_entry_count * entry_size;
    }

    size_t file_block_read_offset = file_read_start / block_size * block_size;
    size_t skip_head_size         = file_read_start - file_block_read_offset;

    char* local_mem_write_entry_for_file = local_write_ptr + read_entry_count * memory_entry_stride;
    size_t first_mem_entry_offset        = 0;
    size_t useful_data_bytes_read        = 0;
    size_t physical_data_bytes_read      = 0;
    while (file_block_read_offset < file_read_end) {
      size_t left_size          = file_read_end - file_block_read_offset;
      size_t left_block_count   = div_rounding_up_unsafe(left_size, block_size);
      size_t read_block_count   = std::min(left_block_count, buffer_block_count);
      size_t physical_read_size = read_block_count * block_size;
      physical_data_bytes_read += physical_read_size;

      ssize_t pread_size = pread64(fd, block_buffer, physical_read_size, file_block_read_offset);
      if (pread_size != physical_read_size &&
          file_block_read_offset + pread_size != file_sizes[i]) {
        WHOLEGRAPH_FAIL_NOTHROW(
          "rank=%d, pread_size=%ld, physical_read_size=%ld, file_block_read_offset=%ld, "
          "file_sizes[i]=%ld, file=%s",
          wg_rank,
          pread_size,
          physical_read_size,
          file_block_read_offset,
          file_sizes[i],
          file_names[i]);
      }
      physical_read_size    = pread_size;
      size_t drop_tail_size = 0;
      if (file_block_read_offset + physical_read_size > file_read_end) {
        drop_tail_size = file_block_read_offset + physical_read_size - file_read_end;
      }

      char* useful_data_ptr   = block_buffer + skip_head_size;
      size_t useful_data_size = physical_read_size - skip_head_size - drop_tail_size;

      useful_data_bytes_read += useful_data_size;

      if (first_mem_entry_offset != 0) {
        // process head
        size_t entry_left_size = entry_size - first_mem_entry_offset;
        WG_CUDA_CHECK_NO_THROW(cudaMemcpy(local_mem_write_entry_for_file + first_mem_entry_offset,
                                          useful_data_ptr,
                                          entry_left_size,
                                          cudaMemcpyDefault));
        local_mem_write_entry_for_file += memory_entry_stride;
        useful_data_ptr += entry_left_size;
        useful_data_size -= entry_left_size;
        entry_left_size = 0;
      }

      size_t full_entry_count = useful_data_size / entry_size;
      size_t full_entry_size  = full_entry_count * entry_size;

      if (full_entry_size > 0) {
        if (entry_size != memory_entry_stride) {
          WG_CUDA_CHECK(cudaMemcpy2D(local_mem_write_entry_for_file,
                                     memory_entry_stride,
                                     useful_data_ptr,
                                     entry_size,
                                     entry_size,
                                     full_entry_count,
                                     cudaMemcpyDefault));
        } else {
          WG_CUDA_CHECK(cudaMemcpy(
            local_mem_write_entry_for_file, useful_data_ptr, full_entry_size, cudaMemcpyDefault));
        }
        local_mem_write_entry_for_file += memory_entry_stride * full_entry_count;
        useful_data_ptr += full_entry_size;
        useful_data_size -= full_entry_size;
      }

      size_t tail_entry_size = useful_data_size % entry_size;
      first_mem_entry_offset = tail_entry_size;
      if (tail_entry_size != 0) {
        // process tail
        WG_CUDA_CHECK_NO_THROW(cudaMemcpy(
          local_mem_write_entry_for_file, useful_data_ptr, tail_entry_size, cudaMemcpyDefault));
      }

      file_block_read_offset += physical_read_size;
      skip_head_size = 0;
    }

    WHOLEGRAPH_INFO(
      "Rank=%d done Reading %ld useful bytes by reading %ld block bytes using DirectIO from file "
      "%s size=%ld.",
      wg_rank,
      useful_data_bytes_read,
      physical_data_bytes_read,
      file_names[i],
      file_sizes[i]);

    close(fd);
    file_entry_offset += file_entry_count;
    read_entry_count += to_read_file_entry_count;
  }
  free(block_buffer);
}

/*!
 * Read from file list to local memory of WholeGraph using DirectIO. Using DirectIO may have better
 * performance by bypassing system cache if it is bottleneck. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeGraph will read the files in parallel
 * and load each part into local memory of each rank.
 * @param local_ptr : Pointer to local memory of WholeGraph
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeGraph.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeGraph
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wg_rank : WholeGraph rank.
 * @param wg_world_size : WholeGraph world size.
 * @param dev_id : the device bound to the rank.
 */
static void read_file_list_to_local_memory_directio_with_multi_thread(
  char* local_ptr,
  size_t local_size,
  size_t local_offset,
  size_t entry_size,
  size_t memory_entry_stride,
  size_t memory_offset,
  int file_count,
  const char** file_names,
  const std::vector<size_t>& file_sizes,
  size_t suggested_buffer_size,
  int wg_rank,
  int wg_world_size,
  int dev_id)
{
  if (memory_offset + entry_size > memory_entry_stride) {
    WHOLEGRAPH_FAIL_NOTHROW("Direct io mode only support reading all entries.");
  }
  size_t local_entry_start_index = local_offset / memory_entry_stride;
  size_t local_entry_count       = local_size / memory_entry_stride;
  char* local_write_ptr          = local_ptr + memory_offset % memory_entry_stride;

  static size_t kAlignSize = 16 * 1024 * 1024;
  suggested_buffer_size    = round_up_unsafe<size_t>(suggested_buffer_size, kAlignSize);

  int threads_per_rank                 = 1;
  const char* threads_per_rank_env_var = std::getenv("WG_LOAD_THREADS_PER_RANK");
  if (threads_per_rank_env_var != nullptr) {
    try {
      threads_per_rank = std::stoi(threads_per_rank_env_var);
    } catch (const std::invalid_argument& e) {
      threads_per_rank = 1;
      WHOLEGRAPH_WARN(
        "Environment variable WG_LOAD_THREADS_PER_RANK value %s is not valid,use the default  %d",
        threads_per_rank_env_var,
        threads_per_rank);
    }
    if (threads_per_rank < 1 || threads_per_rank > std::thread::hardware_concurrency()) {
      threads_per_rank = 1;
      WHOLEGRAPH_WARN(
        "Environment variable WG_LOAD_THREADS_PER_RANK value %s is not valid,use the default  %d",
        threads_per_rank_env_var,
        threads_per_rank);
    }
  }

  auto read_file_thread_fun = [=](int thread_id, int thread_num) {
    WG_CUDA_CHECK(cudaSetDevice(dev_id));

    char* block_buffer;
    WHOLEGRAPH_CHECK_NOTHROW(posix_memalign(reinterpret_cast<void**>(&block_buffer),
                                             kAlignSize,
                                             suggested_buffer_size) == 0);

    const size_t entry_count_per_thread = (local_entry_count + thread_num - 1) / thread_num;
    const size_t entry_count_this_thread =
      std::min(entry_count_per_thread, local_entry_count - entry_count_per_thread * thread_id);
    const size_t entry_file_start_index_this_thread =
      local_entry_start_index + thread_id * entry_count_per_thread;
    const size_t this_thread_entry_start_index = entry_file_start_index_this_thread;
    char* this_thread_write_ptr =
      local_write_ptr + entry_count_per_thread * thread_id * memory_entry_stride;

    if (entry_count_this_thread <= 0) return;

    size_t file_entry_offset = 0;
    size_t read_entry_count  = 0;

    for (int i = 0; i < file_count; i++) {
      size_t file_entry_count = file_sizes[i] / entry_size;
      // already outside reading window
      if (file_entry_offset >= this_thread_entry_start_index + entry_count_this_thread) break;
      // reading window not reached
      if (file_entry_offset + file_entry_count <= this_thread_entry_start_index) {
        file_entry_offset += file_entry_count;
        continue;
      }
      // in reading window
      auto block_size = StatFileBlockSize(file_names[i]);
      if (block_size == 0 || block_size == (size_t)-1 || kAlignSize % block_size != 0) {
        WHOLEGRAPH_FAIL_NOTHROW("block_size=%ld for file %s, but alignment is %ld",
                                 block_size,
                                 file_names[i],
                                 kAlignSize);
      }
      size_t buffer_block_count = suggested_buffer_size / block_size;
      int fd                    = open(file_names[i], O_DIRECT | O_RDONLY);
      if (fd < 0) {
        WHOLEGRAPH_FAIL_NOTHROW("Open file %s with direct io failed.", file_names[i]);
      }

      // maybe in window end, remove possible tailing data that don't belong to current rank.
      size_t to_read_file_entry_count =
        std::min(file_entry_count,
                 this_thread_entry_start_index + entry_count_this_thread - file_entry_offset);

      size_t file_read_end = to_read_file_entry_count * entry_size;
      // if in window begin, remove possible data that belongs to previous rank and skip disk
      // data.
      size_t file_read_start = 0;
      if (file_entry_offset < this_thread_entry_start_index) {
        size_t skip_entry_count = this_thread_entry_start_index - file_entry_offset;
        to_read_file_entry_count -= skip_entry_count;
        file_read_start = skip_entry_count * entry_size;
      }

      size_t file_block_read_offset = file_read_start / block_size * block_size;
      size_t skip_head_size         = file_read_start - file_block_read_offset;

      char* local_mem_write_entry_for_file =
        this_thread_write_ptr + read_entry_count * memory_entry_stride;
      size_t first_mem_entry_offset   = 0;
      size_t useful_data_bytes_read   = 0;
      size_t physical_data_bytes_read = 0;
      while (file_block_read_offset < file_read_end) {
        size_t left_size          = file_read_end - file_block_read_offset;
        size_t left_block_count   = div_rounding_up_unsafe(left_size, block_size);
        size_t read_block_count   = std::min(left_block_count, buffer_block_count);
        size_t physical_read_size = read_block_count * block_size;
        physical_data_bytes_read += physical_read_size;

        ssize_t pread_size = pread64(fd, block_buffer, physical_read_size, file_block_read_offset);
        if (pread_size != physical_read_size &&
            file_block_read_offset + pread_size != file_sizes[i]) {
          WHOLEGRAPH_FAIL_NOTHROW(
            "rank=%d, pread_size=%ld, physical_read_size=%ld, file_block_read_offset=%ld, "
            "file_sizes[i]=%ld, file=%s",
            wg_rank,
            pread_size,
            physical_read_size,
            file_block_read_offset,
            file_sizes[i],
            file_names[i]);
        }
        physical_read_size = pread_size;

        size_t drop_tail_size = 0;
        if (file_block_read_offset + physical_read_size > file_read_end) {
          drop_tail_size = file_block_read_offset + physical_read_size - file_read_end;
        }

        char* useful_data_ptr   = block_buffer + skip_head_size;
        size_t useful_data_size = physical_read_size - skip_head_size - drop_tail_size;

        useful_data_bytes_read += useful_data_size;

        if (first_mem_entry_offset != 0) {
          // process head
          size_t entry_left_size = entry_size - first_mem_entry_offset;
          WG_CUDA_CHECK_NO_THROW(cudaMemcpy(local_mem_write_entry_for_file + first_mem_entry_offset,
                                            useful_data_ptr,
                                            entry_left_size,
                                            cudaMemcpyDefault));
          local_mem_write_entry_for_file += memory_entry_stride;
          useful_data_ptr += entry_left_size;
          useful_data_size -= entry_left_size;
          entry_left_size = 0;
        }

        size_t full_entry_count = useful_data_size / entry_size;
        size_t full_entry_size  = full_entry_count * entry_size;

        if (full_entry_size > 0) {
          if (entry_size != memory_entry_stride) {
            WG_CUDA_CHECK(cudaMemcpy2D(local_mem_write_entry_for_file,
                                       memory_entry_stride,
                                       useful_data_ptr,
                                       entry_size,
                                       entry_size,
                                       full_entry_count,
                                       cudaMemcpyDefault));
          } else {
            WG_CUDA_CHECK(cudaMemcpy(
              local_mem_write_entry_for_file, useful_data_ptr, full_entry_size, cudaMemcpyDefault));
          }
          local_mem_write_entry_for_file += memory_entry_stride * full_entry_count;
          useful_data_ptr += full_entry_size;
          useful_data_size -= full_entry_size;
        }

        size_t tail_entry_size = useful_data_size % entry_size;
        first_mem_entry_offset = tail_entry_size;

        if (tail_entry_size != 0) {
          // process tail
          WG_CUDA_CHECK_NO_THROW(cudaMemcpy(
            local_mem_write_entry_for_file, useful_data_ptr, tail_entry_size, cudaMemcpyDefault));
          // first_mem_entry_offset = tail_entry_size;
        }

        file_block_read_offset += physical_read_size;
        skip_head_size = 0;
      }

      WHOLEGRAPH_INFO(
        "Rank=%d threadid=%d done Reading %ld useful bytes by reading %ld block bytes using "
        "DirectIO from file "
        "%s size=%ld.",
        wg_rank,
        thread_id,
        useful_data_bytes_read,
        physical_data_bytes_read,
        file_names[i],
        file_sizes[i]);

      close(fd);
      file_entry_offset += file_entry_count;
      read_entry_count += to_read_file_entry_count;
    }
    free(block_buffer);
  };

  if (threads_per_rank != 1) {
    std::vector<std::thread> read_file_threads;
    read_file_threads.reserve(threads_per_rank);
    for (int i = 0; i < threads_per_rank; i++) {
      read_file_threads.emplace_back(read_file_thread_fun, i, threads_per_rank);
    }

    for (auto&& thread : read_file_threads) {
      thread.join();
    }
  } else {
    read_file_thread_fun(0, 1);
  }
}

/*!
 * Read from file list to local memory of WholeGraph using DirectIO. Using DirectIO may have better
 * performance by bypassing system cache if it is bottleneck. File list are binary files, which are
 * considered to be concatenated together. All ranks in WholeGraph will read the files in parallel
 * and load each part into local memory of each rank. WholeGraph uses round-robin sharding strategy
 * here.
 * @param local_ptr : Pointer to local memory of WholeGraph
 * @param local_size : Local memory size
 * @param local_offset : The offset of local memory in WholeGraph.
 * @param entry_size : The entry size of each data entry.
 * @param memory_entry_stride : The stride of each entry in WholeGraph
 * @param memory_offset : The start offset to place the read data. Should be in range [0,
 * memory_entry_stride)
 * @param file_count : Total file count of the file list
 * @param file_names : File names of the file list.
 * @param file_sizes : Sizes of each file.
 * @param suggested_buffer_size : Suggested buffer size to read.
 * @param wg_rank : WholeGraph rank.
 * @param wg_world_size : WholeGraph world size.
 * @param round_robin_size : continuous embedding size of a rank using round robin shard strategy.
 * @param dev_id : the device bound to the rank.
 */
static void read_file_list_to_local_memory_roundrobin_directio_with_multi_threads(
  char* local_ptr,
  size_t local_size,
  size_t local_offset,
  size_t entry_size,
  size_t memory_entry_stride,
  size_t memory_offset,
  int file_count,
  const char** file_names,
  const std::vector<size_t>& file_sizes,
  size_t suggested_buffer_size,
  int wg_rank,
  int wg_world_size,
  int round_robin_size,
  int dev_id)
{
  int threads_per_rank                 = 1;
  const char* threads_per_rank_env_var = std::getenv("WG_LOAD_THREADS_PER_RANK");
  if (threads_per_rank_env_var != nullptr) {
    try {
      threads_per_rank = std::stoi(threads_per_rank_env_var);
    } catch (const std::invalid_argument& e) {
      threads_per_rank = 1;
      WHOLEGRAPH_WARN(
        "Environment variable WG_LOAD_THREADS_PER_RANK value %s is not valid,use the default  %d",
        threads_per_rank_env_var,
        threads_per_rank);
    }
    if (threads_per_rank < 1 || threads_per_rank > std::thread::hardware_concurrency()) {
      threads_per_rank = 1;
      WHOLEGRAPH_WARN(
        "Environment variable WG_LOAD_THREADS_PER_RANK value %s is not valid,use the default  %d",
        threads_per_rank_env_var,
        threads_per_rank);
    }
  }

  if (memory_offset + entry_size > memory_entry_stride)
    WHOLEGRAPH_FAIL_NOTHROW("Direct io mode only support reading all entries.");

  static size_t kAlignSize = 16 * 1024 * 1024;
  suggested_buffer_size    = round_up_unsafe<size_t>(suggested_buffer_size, kAlignSize);

  size_t total_file_sizes = 0;
  for (int i = 0; i < file_count; i++)
    total_file_sizes += file_sizes[i];
  size_t total_file_entry_count = total_file_sizes / entry_size;
  if (round_robin_size <= 0 || round_robin_size > total_file_entry_count / wg_world_size)
    WHOLEGRAPH_ERROR("illegal round_robin_size.");
  char* local_write_ptr = local_ptr + memory_offset % memory_entry_stride;

  size_t local_entry_memory_start_index = wg_rank * round_robin_size;
  size_t local_entry_file_start_index =
    local_entry_memory_start_index - memory_offset / memory_entry_stride;

  int extra_entry       = total_file_entry_count % (wg_world_size * round_robin_size);
  int local_extra_entry = (extra_entry > (wg_rank + 1) * round_robin_size)
                            ? round_robin_size
                            : extra_entry - wg_rank * round_robin_size;
  local_extra_entry     = local_extra_entry > 0 ? local_extra_entry : 0;
  size_t local_entry_count =
    total_file_entry_count / (wg_world_size * round_robin_size) * round_robin_size;
  std::atomic_size_t total_read_entry = 0;
  if (wg_rank == 0) {
    local_entry_count -= memory_offset / memory_entry_stride;
    local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
  }

  int64_t local_round_robin_count = local_entry_count / round_robin_size;

  auto read_file_thread_fun = [=, &total_read_entry](int thread_id, int thread_num) {
    WG_CUDA_CHECK(cudaSetDevice(dev_id));

    char* block_buffer;
    WHOLEGRAPH_CHECK_NOTHROW(posix_memalign(reinterpret_cast<void**>(&block_buffer),
                                             kAlignSize,
                                             suggested_buffer_size) == 0);
    int64_t round_robin_count_per_thread = (local_round_robin_count + thread_num - 1) / thread_num;
    int64_t round_robin_count_this_thread =
      std::max(0L,
               std::min(round_robin_count_per_thread,
                        local_round_robin_count - round_robin_count_per_thread * thread_id));
    int64_t local_entry_count_this_thread = round_robin_count_this_thread * round_robin_size;
    if (thread_id == thread_num - 1) {
      // last thread
      local_entry_count_this_thread += local_extra_entry;
    }

    if (local_entry_count_this_thread == 0) return;
    int64_t start_round_robin_id_in_local = thread_id * round_robin_count_per_thread;

    if (round_robin_count_this_thread == 0) {
      // last thread
      if (round_robin_count_per_thread != 1) {
        WHOLEGRAPH_ERROR("round_robin_count_per_thread should be 1,but get %d \n",
                          round_robin_count_per_thread);
      }
      start_round_robin_id_in_local = local_round_robin_count;
    }

    size_t local_entry_file_start_index_this_thread =
      local_entry_file_start_index +
      start_round_robin_id_in_local * wg_world_size * round_robin_size;
    char* this_thread_write_ptr =
      local_write_ptr + start_round_robin_id_in_local * round_robin_size * memory_entry_stride;

    size_t total_read_entry_this_thread = 0;

    size_t next_entry_gap = local_entry_file_start_index_this_thread;
    size_t next_continuous_entry_count =
      round_robin_size > local_entry_count_this_thread - total_read_entry_this_thread
        ? local_entry_count_this_thread - total_read_entry_this_thread
        : round_robin_size;
    size_t read_file_begin_entry_off = 0;
    for (int i = 0; i < file_count; i++) {
      size_t file_entry_count = file_sizes[i] / entry_size;
      if (file_entry_count <= next_entry_gap) {
        next_entry_gap -= file_entry_count;
        continue;
      }

      auto block_size = StatFileBlockSize(file_names[i]);
      if (block_size == 0 || block_size == (size_t)-1 || kAlignSize % block_size != 0) {
        WHOLEGRAPH_FAIL_NOTHROW("block_size=%ld for file %s, but alignment is %ld",
                                 block_size,
                                 file_names[i],
                                 kAlignSize);
      }

      size_t buffer_block_count = suggested_buffer_size / block_size;
      int fd                    = open(file_names[i], O_DIRECT | O_RDONLY);
      if (fd < 0) {
        WHOLEGRAPH_FAIL_NOTHROW("Open file %s with direct io failed.", file_names[i]);
      }

      size_t read_size_from_cur_file = 0;
      size_t useful_data_bytes_read  = 0;
      read_file_begin_entry_off      = 0;

      /*|***read_file_begin_entry_off***|***entry_gap***|***cur_file_read_entry_count***|******|*/
      while (read_file_begin_entry_off < file_entry_count) {
        if (read_file_begin_entry_off + next_entry_gap >= file_entry_count) {
          next_entry_gap = (read_file_begin_entry_off + next_entry_gap) - file_entry_count;
          break;
        }
        size_t cur_file_read_entry_count;
        if (read_file_begin_entry_off + next_entry_gap + next_continuous_entry_count >
            file_entry_count) {
          cur_file_read_entry_count = file_entry_count - read_file_begin_entry_off - next_entry_gap;
        } else {
          cur_file_read_entry_count = next_continuous_entry_count;
        }

        // read concerned vars
        size_t cur_read_entry_start = read_file_begin_entry_off + next_entry_gap;
        size_t cur_read_byte_start  = (cur_read_entry_start * entry_size) / block_size * block_size;
        size_t cur_read_byte_end = (cur_read_entry_start + cur_file_read_entry_count) * entry_size;
        size_t skip_head_size    = cur_read_entry_start * entry_size - cur_read_byte_start;
        // write concerned vars
        char* local_mem_write_entry_for_file =
          this_thread_write_ptr + total_read_entry_this_thread * memory_entry_stride;
        size_t first_mem_entry_offset = 0;

        while (cur_read_byte_start < cur_read_byte_end) {
          size_t left_size          = cur_read_byte_end - cur_read_byte_start;
          size_t left_block_count   = div_rounding_up_unsafe(left_size, block_size);
          size_t read_block_count   = std::min(left_block_count, buffer_block_count);
          size_t physical_read_size = read_block_count * block_size;
          // physical_data_bytes_read += physical_read_size;
          read_size_from_cur_file += physical_read_size;

          ssize_t pread_size = pread64(fd, block_buffer, physical_read_size, cur_read_byte_start);
          if (pread_size != physical_read_size &&
              cur_read_byte_start + pread_size != file_sizes[i]) {
            WHOLEGRAPH_FAIL_NOTHROW(
              "rank=%d, pread_size=%ld, physical_read_size=%ld, file_block_read_offset=%ld, "
              "file_sizes[i]=%ld, file=%s",
              wg_rank,
              pread_size,
              physical_read_size,
              cur_read_byte_start,
              file_sizes[i],
              file_names[i]);
          }
          physical_read_size    = pread_size;
          size_t drop_tail_size = 0;
          if (cur_read_byte_start + physical_read_size > cur_read_byte_end) {
            drop_tail_size = cur_read_byte_start + physical_read_size - cur_read_byte_end;
          }

          char* useful_data_ptr   = block_buffer + skip_head_size;
          size_t useful_data_size = physical_read_size - skip_head_size - drop_tail_size;
          useful_data_bytes_read += useful_data_size;

          if (first_mem_entry_offset != 0) {
            size_t entry_left_size = entry_size - first_mem_entry_offset;
            WG_CUDA_CHECK_NO_THROW(
              cudaMemcpy(local_mem_write_entry_for_file + first_mem_entry_offset,
                         useful_data_ptr,
                         entry_left_size,
                         cudaMemcpyDefault));
            local_mem_write_entry_for_file += memory_entry_stride;
            useful_data_ptr += entry_left_size;
            useful_data_size -= entry_left_size;
            entry_left_size = 0;
          }

          size_t full_entry_count = useful_data_size / entry_size;
          size_t full_entry_size  = full_entry_count * entry_size;

          if (full_entry_size > 0) {
            if (entry_size != memory_entry_stride) {
              WG_CUDA_CHECK(cudaMemcpy2D(local_mem_write_entry_for_file,
                                         memory_entry_stride,
                                         useful_data_ptr,
                                         entry_size,
                                         entry_size,
                                         full_entry_count,
                                         cudaMemcpyDefault));
            } else {
              WG_CUDA_CHECK(cudaMemcpy(local_mem_write_entry_for_file,
                                       useful_data_ptr,
                                       full_entry_size,
                                       cudaMemcpyDefault));
            }
            local_mem_write_entry_for_file += memory_entry_stride * full_entry_count;
            useful_data_ptr += full_entry_size;
            useful_data_size -= full_entry_size;
          }

          size_t tail_entry_size = useful_data_size % entry_size;
          first_mem_entry_offset = tail_entry_size;
          if (tail_entry_size != 0) {
            // process tail
            WG_CUDA_CHECK_NO_THROW(cudaMemcpy(
              local_mem_write_entry_for_file, useful_data_ptr, tail_entry_size, cudaMemcpyDefault));
          }

          cur_read_byte_start += physical_read_size;
          skip_head_size = 0;
        }

        total_read_entry_this_thread += cur_file_read_entry_count;
        // read_size_from_cur_file += cur_file_read_entry_count * entry_size;
        if (read_file_begin_entry_off + next_entry_gap + next_continuous_entry_count >
            file_entry_count) {
          read_file_begin_entry_off = file_entry_count;
          next_continuous_entry_count -= cur_file_read_entry_count;
          next_entry_gap = 0;
        } else {
          read_file_begin_entry_off += cur_file_read_entry_count + next_entry_gap;
          next_continuous_entry_count =
            round_robin_size > local_entry_count_this_thread - total_read_entry_this_thread
              ? local_entry_count_this_thread - total_read_entry_this_thread
              : round_robin_size;
          next_entry_gap = (wg_world_size - 1) * round_robin_size;
        }
        if (total_read_entry_this_thread > local_entry_count_this_thread) {
          WHOLEGRAPH_ERROR(
            "file read error from rank %d, thread_id=%d should read %lu entries, infact %lu "
            "entries.",
            wg_rank,
            thread_id,
            local_entry_count_this_thread,
            total_read_entry_this_thread);
          break;
        } else if (total_read_entry_this_thread == local_entry_count_this_thread) {
          break;
        }
      }
      close(fd);
      WHOLEGRAPH_INFO(
        "Rank=%d  thread_id=%d done Reading useful %ld bytes by totally reading %ld bytes from "
        "file %s size=%ld "
        "using direct IO",
        wg_rank,
        thread_id,
        useful_data_bytes_read,
        read_size_from_cur_file,
        file_names[i],
        file_sizes[i]);
      if (total_read_entry_this_thread == local_entry_count_this_thread) break;
    }
    total_read_entry.fetch_add(total_read_entry_this_thread);
  };

  WHOLEGRAPH_INFO("Rank=%d use %d threads to read file.", wg_rank, threads_per_rank);

  if (threads_per_rank > 1) {
    std::vector<std::thread> read_file_threads;
    read_file_threads.reserve(threads_per_rank);
    for (int i = 0; i < threads_per_rank; i++) {
      read_file_threads.emplace_back(read_file_thread_fun, i, threads_per_rank);
    }

    for (auto&& thread : read_file_threads) {
      thread.join();
    }
  } else {
    read_file_thread_fun(0, 1);
  }

  WHOLEGRAPH_INFO("Rank=%d done Reading %ld entries, infact read %ld entries",
                   wg_rank,
                   total_read_entry.load(),
                   local_entry_count);
}

wholegraph_error_code_t load_file_to_handle(wholegraph_handle_t wholegraph_handle,
                                             size_t memory_offset,
                                             size_t memory_entry_stride,
                                             size_t entry_size,
                                             const char** file_names,
                                             int file_count,
                                             int round_robin_size) noexcept
{
  if (entry_size <= 0 || memory_offset < 0 || memory_offset + entry_size > memory_entry_stride) {
    WHOLEGRAPH_ERROR("Invalid input, entry_size=%ld, memory_entry_stride=%ld, memory_offset=%ld",
                      entry_size,
                      memory_entry_stride,
                      memory_offset);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  size_t wg_data_granularity = wholegraph_get_data_granularity(wholegraph_handle);
  if (wg_data_granularity % memory_entry_stride != 0) {
    WHOLEGRAPH_ERROR("Invalid input, memory_entry_stride=%ld, but wg_data_granularity=%ld",
                      memory_entry_stride,
                      wg_data_granularity);
    return WHOLEGRAPH_INVALID_INPUT;
  }

  size_t wg_total_size = wholegraph_get_total_size(wholegraph_handle);
  size_t expected_file_size =
    get_handle_partial_size(wg_total_size, memory_offset, memory_entry_stride, entry_size);

  if (file_count < 0 || file_count >= 65536) {
    WHOLEGRAPH_ERROR("input file count=%d", file_count);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  std::vector<size_t> file_sizes(file_count, 0);

  size_t file_total_size = 0;

  for (int i = 0; i < file_count; i++) {
    if (file_names[i] == nullptr) {
      WHOLEGRAPH_ERROR("input file %d of %d is nullptr.", i, file_count);
      return WHOLEGRAPH_INVALID_INPUT;
    }
    if (!IsFileExist(file_names[i], R_OK)) {
      WHOLEGRAPH_ERROR(
        "input_file[%d] of %d (%s) cannot open for read.", i, file_count, file_names[i]);
      return WHOLEGRAPH_INVALID_INPUT;
    }
    file_sizes[i] = StatFileSize(file_names[i]);
    if (file_sizes[i] == static_cast<size_t>(-1)) {
      WHOLEGRAPH_ERROR(
        "input_file[%d] of %d (%s) stat size failed.", i, file_count, file_names[i]);
      return WHOLEGRAPH_INVALID_INPUT;
    }
    if (file_sizes[i] % entry_size != 0) {
      WHOLEGRAPH_ERROR("input_file[%d] of %d (%s) size=%ld, but entry_size=%ld failed.",
                        i,
                        file_count,
                        file_names[i],
                        file_sizes[i],
                        entry_size);
      return WHOLEGRAPH_INVALID_INPUT;
    }
    file_total_size += file_sizes[i];
  }

  if (file_total_size > expected_file_size) {
    WHOLEGRAPH_ERROR("all %d input file size is %ld, but expected %ld",
                      file_count,
                      file_total_size,
                      expected_file_size);
    return WHOLEGRAPH_INVALID_VALUE;
  }

  try {
    wholegraph_comm_t wg_comm;
    WHOLEGRAPH_CHECK(wholegraph_get_communicator(&wg_comm, wholegraph_handle) ==
                      WHOLEGRAPH_SUCCESS);

    int wg_rank, wg_world_size;
    WHOLEGRAPH_CHECK(wholegraph_communicator_get_rank(&wg_rank, wg_comm) == WHOLEGRAPH_SUCCESS);
    WHOLEGRAPH_CHECK(wholegraph_communicator_get_size(&wg_world_size, wg_comm) ==
                      WHOLEGRAPH_SUCCESS);
    WG_COMM_CHECK_ALL_SAME(wg_comm, file_count);

    for (int i = 0; i < file_count; i++) {
      WG_COMM_CHECK_ALL_SAME(wg_comm, file_sizes[i]);
    }

    char* local_ptr = nullptr;
    size_t local_size, local_offset;

    WHOLEGRAPH_CHECK(wholegraph_get_local_memory(
                        (void**)(&local_ptr), &local_size, &local_offset, wholegraph_handle) ==
                      WHOLEGRAPH_SUCCESS);

    int suggested_buffer_size_mb    = 16;
    const char* buffer_size_env_var = std::getenv("WG_LOAD_BUFFER_SIZE_MB");
    if (buffer_size_env_var != nullptr) {
      try {
        suggested_buffer_size_mb = std::stoi(buffer_size_env_var);
      } catch (const std::invalid_argument& e) {
        suggested_buffer_size_mb = 16;
        WHOLEGRAPH_WARN(
          "Environment variable WG_LOAD_BUFFER_SIZE_MB value %s is not valid, using default %d",
          buffer_size_env_var,
          suggested_buffer_size_mb);
      }
      if (suggested_buffer_size_mb < 1) {
        suggested_buffer_size_mb = 16;
        WHOLEGRAPH_WARN(
          "Environment variable WG_LOAD_BUFFER_SIZE_MB value %s is not valid, using default %d",
          buffer_size_env_var,
          suggested_buffer_size_mb);
      }
    }
    size_t suggested_buffer_size = static_cast<size_t>(suggested_buffer_size_mb) * 1024 * 1024;

    const char* directio_env_var = std::getenv("WG_LOAD_USE_DIRECTIO");
    bool use_direct_io           = false;
    if (directio_env_var != nullptr && directio_env_var[0] == '1' && directio_env_var[1] == '\0') {
      use_direct_io = true;
    }
    if (!use_direct_io) {
      if (round_robin_size == 0) {
        read_file_list_to_local_memory_with_multi_threads(local_ptr,
                                                          local_size,
                                                          local_offset,
                                                          entry_size,
                                                          memory_entry_stride,
                                                          memory_offset,
                                                          file_count,
                                                          file_names,
                                                          file_sizes,
                                                          suggested_buffer_size,
                                                          wg_rank,
                                                          wg_world_size,
                                                          wg_comm->dev_id);
      } else {
        read_file_list_to_local_memory_roundrobin_with_multi_threads(local_ptr,
                                                                     local_size,
                                                                     local_offset,
                                                                     entry_size,
                                                                     memory_entry_stride,
                                                                     memory_offset,
                                                                     file_count,
                                                                     file_names,
                                                                     file_sizes,
                                                                     suggested_buffer_size,
                                                                     wg_rank,
                                                                     wg_world_size,
                                                                     round_robin_size,
                                                                     wg_comm->dev_id);
      }
    } else {
      if (round_robin_size == 0) {
        read_file_list_to_local_memory_directio_with_multi_thread(local_ptr,
                                                                  local_size,
                                                                  local_offset,
                                                                  entry_size,
                                                                  memory_entry_stride,
                                                                  memory_offset,
                                                                  file_count,
                                                                  file_names,
                                                                  file_sizes,
                                                                  suggested_buffer_size,
                                                                  wg_rank,
                                                                  wg_world_size,
                                                                  wg_comm->dev_id);
      } else {
        read_file_list_to_local_memory_roundrobin_directio_with_multi_threads(local_ptr,
                                                                              local_size,
                                                                              local_offset,
                                                                              entry_size,
                                                                              memory_entry_stride,
                                                                              memory_offset,
                                                                              file_count,
                                                                              file_names,
                                                                              file_sizes,
                                                                              suggested_buffer_size,
                                                                              wg_rank,
                                                                              wg_world_size,
                                                                              round_robin_size,
                                                                              wg_comm->dev_id);
      }
    }

    wg_comm->barrier();
  } catch (wholegraph::logic_error& wle) {
    WHOLEGRAPH_ERROR("Logic error: %s", wle.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (wholegraph::cuda_error& wce) {
    WHOLEGRAPH_ERROR("CUDA error: %s", wce.what());
    return WHOLEGRAPH_CUDA_ERROR;
  } catch (...) {
    WHOLEGRAPH_ERROR("Unknow error caught at file %s, line %d", __FILE__, __LINE__);
    return WHOLEGRAPH_UNKNOW_ERROR;
  }

  return WHOLEGRAPH_SUCCESS;
}

wholegraph_error_code_t store_handle_to_file(wholegraph_handle_t wholegraph_handle,
                                              size_t memory_offset,
                                              size_t memory_entry_stride,
                                              size_t entry_size,
                                              const char* local_file_name) noexcept
{
  if (entry_size <= 0 || memory_offset < 0 || memory_offset + entry_size > memory_entry_stride) {
    WHOLEGRAPH_ERROR("Invalid input, entry_size=%ld, memory_entry_stride=%ld, memory_offset=%ld",
                      entry_size,
                      memory_entry_stride,
                      memory_offset);
    return WHOLEGRAPH_INVALID_INPUT;
  }
  size_t wg_data_granularity = wholegraph_get_data_granularity(wholegraph_handle);
  if (wg_data_granularity % memory_entry_stride != 0) {
    WHOLEGRAPH_ERROR("Invalid input, memory_entry_stride=%ld, but wg_data_granularity=%ld",
                      memory_entry_stride,
                      wg_data_granularity);
    return WHOLEGRAPH_INVALID_INPUT;
  }

  try {
    wholegraph_comm_t wg_comm;
    WHOLEGRAPH_CHECK(wholegraph_get_communicator(&wg_comm, wholegraph_handle) ==
                      WHOLEGRAPH_SUCCESS);

    int wg_rank;
    WHOLEGRAPH_CHECK(wholegraph_communicator_get_rank(&wg_rank, wg_comm) == WHOLEGRAPH_SUCCESS);

    char* local_ptr = nullptr;
    size_t local_size, local_offset;

    wg_comm->barrier();

    WHOLEGRAPH_CHECK(wholegraph_get_local_memory(
                        (void**)(&local_ptr), &local_size, &local_offset, wholegraph_handle) ==
                      WHOLEGRAPH_SUCCESS);

    constexpr int kSuggestedBufferSize = 16 * 1024 * 1024;
    size_t buffer_size;
    size_t buffer_entry_count = 1;
    if (kSuggestedBufferSize < entry_size) {
      buffer_size = entry_size;
    } else {
      buffer_entry_count = kSuggestedBufferSize / entry_size;
      buffer_size        = buffer_entry_count * entry_size;
    }
    std::vector<char> file_write_buffer(buffer_size);

    size_t local_entry_count = local_size / memory_entry_stride;
    char* local_write_ptr    = local_ptr + memory_offset % memory_entry_stride;
    if (wg_rank == 0) {
      local_entry_count -= memory_offset / memory_entry_stride;
      local_write_ptr += (memory_offset / memory_entry_stride) * memory_entry_stride;
    }

    FILE* fp = fopen(local_file_name, "wb");
    if (fp == nullptr) {
      WHOLEGRAPH_ERROR("Rank=%d, open output file %s failed.\n", wg_rank, local_file_name);
    }

    size_t left_entry_count = local_entry_count;
    while (left_entry_count > 0) {
      size_t write_entry_count = std::min(left_entry_count, buffer_entry_count);
      if (entry_size != memory_entry_stride) {
        WG_CUDA_CHECK(cudaMemcpy2D(file_write_buffer.data(),
                                   entry_size,
                                   local_write_ptr,
                                   memory_entry_stride,
                                   entry_size,
                                   write_entry_count,
                                   cudaMemcpyDefault));
      } else {
        WG_CUDA_CHECK(cudaMemcpy(file_write_buffer.data(),
                                 local_write_ptr,
                                 write_entry_count * entry_size,
                                 cudaMemcpyDefault));
      }
      local_write_ptr += write_entry_count * memory_entry_stride;
      int ret = fwrite(file_write_buffer.data(), entry_size, write_entry_count, fp);

      if (ret != write_entry_count) {
        WHOLEGRAPH_ERROR(
          "File %s line %d: writing to file %s, write_entry_count=%ld, entry_size=%ld, "
          "returned %d, error=%s\n",
          __FILE__,
          __LINE__,
          local_file_name,
          write_entry_count,
          entry_size,
          ret,
          strerror(errno));
      }

      left_entry_count -= write_entry_count;
    }

    fclose(fp);

    WHOLEGRAPH_INFO("Rank=%d done writing to file %s.", wg_rank, local_file_name);

    wg_comm->barrier();

  } catch (wholegraph::logic_error& wle) {
    WHOLEGRAPH_ERROR("Logic error: %s", wle.what());
    return WHOLEGRAPH_LOGIC_ERROR;
  } catch (wholegraph::cuda_error& wce) {
    WHOLEGRAPH_ERROR("CUDA error: %s", wce.what());
    return WHOLEGRAPH_CUDA_ERROR;
  } catch (...) {
    WHOLEGRAPH_ERROR("Unknow error caught at file %s, line %d", __FILE__, __LINE__);
    return WHOLEGRAPH_UNKNOW_ERROR;
  }

  return WHOLEGRAPH_SUCCESS;
}

}  // namespace wholegraph
