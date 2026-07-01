/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdio.h>
#include <unistd.h>

#include <wholegraph/global_reference.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief WholeGraph Error Code definition
 *
 * Defines error code of WholeGraph library.
 */
enum wholegraph_error_code_t {
  WHOLEGRAPH_SUCCESS = 0,         /*!< success */
  WHOLEGRAPH_UNKNOW_ERROR,        /*!< unknown error */
  WHOLEGRAPH_NOT_IMPLEMENTED,     /*!< method is not implemented */
  WHOLEGRAPH_LOGIC_ERROR,         /*!< logic error */
  WHOLEGRAPH_CUDA_ERROR,          /*!< CUDA error */
  WHOLEGRAPH_COMMUNICATION_ERROR, /*!< communication error */
  WHOLEGRAPH_INVALID_INPUT,       /*!< input is invalid, e.g. nullptr */
  WHOLEGRAPH_INVALID_VALUE,       /*!< input value is invalid */
  WHOLEGRAPH_OUT_OF_MEMORY,       /*!< out of memory */
  WHOLEGRAPH_NOT_SUPPORTED,       /*!< not supported */
  WHOLEGRAPH_SYSTEM_ERROR,        /*!< system error>*/
};

#define WHOLEGRAPH_RETURN_ON_FAIL(X)                                                 \
  do {                                                                                \
    auto err = X;                                                                     \
    if (err != WHOLEGRAPH_SUCCESS) {                                                 \
      const char* error_str = #X;                                                     \
      fprintf(stderr, "File %s line %d %s failed.\n", __FILE__, __LINE__, error_str); \
      return err;                                                                     \
    }                                                                                 \
  } while (0)

/**
 * @brief Memory Type of WholeGraph
 *
 * Memory Type is the Memory Address Mapping Type of WholeGraph
 */
enum wholegraph_memory_type_t {
  WHOLEGRAPH_MT_NONE = 0,    /*!< Not defined.  */
  WHOLEGRAPH_MT_CONTINUOUS,  /*!< Memory from all ranks are mapped in continuous address space */
  WHOLEGRAPH_MT_DISTRIBUTED, /*!< Memory from other ranks are not mapped. */
};

/**
 * @brief Memory Location of WholeGraph
 *
 * Memory Location of WholeGraph can be host or device.
 */
enum wholegraph_memory_location_t {
  WHOLEGRAPH_ML_NONE = 0, /*!< Not defined */
  WHOLEGRAPH_ML_DEVICE,   /*!< Device Memory */
  WHOLEGRAPH_ML_HOST,     /*!< Host Memory */
};

enum wholegraph_distributed_backend_t {
  WHOLEGRAPH_DB_NONE = 0, /*!< Not defined */
  WHOLEGRAPH_DB_NCCL,
};

enum LogLevel {
  LEVEL_FATAL = 0, /*!< Fatal */
  LEVEL_ERROR,     /*!< Error */
  LEVEL_WARN,      /*!< Warn */
  LEVEL_INFO,      /*!< Info */
  LEVEL_DEBUG,     /*!< Debug*/
  LEVEL_TRACE      /*!< Trace */
};

#define WHOLEGRAPH_SPILT_NO_COLOR -1
/**
 * Initialize WholeGraph library
 * @param flags : reserved should be 0
 * @param log_level : wholegraph log level, the default level is "info"
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_init(unsigned int flags, LogLevel log_level = LEVEL_INFO);

/**
 * Finalize WholeGraph library
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_finalize();

/**
 * @brief Opaque handle to communicator
 *
 * An Opaque handle to communicator
 */
typedef struct wholegraph_comm_* wholegraph_comm_t;

struct clique_info_t {
  int is_in_clique;  // is_in_clique >0 means the gpu belongs to  a mnnvl domain
  int clique_first_rank;
  int clique_rank;      // the rank of gpu in a mnnvl domain
  int clique_rank_num;  // the num of gpu in the mnnvl domain
  int clique_id;        // the id of clique
  int clique_num;       // the num of clique in the comm domain.
};

#define WHOLEGRAPH_UNIQUE_ID_BYTES (128)
/**
 * @brief Unique ID for WholeGraph Communicators
 *
 * An Opaque handle to WholeGraph Communicators, exposes as char array.
 * Underlying implementation may be ncclUniqueId_t
 */
struct wholegraph_unique_id_t {
  char internal[WHOLEGRAPH_UNIQUE_ID_BYTES];
};

/**
 * Create UniqueID for WholeGraph Communicator
 * @param unique_id : returned UniqueID
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_create_unique_id(wholegraph_unique_id_t* unique_id);

/**
 * Create WholeGraph Communicator
 * @param comm : returned WholeGraph Communicator
 * @param unique_id : UniqueID
 * @param rank : rank of this process.
 * @param size : number of processes in this Communicator
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_create_communicator(wholegraph_comm_t* comm,
                                                         wholegraph_unique_id_t unique_id,
                                                         int rank,
                                                         int size);

/**
 * Split WholeGraph Communicator
 * @param new_comm: returned the splited wholeMemory Communicator
 * @param comm: WholeGraph Communicator to split
 * @param color: color value to split communicator,Ranks which pass the same color value will be
 * part of the same group; color must be a non-negative value. If it is passed as
 * WHOLEGRAPH_SPLIT_NOCOLOR, it means that the rank will not be part of any group, therefore
 * returning NULL as newcomm.
 * @param key: key value to split communicator,the value of key will determine the
 * rank order, and the smaller key means the smaller rank in new communicator. If keys are equal
 * between ranks, then the rank in the original communicator will be used to order ranks.
 * @return : wholegraph_error_code_t

*/
wholegraph_error_code_t wholegraph_split_communicator(wholegraph_comm_t* new_comm,
                                                        wholegraph_comm_t comm,
                                                        int color,
                                                        int key);
/**
 * Destroy WholeGraph Communicator
 * @param comm : WholeGraph Communicator to destroy
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_destroy_communicator(wholegraph_comm_t comm);

/**
 * Check if combination of WholeGraph type and location is supported in the communicator
 * @param comm : WholeGraph Communicator
 * @param memory_type : WholeGraph type
 * @param memory_location : WholeGraph Location
 * @return WHOLEGRAPH_SUCCESS if supported else WHOLEGRAPH_NOT_SUPPORTED
 */
wholegraph_error_code_t wholegraph_communicator_support_type_location(
  wholegraph_comm_t comm,
  wholegraph_memory_type_t memory_type,
  wholegraph_memory_location_t memory_location);

/**
 * Get the rank of current process in the WholeGraph Communicator
 * @param rank : returned rank
 * @param comm : WholeGraph Communicator
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_communicator_get_rank(int* rank, wholegraph_comm_t comm);

/**
 * Get the size of WholeGraph Communicator
 * @param size : returned size
 * @param comm : WholeGraph Communicator
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_communicator_get_size(int* size, wholegraph_comm_t comm);

/**
 * Get the local rank size of current process in the WholeGraph Communicator
 * @param local_size : returned local rank size
 * @param comm : WholeGraph Communicator
 * @return : wholegraph_error_code_t
 */

wholegraph_error_code_t wholegraph_communicator_get_local_size(int* local_size,
                                                                 wholegraph_comm_t comm);

/**
 * Get the clique info of WholeGraph Communicator
 * @param clique_info : returned clique info
 * @param comm : WholeGraph Communicator
 * @return : wholegraph_error_code_t
 */

wholegraph_error_code_t wholegraph_communicator_get_clique_info(clique_info_t* clique_info,
                                                                  wholegraph_comm_t comm);

wholegraph_error_code_t wholegraph_communicator_set_distributed_backend(
  wholegraph_comm_t comm, wholegraph_distributed_backend_t distributed_backend);

wholegraph_distributed_backend_t wholegraph_communicator_get_distributed_backend(
  wholegraph_comm_t comm);
/**
 * Barrier on WholeGraph Communicator
 * @param comm : WholeGraph Communicator
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_communicator_barrier(wholegraph_comm_t comm);

/**
 * @brief Opaque handle to WholeGraph
 *
 * An Opaque handle to WholeGraph
 */
typedef struct wholegraph_handle_* wholegraph_handle_t;

/**
 * Malloc WholeGraph
 * @param wholegraph_handle_ptr : returned WholeGraph Handle
 * @param total_size : total allocated size in bytes.
 * @param comm : WholeGraph Communicator
 * @param memory_type : WholeGraph type
 * @param memory_location : memory location, host or device
 * @param data_granularity : granularity size of data, which is guaranteed not to be partitioned.
 * @param rank_entry_partition : entry count of each rank (size of entry equal to data_granularity)
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_malloc(wholegraph_handle_t* wholegraph_handle_ptr,
                                            size_t total_size,
                                            wholegraph_comm_t comm,
                                            wholegraph_memory_type_t memory_type,
                                            wholegraph_memory_location_t memory_location,
                                            size_t data_granularity,
                                            size_t* rank_entry_partition = nullptr);

/**
 * Free allocated WholeGraph Handle
 * @param wholegraph_handle : WholeGraph Handle to free
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_free(wholegraph_handle_t wholegraph_handle);

/**
 * Get underlying WholeGraph Communicator from WholeGraph Handle
 * @param comm : returned WholeGraph Communicator
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_communicator(wholegraph_comm_t* comm,
                                                      wholegraph_handle_t wholegraph_handle);

/**
 * Get WholeGraph Type
 * @param wholegraph_handle : WholeGraph Handle
 * @return : WholeGraph Type
 */
wholegraph_memory_type_t wholegraph_get_memory_type(wholegraph_handle_t wholegraph_handle);

/**
 * Get WholeGraph Location
 * @param wholegraph_handle : WholeGraph Handle
 * @return : WholeGraph Location
 */
wholegraph_memory_location_t wholegraph_get_memory_location(
  wholegraph_handle_t wholegraph_handle);

wholegraph_distributed_backend_t wholegraph_get_distributed_backend(
  wholegraph_handle_t wholegraph_handle);

/**
 * Get total size of WholeGraph
 * @param wholegraph_handle : WholeGraph Handle
 * @return : total size
 */
size_t wholegraph_get_total_size(wholegraph_handle_t wholegraph_handle);

/**
 * Get data granularity of WholeGraph Handle
 * @param wholegraph_handle : WholeGraph Handle
 * @return : data granularity size
 */
size_t wholegraph_get_data_granularity(wholegraph_handle_t wholegraph_handle);

/**
 * Get local memory from WholeGraph Handle of current rank, local memory has direct access to the
 * memory. But local memory doesn't have to be on local GPU.
 * @param local_ptr : returned local memory pointer
 * @param local_size : returned local memory size
 * @param local_offset : returned local memory offset from WholeGraph
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_local_memory(void** local_ptr,
                                                      size_t* local_size,
                                                      size_t* local_offset,
                                                      wholegraph_handle_t wholegraph_handle);

/**
 * Get local memory size from WholeGraph Handle of current rank
 * @param local_size : returned local memory size
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_local_size(size_t* local_size,
                                                    wholegraph_handle_t wholegraph_handle);

/**
 * Get local memory offset from WholeGraph Handle of current rank
 * @param local_offset : returned local memory offset
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_local_offset(size_t* local_offset,
                                                      wholegraph_handle_t wholegraph_handle);

/**
 * Get local memory of specified rank from WholeGraph Handle
 * @param rank_memory_ptr : returned local memory pointer of specified rank
 * @param rank_memory_size : returned local memory size of specified rank
 * @param rank_memory_offset : returned local memory offset of specified rank from WholeGraph
 * @param rank : rank specified
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_rank_memory(void** rank_memory_ptr,
                                                     size_t* rank_memory_size,
                                                     size_t* rank_memory_offset,
                                                     int rank,
                                                     wholegraph_handle_t wholegraph_handle);

/**
 * Get the equal partition plan WholeGraph uses by default
 * @param entry_per_rank : returned entry count per rank
 * @param total_entry_count : total entry count
 * @param world_size : communicator world size
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_equal_entry_partition_plan(size_t* entry_per_rank,
                                                                size_t total_entry_count,
                                                                int world_size);

/**
 * Get global memory pointer from WholeGraph Handle.
 * Only Continuous memory type has global pointer.
 * @param global_ptr : returned pointer of WholeGraph
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_global_pointer(void** global_ptr,
                                                        wholegraph_handle_t wholegraph_handle);

/**
 * Get global reference from WholeGraph Handle
 * WholeGraph global reference is common data structure for mapped Memory Types.
 * @param wholegraph_gref : returned WholeGraph global reference
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_global_reference(wholegraph_gref_t* wholegraph_gref,
                                                          wholegraph_handle_t wholegraph_handle);

/**
 * Get memory size of each rank from WholeGraph Handle
 * @param rank_mem_sizes : returned memory size of each rank
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_rank_partition_sizes(
  size_t* rank_mem_sizes, wholegraph_handle_t wholegraph_handle);

/**
 * Get memory offset of each rank from WholeGraph Handle
 * @param rank_mem_offsets : returned memory offset of each rank
 * @param wholegraph_handle : WholeGraph Handle
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_get_rank_partition_offsets(
  size_t* rank_mem_offsets, wholegraph_handle_t wholegraph_handle);

/**
 * Fork a new process and get device count. Should be called before other CUDA call
 * @return : CUDA device count, -1 on error
 */
int fork_get_device_count();

/**
 * Load WholeGraph from binary files, all rank should be called together
 * @param wholegraph_handle : WholeGraph Handle
 * @param memory_offset : load to memory offset
 * @param memory_entry_size : entry size of WholeGraph
 * @param file_entry_size : entry size in file, should be less than or equal to memory_entry_size
 * @param file_names : file names, all binary files will be logically concatenated and loaded.
 * @param file_count : number of files.
 * @param round_robin_size : continuous embedding number for a rank under round-robin shard mode
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_load_from_file(wholegraph_handle_t wholegraph_handle,
                                                    size_t memory_offset,
                                                    size_t memory_entry_size,
                                                    size_t file_entry_size,
                                                    const char** file_names,
                                                    int file_count,
                                                    int round_robin_size);

/**
 * Store local WholeGraph to file, this should be called by all ranks, with different
 * local_file_name.
 * @param wholegraph_handle : WholeGraph Handle
 * @param memory_offset : memory offset to store
 * @param memory_entry_stride : entry size of WholeGraph
 * @param file_entry_size : entry size in file, should be less than or equal to memory_entry_size
 * @param local_file_name : local file to store to
 * @return : wholegraph_error_code_t
 */
wholegraph_error_code_t wholegraph_store_to_file(wholegraph_handle_t wholegraph_handle,
                                                   size_t memory_offset,
                                                   size_t memory_entry_stride,
                                                   size_t file_entry_size,
                                                   const char* local_file_name);

/**
 * @param comm : WholeGraph Comm
 * @return : bool
 */
bool wholegraph_is_intranode_communicator(wholegraph_comm_t comm);

bool wholegraph_is_intra_mnnvl_communicator(wholegraph_comm_t comm);

#ifdef __cplusplus
}
#endif
