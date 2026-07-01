/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "wholegraph_binding.hpp"

using namespace nb::literals;

void register_wholegraph_binding(nb::module_& m)
{
  nb::enum_<wholegraph_error_code_t>(m, "WholeGraphErrorCode", nb::is_arithmetic())
    .value("Success", WHOLEGRAPH_SUCCESS)
    .value("UnknowError", WHOLEGRAPH_UNKNOW_ERROR)
    .value("NotImplemented", WHOLEGRAPH_NOT_IMPLEMENTED)
    .value("LogicError", WHOLEGRAPH_LOGIC_ERROR)
    .value("CUDAError", WHOLEGRAPH_CUDA_ERROR)
    .value("CommunicationError", WHOLEGRAPH_COMMUNICATION_ERROR)
    .value("InvalidInput", WHOLEGRAPH_INVALID_INPUT)
    .value("InvalidValue", WHOLEGRAPH_INVALID_VALUE)
    .value("OutOfMemory", WHOLEGRAPH_OUT_OF_MEMORY)
    .value("NotSupported", WHOLEGRAPH_NOT_SUPPORTED)
    .export_values();

  nb::enum_<wholegraph_memory_type_t>(m, "WholeGraphMemoryType", nb::is_arithmetic())
    .value("MtNone", WHOLEGRAPH_MT_NONE)
    .value("MtContinuous", WHOLEGRAPH_MT_CONTINUOUS)
    .value("MtDistributed", WHOLEGRAPH_MT_DISTRIBUTED)
    .export_values();

  nb::enum_<wholegraph_memory_location_t>(m, "WholeGraphMemoryLocation", nb::is_arithmetic())
    .value("MlNone", WHOLEGRAPH_ML_NONE)
    .value("MlDevice", WHOLEGRAPH_ML_DEVICE)
    .value("MlHost", WHOLEGRAPH_ML_HOST)
    .export_values();

  nb::enum_<wholegraph_distributed_backend_t>(
    m, "WholeGraphDistributedBackend", nb::is_arithmetic())
    .value("DbNone", WHOLEGRAPH_DB_NONE)
    .value("DbNCCL", WHOLEGRAPH_DB_NCCL)
    .export_values();

  nb::enum_<LogLevel>(m, "WholeGraphLogLevel", nb::is_arithmetic())
    .value("LevFatal", LEVEL_FATAL)
    .value("LevError", LEVEL_ERROR)
    .value("LevWarn", LEVEL_WARN)
    .value("LevInfo", LEVEL_INFO)
    .value("LevDebug", LEVEL_DEBUG)
    .value("LevTrace", LEVEL_TRACE)
    .export_values();

  nb::enum_<WholeGraphMemoryAllocTypeNB>(
    m, "WholeGraphMemoryAllocType", nb::is_arithmetic())
    .value("MatNone", WholeGraphMemoryAllocTypeNB::MatNone)
    .value("MatDevice", WholeGraphMemoryAllocTypeNB::MatDevice)
    .value("MatHost", WholeGraphMemoryAllocTypeNB::MatHost)
    .value("MatPinned", WholeGraphMemoryAllocTypeNB::MatPinned)
    .export_values();

  nb::enum_<wholegraph_dtype_t>(m, "WholeGraphDataType", nb::is_arithmetic())
    .value("DtUnknown", WHOLEGRAPH_DT_UNKNOWN)
    .value("DtFloat", WHOLEGRAPH_DT_FLOAT)
    .value("DtHalf", WHOLEGRAPH_DT_HALF)
    .value("DtDouble", WHOLEGRAPH_DT_DOUBLE)
    .value("DtBF16", WHOLEGRAPH_DT_BF16)
    .value("DtInt", WHOLEGRAPH_DT_INT)
    .value("DtInt64", WHOLEGRAPH_DT_INT64)
    .value("DtInt16", WHOLEGRAPH_DT_INT16)
    .value("DtInt8", WHOLEGRAPH_DT_INT8)
    .value("DtCount", WHOLEGRAPH_DT_COUNT)
    .export_values();

  nb::enum_<wholegraph_access_type_t>(m, "WholeGraphAccessType", nb::is_arithmetic())
    .value("AtNone", WHOLEGRAPH_AT_NONE)
    .value("AtReadOnly", WHOLEGRAPH_AT_READONLY)
    .value("AtReadWrite", WHOLEGRAPH_AT_READWRITE)
    .export_values();

  nb::enum_<wholegraph_optimizer_type_t>(
    m, "WholeGraphOptimizerType", nb::is_arithmetic())
    .value("OptNone", WHOLEGRAPH_OPT_NONE)
    .value("OptSgd", WHOLEGRAPH_OPT_SGD)
    .value("OptLazyAdam", WHOLEGRAPH_OPT_LAZY_ADAM)
    .value("OptRmsProp", WHOLEGRAPH_OPT_RMSPROP)
    .value("OptAdaGrad", WHOLEGRAPH_OPT_ADAGRAD)
    .export_values();

  nb::enum_<WholeGraphViewTypeNB>(m, "WholeGraphViewType", nb::is_arithmetic())
    .value("VtNone", WholeGraphViewTypeNB::VtNone)
    .value("VtLocal", WholeGraphViewTypeNB::VtLocal)
    .value("VtGlobal", WholeGraphViewTypeNB::VtGlobal)
    .value("VtRemote", WholeGraphViewTypeNB::VtRemote)
    .export_values();

  nb::enum_<DLDeviceTypeNB>(m, "DLDeviceType", nb::is_arithmetic())
    .value("kDLCPU", DLDeviceTypeNB::kDLCPU)
    .value("kDLCUDA", DLDeviceTypeNB::kDLCUDA)
    .value("kDLCUDAHost", DLDeviceTypeNB::kDLCUDAHost)
    .export_values();

  nb::class_<PyMemoryAllocTypeNB>(m, "PyMemoryAllocType")
    .def(nb::init<>())
    .def("set_type", &PyMemoryAllocTypeNB::set_type, "new_type"_a)
    .def("get_type", &PyMemoryAllocTypeNB::get_type)
    .def("set_ctype", &PyMemoryAllocTypeNB::set_ctype, "alloc_type"_a)
    .def("get_ctype", &PyMemoryAllocTypeNB::get_ctype);

  nb::class_<GlobalContextWrapperNB>(m, "GlobalContextWrapper")
    .def(nb::init<>())
    .def("create_context",
         &GlobalContextWrapperNB::create_context,
         "temp_create_context_fn"_a,
         "temp_destroy_context_fn"_a,
         "temp_malloc_fn"_a,
         "temp_free_fn"_a,
         "temp_global_context"_a.none(),
         "output_malloc_fn"_a,
         "output_free_fn"_a,
         "output_global_context"_a.none())
    .def("get_env_fns", &GlobalContextWrapperNB::get_env_fns);

  nb::class_<PyWholeGraphUniqueIDNB>(m, "PyWholeGraphUniqueID")
    .def(nb::init<>())
    .def("__len__", &PyWholeGraphUniqueIDNB::length)
    .def("__dlpack__", &PyWholeGraphUniqueIDNB::dlpack, "stream"_a = nb::none())
    .def("__dlpack_device__", &PyWholeGraphUniqueIDNB::dlpack_device);

  nb::class_<PyWholeGraphCommNB>(m, "PyWholeGraphComm")
    .def(nb::init<>())
    .def("get_c_handle", &PyWholeGraphCommNB::get_c_handle)
    .def("support_type_location",
         &PyWholeGraphCommNB::support_type_location,
         "memory_type"_a,
         "memory_location"_a)
    .def("get_rank", &PyWholeGraphCommNB::get_rank)
    .def("get_size", &PyWholeGraphCommNB::get_size)
    .def("get_clique_info", &PyWholeGraphCommNB::get_clique_info)
    .def("barrier", &PyWholeGraphCommNB::barrier)
    .def("get_distributed_backend", &PyWholeGraphCommNB::get_distributed_backend)
    .def("set_distributed_backend",
         &PyWholeGraphCommNB::set_distributed_backend,
         "distributed_backend"_a);

  nb::class_<PyWholeGraphHandleNB>(m, "PyWholeGraphHandle")
    .def(nb::init<>())
    .def("get_c_handle", &PyWholeGraphHandleNB::get_c_handle)
    .def("get_communicator", &PyWholeGraphHandleNB::get_communicator)
    .def("get_memory_type", &PyWholeGraphHandleNB::get_memory_type)
    .def("get_memory_location", &PyWholeGraphHandleNB::get_memory_location)
    .def("get_global_flatten_tensor",
         &PyWholeGraphHandleNB::get_global_flatten_tensor,
         "import_dlpack_fn"_a,
         "data_type"_a,
         "view_from_device"_a,
         "view_from_device_id"_a)
    .def("get_local_flatten_tensor",
         &PyWholeGraphHandleNB::get_local_flatten_tensor,
         "import_dlpack_fn"_a,
         "data_type"_a,
         "view_from_device"_a,
         "view_from_device_id"_a)
    .def("from_filelist",
         &PyWholeGraphHandleNB::from_filelist,
         "memory_offset"_a,
         "memory_entry_size"_a,
         "file_entry_size"_a,
         "round_robin_size"_a,
         "file_list"_a)
    .def("to_file",
         &PyWholeGraphHandleNB::to_file,
         "memory_offset"_a,
         "memory_entry_size"_a,
         "file_entry_size"_a,
         "file_name"_a);

  nb::class_<PyWholeGraphTensorDescriptionNB>(m, "PyWholeGraphTensorDescription")
    .def(nb::init<>())
    .def("set_dtype", &PyWholeGraphTensorDescriptionNB::set_dtype, "dtype"_a)
    .def("set_shape", &PyWholeGraphTensorDescriptionNB::set_shape, "shape"_a)
    .def("set_stride", &PyWholeGraphTensorDescriptionNB::set_stride, "strides"_a)
    .def("set_storage_offset",
         &PyWholeGraphTensorDescriptionNB::set_storage_offset,
         "storage_offset"_a)
    .def_prop_ro("dtype", &PyWholeGraphTensorDescriptionNB::dtype)
    .def("dim", &PyWholeGraphTensorDescriptionNB::dim)
    .def_prop_ro("shape", &PyWholeGraphTensorDescriptionNB::shape)
    .def("stride", &PyWholeGraphTensorDescriptionNB::stride)
    .def("storage_offset", &PyWholeGraphTensorDescriptionNB::storage_offset);

  nb::class_<WrappedLocalTensorNB>(m, "WrappedLocalTensor")
    .def(nb::init<>())
    .def("wrap_tensor",
         &WrappedLocalTensorNB::wrap_tensor,
         "py_desc"_a,
         "data_ptr"_a,
         nb::rv_policy::reference_internal)
    .def("get_c_handle", &WrappedLocalTensorNB::get_c_handle);

  nb::class_<PyWholeGraphFlattenDlpackNB>(m, "PyWholeGraphFlattenDlpack")
    .def(nb::init<>())
    .def("__len__", &PyWholeGraphFlattenDlpackNB::length)
    .def("set_view_device",
         &PyWholeGraphFlattenDlpackNB::set_view_device,
         "device_type"_a,
         "device_id"_a)
    .def("get_view",
         &PyWholeGraphFlattenDlpackNB::get_view,
         "handle"_a,
         "data_type"_a,
         "view_type"_a,
         "target_rank"_a)
    .def_prop_ro("ptr", &PyWholeGraphFlattenDlpackNB::ptr)
    .def_prop_ro("__cuda_array_interface__",
                 &PyWholeGraphFlattenDlpackNB::cuda_array_interface)
    .def("__dlpack__", &PyWholeGraphFlattenDlpackNB::dlpack, "stream"_a = nb::none())
    .def("__dlpack_device__", &PyWholeGraphFlattenDlpackNB::dlpack_device);

  nb::class_<PyWholeGraphTensorNB>(m, "PyWholeGraphTensor")
    .def(nb::init<>())
    .def("get_c_handle", &PyWholeGraphTensorNB::get_c_handle)
    .def("get_wholegraph_handle", &PyWholeGraphTensorNB::get_wholegraph_handle)
    .def_prop_ro("dtype", &PyWholeGraphTensorNB::dtype)
    .def("dim", &PyWholeGraphTensorNB::dim)
    .def_prop_ro("shape", &PyWholeGraphTensorNB::shape)
    .def("stride", &PyWholeGraphTensorNB::stride)
    .def("storage_offset", &PyWholeGraphTensorNB::storage_offset)
    .def("get_local_entry_count", &PyWholeGraphTensorNB::get_local_entry_count)
    .def("get_local_entry_start", &PyWholeGraphTensorNB::get_local_entry_start)
    .def("get_sub_tensor", &PyWholeGraphTensorNB::get_sub_tensor, "starts"_a, "ends"_a)
    .def("get_tensor_in_window",
         &PyWholeGraphTensorNB::get_tensor_in_window,
         "flatten_tensor"_a,
         "storage_window_offset"_a)
    .def("get_local_tensor",
         &PyWholeGraphTensorNB::get_local_tensor,
         "import_dlpack_fn"_a,
         "view_from_device"_a,
         "view_from_device_id"_a)
    .def("get_global_tensor",
         &PyWholeGraphTensorNB::get_global_tensor,
         "import_dlpack_fn"_a,
         "view_from_device"_a,
         "view_from_device_id"_a)
    .def("from_filelist",
         &PyWholeGraphTensorNB::from_filelist,
         "file_list"_a,
         "round_robin_size"_a = 0)
    .def("to_file", &PyWholeGraphTensorNB::to_file, "filename"_a);

  nb::class_<WholeGraphCachePolicyNB>(m, "WholeGraphCachePolicy")
    .def(nb::init<>())
    .def("create_policy",
         &WholeGraphCachePolicyNB::create_policy,
         "cache_comm"_a,
         "memory_type"_a,
         "memory_location"_a,
         "access_type"_a,
         "ratio"_a)
    .def("destroy_policy", &WholeGraphCachePolicyNB::destroy_policy)
    .def("get_c_handle",
         [](WholeGraphCachePolicyNB const& policy) {
           return reinterpret_cast<int64_t>(policy.c_handle());
         });

  nb::class_<WholeGraphOptimizerNB>(m, "WholeGraphOptimizer")
    .def(nb::init<>())
    .def("create_optimizer",
         &WholeGraphOptimizerNB::create_optimizer,
         "optimizer_type"_a,
         "param_dict"_a)
    .def("add_embedding", &WholeGraphOptimizerNB::add_embedding, "embedding"_a)
    .def("destroy_optimizer", &WholeGraphOptimizerNB::destroy_optimizer)
    .def("get_c_handle",
         [](WholeGraphOptimizerNB const& optimizer) {
           return reinterpret_cast<int64_t>(optimizer.c_handle());
         });

  nb::class_<PyWholeGraphEmbeddingNB>(m, "PyWholeGraphEmbedding")
    .def(nb::init<>())
    .def("get_c_handle", &PyWholeGraphEmbeddingNB::get_c_handle)
    .def("get_embedding_tensor", &PyWholeGraphEmbeddingNB::get_embedding_tensor)
    .def("get_optimizer_state_names", &PyWholeGraphEmbeddingNB::get_optimizer_state_names)
    .def("get_optimizer_state", &PyWholeGraphEmbeddingNB::get_optimizer_state, "name"_a)
    .def("writeback_all_cache", &PyWholeGraphEmbeddingNB::writeback_all_cache, "stream_int"_a)
    .def("drop_all_cache", &PyWholeGraphEmbeddingNB::drop_all_cache, "stream_int"_a)
    .def("destroy_embedding", &PyWholeGraphEmbeddingNB::destroy_embedding);

  m.def("init", &init, "flags"_a, "log_level"_a = LEVEL_INFO);
  m.def("finalize", &finalize);
  m.def("create_unique_id", &create_unique_id);
  m.def("get_type_string", &get_type_string, "data_type"_a);
  m.def("create_communicator",
        &create_communicator,
        "py_uid"_a,
        "world_rank"_a,
        "world_size"_a);
  m.def("destroy_communicator", &destroy_communicator, "py_comm"_a);
  m.def("split_communicator", &split_communicator, "comm"_a, "color"_a, "key"_a);
  m.def("communicator_set_distributed_backend",
        &communicator_set_distributed_backend,
        "py_comm"_a,
        "distributed_backend"_a);
  m.def("equal_partition_plan",
        &equal_partition_plan,
        "entry_count"_a,
        "world_size"_a);
  m.def("malloc",
        &malloc_wholegraph,
        "total_size"_a,
        "py_comm"_a,
        "memory_type"_a,
        "memory_location"_a,
        "data_granularity"_a,
        "rank_entry_partition"_a = nb::none());
  m.def("free", &free_handle, "handle"_a);
  m.def("create_wholegraph_array",
        &create_wholegraph_array,
        "dtype"_a,
        "size"_a,
        "comm"_a,
        "mem_type"_a,
        "mem_location"_a,
        "tensor_entry_partition"_a = nb::none());
  m.def("create_wholegraph_matrix",
        &create_wholegraph_matrix,
        "dtype"_a,
        "row"_a,
        "column"_a,
        "stride"_a,
        "comm"_a,
        "mem_type"_a,
        "mem_location"_a,
        "tensor_entry_partition"_a = nb::none());
  m.def("create_wholegraph_tensor",
        &create_wholegraph_tensor,
        "tensor_description"_a,
        "comm"_a,
        "mem_type"_a,
        "mem_location"_a,
        "tensor_entry_partition"_a = nb::none());
  m.def("make_tensor_as_wholegraph",
        &make_tensor_as_wholegraph,
        "tensor_description"_a,
        "data_ptr"_a);
  m.def("make_handle_as_wholegraph",
        &make_handle_as_wholegraph,
        "tensor_description"_a,
        "handle"_a);
  m.def("destroy_wholegraph_tensor", &destroy_wholegraph_tensor, "wholegraph_tensor"_a);
  m.def("py_get_wholegraph_tensor_count", &get_wholegraph_tensor_count);
  m.def("create_non_cache_policy", &create_non_cache_policy);
  m.def("create_embedding",
        &create_embedding,
        "tensor_description"_a,
        "comm"_a,
        "memory_type"_a,
        "memory_location"_a,
        "cache_policy"_a,
        "embedding_entry_partition"_a = nb::none(),
        "user_defined_sms"_a = -1,
        "round_robin_size"_a = 0);
  m.def("wholegraph_env_test_op",
        &py_wholegraph_env_test_op,
        "input_tensor"_a,
        "output_fixed_tensor"_a,
        "output_variable_device_tensor_handle"_a,
        "output_variable_pinned_tensor_handle"_a,
        "output_variable_host_tensor_handle"_a,
        "output_variable_entry_count"_a,
        "p_env_fns_int"_a,
        "stream_int"_a);
  m.def("wholegraph_gather_op",
        &wholegraph_gather_op,
        "wholegraph_tensor"_a,
        "indices_tensor"_a,
        "output_tensor"_a,
        "p_env_fns_int"_a,
        "stream_int"_a);
  m.def("wholegraph_scatter_op",
        &wholegraph_scatter_op,
        "input_tensor"_a,
        "indices_tensor"_a,
        "wholegraph_tensor"_a,
        "p_env_fns_int"_a,
        "stream_int"_a);
  m.def("EmbeddingGatherForward",
        &embedding_gather_forward,
        "embedding"_a,
        "indices_tensor"_a,
        "output_tensor"_a,
        "adjust_cache"_a,
        "p_env_fns_int"_a,
        "stream_int"_a);
  m.def("EmbeddingGatherGradientApply",
        &embedding_gather_gradient_apply,
        "embedding"_a,
        "indices_tensor"_a,
        "grads_tensor"_a,
        "adjust_cache"_a,
        "lr"_a,
        "p_env_fns_int"_a,
        "stream_int"_a);
  m.def("csr_unweighted_sample_without_replacement",
        &csr_unweighted_sample_without_replacement,
        "wg_csr_row_ptr_tensor"_a,
        "wg_csr_col_ptr_tensor"_a,
        "center_nodes_tensor"_a,
        "max_sample_count"_a,
        "output_sample_offset_tensor"_a,
        "output_dest_memory_context"_a,
        "output_center_localid_memory_context"_a,
        "output_edge_gid_memory_context"_a,
        "random_seed"_a,
        "p_env_fns_int"_a,
        "stream_int"_a);
  m.def("csr_weighted_sample_without_replacement",
        &csr_weighted_sample_without_replacement,
        "wg_csr_row_ptr_tensor"_a,
        "wg_csr_col_ptr_tensor"_a,
        "wg_csr_weight_ptr_tensor"_a,
        "center_nodes_tensor"_a,
        "max_sample_count"_a,
        "output_sample_offset_tensor"_a,
        "output_dest_memory_context"_a,
        "output_center_localid_memory_context"_a,
        "output_edge_gid_memory_context"_a,
        "random_seed"_a,
        "p_env_fns_int"_a,
        "stream_int"_a);
  m.def("host_generate_random_positive_int",
        &host_generate_random_positive_int,
        "random_seed"_a,
        "subsequence"_a,
        "output"_a);
  m.def("host_generate_exponential_distribution_negative_float",
        &host_generate_exponential_distribution_negative_float,
        "random_seed"_a,
        "subsequence"_a,
        "output"_a);
  m.def("append_unique",
        &append_unique,
        "target_nodes_tensor"_a,
        "neighbor_nodes_tensor"_a,
        "output_unique_node_memory_context"_a,
        "output_neighbor_raw_to_unique_mapping_tensor"_a,
        "p_env_fns_int"_a,
        "stream_int"_a);
  m.def("add_csr_self_loop",
        &add_csr_self_loop,
        "csr_row_ptr_tensor"_a,
        "csr_col_ptr_tensor"_a,
        "output_csr_row_ptr_tensor"_a,
        "output_csr_col_ptr_tensor"_a,
        "stream_int"_a);
  m.def("fork_get_gpu_count", &fork_get_gpu_count);
  m.def("load_wholegraph_handle_from_filelist",
        &load_wholegraph_handle_from_filelist,
        "wholegraph_handle_int_ptr"_a,
        "memory_offset"_a,
        "memory_entry_size"_a,
        "file_entry_size"_a,
        "round_robin_size"_a,
        "file_list"_a);
  m.def("store_wholegraph_handle_to_file",
        &store_wholegraph_handle_to_file,
        "wholegraph_handle_int_ptr"_a,
        "memory_offset"_a,
        "memory_entry_size"_a,
        "file_entry_size"_a,
        "file_name"_a);
}
