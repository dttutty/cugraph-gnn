# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pylibwholegraph.binding.wholegraph_binding as wgb
from pylibwholegraph.utils.imports import import_optional
import os

torch = import_optional("torch")

WholeGraphDataType = wgb.WholeGraphDataType


def torch_dtype_to_wholegraph_dtype(torch_dtype: "torch.dtype"):
    """
    Convert torch.dtype to WholeGraphDataType
    :param torch_dtype: torch.dtype
    :return: WholeGraphDataType
    """
    if torch_dtype == torch.float:
        return WholeGraphDataType.DtFloat
    elif torch_dtype == torch.half:
        return WholeGraphDataType.DtHalf
    elif torch_dtype == torch.double:
        return WholeGraphDataType.DtDouble
    elif torch_dtype == torch.bfloat16:
        return WholeGraphDataType.DtBF16
    elif torch_dtype == torch.int:
        return WholeGraphDataType.DtInt
    elif torch_dtype == torch.int64:
        return WholeGraphDataType.DtInt64
    elif torch_dtype == torch.int16:
        return WholeGraphDataType.DtInt16
    elif torch_dtype == torch.int8:
        return WholeGraphDataType.DtInt8
    else:
        raise ValueError("torch_dtype: %s not supported" % (torch_dtype,))


def wholegraph_dtype_to_torch_dtype(wg_dtype: WholeGraphDataType):
    """
    Convert WholeGraphDataType to torch.dtype
    :param wg_dtype: WholeGraphDataType
    :return: torch.dtype
    """
    if wg_dtype == WholeGraphDataType.DtFloat:
        return torch.float
    elif wg_dtype == WholeGraphDataType.DtHalf:
        return torch.half
    elif wg_dtype == WholeGraphDataType.DtDouble:
        return torch.double
    elif wg_dtype == WholeGraphDataType.DtBF16:
        return torch.bfloat16
    elif wg_dtype == WholeGraphDataType.DtInt:
        return torch.int
    elif wg_dtype == WholeGraphDataType.DtInt64:
        return torch.int64
    elif wg_dtype == WholeGraphDataType.DtInt16:
        return torch.int16
    elif wg_dtype == WholeGraphDataType.DtInt8:
        return torch.int8
    else:
        raise ValueError("WholeGraph dtype: %s not supported" % (int(wg_dtype),))


def get_file_size(filename: str):
    """
    Get file size.
    :param filename: file name
    :return: size of file
    """
    if not os.path.isfile(filename):
        raise ValueError("File %s not found or not file" % (filename,))
    if not os.access(filename, os.R_OK):
        raise ValueError("File %s not readable" % (filename,))
    file_size = os.path.getsize(filename)
    return file_size


def str_to_wholegraph_memory_type(memory_type: str):
    if memory_type == "continuous":
        return wgb.WholeGraphMemoryType.MtContinuous
    elif memory_type == "distributed":
        return wgb.WholeGraphMemoryType.MtDistributed
    else:
        raise ValueError(
            f"WholeGraph memory type {memory_type} not supported,"
            " should be (continuous, distributed)"
        )


def str_to_wholegraph_log_level(str_log_level: str):
    if str_log_level == "error":
        return wgb.WholeGraphLogLevel.LevError
    elif str_log_level == "warn":
        return wgb.WholeGraphLogLevel.LevWarn
    elif str_log_level == "info":
        return wgb.WholeGraphLogLevel.LevInfo
    elif str_log_level == "debug":
        return wgb.WholeGraphLogLevel.LevDebug
    elif str_log_level == "trace":
        return wgb.WholeGraphLogLevel.LevTrace
    else:
        raise ValueError(
            f"WholeGraph log level {str_log_level} not supported,"
            " should be (error, warn, info, debug, trace)"
        )


def str_to_wholegraph_location(memory_location: str):
    if memory_location == "cuda":
        return wgb.WholeGraphMemoryLocation.MlDevice
    elif memory_location == "cpu":
        return wgb.WholeGraphMemoryLocation.MlHost
    else:
        raise ValueError(
            "WholeGraph memory location %s not supported, should be (cuda, cpu)"
            % (memory_location,)
        )


def str_to_wholegraph_access_type(access_type: str):
    if access_type == "readonly" or access_type == "ro":
        return wgb.WholeGraphAccessType.AtReadOnly
    elif access_type == "readwrite" or access_type == "rw":
        return wgb.WholeGraphAccessType.AtReadWrite
    else:
        raise ValueError(
            f"WholeGraph access {access_type} not supported, "
            "should be (readonly, ro, readwrite, rw)"
        )


def str_to_wholegraph_optimizer_type(optimizer_type: str):
    if optimizer_type == "sgd":
        return wgb.WholeGraphOptimizerType.OptSgd
    elif optimizer_type == "adam":
        return wgb.WholeGraphOptimizerType.OptLazyAdam
    elif optimizer_type == "adagrad":
        return wgb.WholeGraphOptimizerType.OptAdaGrad
    elif optimizer_type == "rmsprop":
        return wgb.WholeGraphOptimizerType.OptRmsProp
    else:
        raise ValueError(
            f"WholeGraph optimizer {optimizer_type} not"
            " supported, should be (sgd, adam, adagrad, rmsprop)"
        )


def str_to_wholegraph_distributed_backend_type(distributed_backend: str):
    if distributed_backend == "nccl":
        return wgb.WholeGraphDistributedBackend.DbNCCL
    else:
        raise ValueError(
            "WholeGraph distributed backend"
            f" {distributed_backend} not supported,"
            " should be nccl"
        )


def wholegraph_distributed_backend_type_to_str(
    distributed_backend: wgb.WholeGraphDistributedBackend,
):
    if distributed_backend == wgb.WholeGraphDistributedBackend.DbNCCL:
        return "nccl"
    else:
        raise ValueError(
            "WholeGraph distributed_backend"
            " not supported, should be DbNCCL"
        )


def get_part_file_name(prefix: str, part_id: int, part_count: int):
    return "%s_part_%d_of_%d" % (prefix, part_id, part_count)


def get_part_file_list(prefix: str, part_count: int):
    filelist = []
    for part_id in range(part_count):
        filelist.append("%s_part_%d_of_%d" % (prefix, part_id, part_count))
    return filelist
