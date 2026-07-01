# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Union, List

from pylibwholegraph.utils.imports import import_optional

from .comm import get_global_communicator
from .embedding import create_embedding, create_embedding_from_filelist
from .storage_backend import (
    StorageBackend,
    backend_to_wholegraph_memory_type,
)
from .tensor import create_wholegraph_tensor, create_wholegraph_tensor_from_filelist

torch = import_optional("torch")


def copy_host_global_tensor_to_local(wg_tensor, host_tensor, wg_comm):
    local_tensor, local_start = wg_tensor.get_local_tensor(host_view=False)

    local_tensor.copy_(host_tensor[local_start : local_start + local_tensor.shape[0]])
    wg_comm.barrier()


def _create_wg_dist_storage(
    *,
    file_list=None,
    shape: list,
    dtype: "torch.dtype",
    location: str,
    partition_book: Union[List[int], None],
    backend: StorageBackend,
    is_embedding: bool = False,
    **kwargs,
):
    global_comm = get_global_communicator()
    memory_type = backend_to_wholegraph_memory_type(backend)
    cache_policy = kwargs.pop("cache_policy", None)

    if is_embedding or cache_policy is not None:
        if len(shape) != 2:
            raise ValueError("The shape of the embedding tensor must be 2D.")

        if file_list is None:
            return create_embedding(
                global_comm,
                memory_type,
                location,
                dtype,
                shape,
                cache_policy=cache_policy,
                embedding_entry_partition=partition_book,
                **kwargs,
            )

        return create_embedding_from_filelist(
            global_comm,
            memory_type,
            location,
            file_list,
            dtype,
            shape[1],
            cache_policy=cache_policy,
            embedding_entry_partition=partition_book,
            **kwargs,
        )

    if len(shape) not in [1, 2]:
        raise ValueError("The shape of the tensor must be 2D or 1D.")

    if file_list is None:
        return create_wholegraph_tensor(
            global_comm,
            memory_type,
            location,
            shape,
            dtype,
            strides=None,
            tensor_entry_partition=partition_book,
        )

    last_dim_size = 0 if len(shape) == 1 else shape[1]
    return create_wholegraph_tensor_from_filelist(
        global_comm,
        memory_type,
        location,
        file_list,
        dtype,
        last_dim_size,
        tensor_entry_partition=partition_book,
    )


def create_wg_dist_tensor(
    shape: list,
    dtype: "torch.dtype",
    location: str = "cpu",
    partition_book: Union[List[int], None] = None,
    backend: StorageBackend = "nccl",
    **kwargs,
):
    """
    Create a WholeGraph-managed distributed tensor.

    Parameters
    ----------
    shape : list
        The shape of the tensor. It has to be a two-dimensional
        or one-dimensional tensor for now.
        The first dimension typically is the number of nodes/edges.
        The second dimension is the feature/embedding dimension.
    dtype : torch.dtype
        The dtype of the tensor.
    location : str, optional
        The desired location to store the embedding [ "cpu" | "cuda" ]
    partition_book : list, optional
        The partition book for the embedding tensor.
        The length of the partition book should be the same as the number of ranks.
        Defaults to an even partitioning scheme.
    backend : str, optional
        The backend for the distributed tensor [ "nccl" | "vmm" ]
    """
    return _create_wg_dist_storage(
        shape=shape,
        dtype=dtype,
        location=location,
        partition_book=partition_book,
        backend=backend,
        **kwargs,
    )


def create_wg_dist_tensor_from_files(
    file_list: List[str],
    shape: list,
    dtype: "torch.dtype",
    location: str = "cpu",
    partition_book: Union[List[int], None] = None,
    backend: StorageBackend = "nccl",
    **kwargs,
):
    """
    Create a WholeGraph-managed distributed tensor from a list of files.

    Parameters
    ----------
    file_list : list
        The list of files to load the embedding tensor.
    shape : list
        The shape of the tensor. It has to be a two-dimensional
        or one-dimensional tensor for now.
        The first dimension typically is the number of nodes/edges.
        The second dimension is the feature/embedding dimension.
    dtype : torch.dtype
        The dtype of the tensor.
    location : str, optional
        The desired location to store the embedding [ "cpu" | "cuda" ]
    partition_book : list, optional
        The partition book for the embedding tensor.
        The length of the partition book should be the same as the number of ranks.
        Defaults to an even partitioning scheme.
    backend : str, optional
        The backend for the distributed tensor [ "nccl" | "vmm" ]
    """
    return _create_wg_dist_storage(
        file_list=file_list,
        shape=shape,
        dtype=dtype,
        location=location,
        partition_book=partition_book,
        backend=backend,
        **kwargs,
    )


def is_empty(a):
    return a.numel() == 0


def empty(dim: int = 1):
    if dim == 1:
        return torch.tensor([], dtype=torch.int32)
    elif dim == 2:
        return torch.tensor([], dtype=torch.int32).view(0, 1)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
