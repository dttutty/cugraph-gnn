# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Optional, Literal

from pylibwholegraph.utils.imports import import_optional

from .comm import get_global_communicator

torch = import_optional("torch")

StorageBackend = Literal["nccl", "vmm"]
WholeGraphMemoryType = Literal["distributed", "continuous"]
WholeGraphLocation = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class StoragePolicy:
    r"""Resolved WholeGraph storage policy for higher-level adapters."""

    backend: StorageBackend
    location: WholeGraphLocation = "cpu"

    @property
    def wholegraph_memory_type(self) -> WholeGraphMemoryType:
        return backend_to_wholegraph_memory_type(self.backend)


def resolve_storage_policy(
    *,
    backend: Optional[StorageBackend] = None,
    location: WholeGraphLocation = "cpu",
) -> StoragePolicy:
    if location not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported WholeGraph storage location: {location}")

    resolved_backend = infer_default_backend() if backend is None else backend
    backend_to_wholegraph_memory_type(resolved_backend)
    return StoragePolicy(backend=resolved_backend, location=location)


def backend_to_wholegraph_memory_type(backend: StorageBackend) -> WholeGraphMemoryType:
    if backend == "nccl":
        return "distributed"
    if backend == "vmm":
        return "continuous"
    raise ValueError(f"Unsupported backend: {backend}")


def infer_default_backend() -> StorageBackend:
    """
    Select the default WholeGraph storage backend for the current process group.

    Higher-level adapters use this policy for internal WholeGraph-backed edge
    and feature tensors while keeping their public APIs storage-agnostic.
    """
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = torch.distributed.get_world_size()

    if local_size == world_size:
        return "vmm"

    return "vmm" if has_mapped_storage_support() else "nccl"


def has_mapped_storage_support() -> bool:
    r"""Check whether the current communicator supports mapped storage."""
    global_comm = get_global_communicator("nccl")
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = torch.distributed.get_world_size()

    if local_size == world_size:
        return global_comm.support_type_location("continuous", "cuda")

    is_cuda_supported = global_comm.support_type_location("continuous", "cuda")
    is_cpu_supported = global_comm.support_type_location("continuous", "cpu")
    return is_cuda_supported and is_cpu_supported
