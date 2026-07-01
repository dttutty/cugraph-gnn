# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import wholegraph_torch.storage_backend as storage_backend


class _FakeDistributed:
    def __init__(self, world_size):
        self._world_size = world_size

    def get_world_size(self):
        return self._world_size


class _FakeTorch:
    def __init__(self, world_size):
        self.distributed = _FakeDistributed(world_size)


class _FakeCommunicator:
    def __init__(self, *, cuda_supported, cpu_supported):
        self._support = {
            ("continuous", "cuda"): cuda_supported,
            ("continuous", "cpu"): cpu_supported,
        }

    def support_type_location(self, memory_type, location):
        return self._support[memory_type, location]


def _patch_runtime(
    monkeypatch,
    *,
    local_world_size,
    world_size,
    cuda_supported=False,
    cpu_supported=False,
):
    communicator = _FakeCommunicator(
        cuda_supported=cuda_supported,
        cpu_supported=cpu_supported,
    )

    def _get_global_communicator(backend):
        assert backend == "nccl"
        return communicator

    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world_size))
    monkeypatch.setattr(storage_backend, "torch", _FakeTorch(world_size))
    monkeypatch.setattr(
        storage_backend,
        "get_global_communicator",
        _get_global_communicator,
    )


def test_backend_to_wholegraph_memory_type():
    assert storage_backend.backend_to_wholegraph_memory_type("nccl") == "distributed"
    assert storage_backend.backend_to_wholegraph_memory_type("vmm") == "continuous"

    with pytest.raises(ValueError, match="Unsupported backend"):
        storage_backend.backend_to_wholegraph_memory_type("nvshmem")


def test_resolve_storage_policy_uses_explicit_backend():
    policy = storage_backend.resolve_storage_policy(backend="nccl", location="cuda")

    assert policy.backend == "nccl"
    assert policy.location == "cuda"
    assert policy.wholegraph_memory_type == "distributed"


def test_resolve_storage_policy_rejects_invalid_location():
    with pytest.raises(ValueError, match="Unsupported WholeGraph storage location"):
        storage_backend.resolve_storage_policy(backend="nccl", location="host")


def test_resolve_storage_policy_rejects_invalid_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        storage_backend.resolve_storage_policy(backend="nvshmem")


def test_infer_default_backend_uses_vmm_for_single_node(monkeypatch):
    _patch_runtime(monkeypatch, local_world_size=2, world_size=2)

    assert storage_backend.infer_default_backend() == "vmm"


def test_resolve_storage_policy_uses_inferred_backend(monkeypatch):
    _patch_runtime(monkeypatch, local_world_size=2, world_size=2)

    policy = storage_backend.resolve_storage_policy(location="cpu")

    assert policy.backend == "vmm"
    assert policy.location == "cpu"
    assert policy.wholegraph_memory_type == "continuous"


def test_infer_default_backend_uses_vmm_when_multinode_mapped_storage_exists(
    monkeypatch,
):
    _patch_runtime(
        monkeypatch,
        local_world_size=2,
        world_size=4,
        cuda_supported=True,
        cpu_supported=True,
    )

    assert storage_backend.infer_default_backend() == "vmm"


def test_infer_default_backend_falls_back_to_nccl_for_multinode_without_mapping(
    monkeypatch,
):
    _patch_runtime(
        monkeypatch,
        local_world_size=2,
        world_size=4,
        cuda_supported=True,
        cpu_supported=False,
    )

    assert storage_backend.infer_default_backend() == "nccl"
