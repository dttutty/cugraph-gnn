# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pylibwholegraph.binding.wholegraph_binding as wgb
from pylibwholegraph.utils.multiprocess import multiprocess_run
from wholegraph_torch.initialize import init_torch_env_and_create_wg_comm
from wholegraph_torch.dlpack_utils import torch_import_from_dlpack


# Run with:
# python3 -m pytest ../tests/pylibwholegraph/test_pylibwholegraph_binding.py -s


def single_test_case(wg_comm, mt, ml, malloc_size, granularity):
    torch = pytest.importorskip("torch")
    world_rank = wg_comm.get_rank()
    print("Rank=%d testing mt=%s, ml=%s" % (world_rank, mt, ml))
    h = wgb.malloc(malloc_size, wg_comm, mt, ml, granularity)
    global_tensor = None
    view_device = wgb.WholeGraphMemoryLocation.MlDevice
    view_device_id = world_rank
    tensor_data_type = wgb.WholeGraphDataType.DtInt64
    elt_size = 8

    local_tensor, local_offset = h.get_local_flatten_tensor(
        torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
    )
    local_data_torch = torch.arange(
        local_offset, local_offset + local_tensor.shape[0], dtype=torch.int64
    )
    local_tensor.copy_(local_data_torch)

    local_view_tensor, _ = h.get_local_flatten_tensor(
        torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
    )
    assert torch.equal(local_view_tensor.cpu(), local_data_torch)
    del local_data_torch, local_view_tensor

    wg_comm.barrier()

    if mt == wgb.WholeGraphMemoryType.MtDistributed:
        with pytest.raises(ValueError):
            global_tensor, _ = h.get_global_flatten_tensor(
                torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
            )
    else:
        global_tensor, _ = h.get_global_flatten_tensor(
            torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
        )
        global_data_torch = torch.arange(0, malloc_size // elt_size, dtype=torch.int64)
        assert torch.equal(global_tensor.cpu(), global_data_torch)
        del global_data_torch

    wgb.free(h)


def routine_func(world_rank: int, world_size: int):
    wg_comm, _ = init_torch_env_and_create_wg_comm(
        world_rank, world_size, world_rank, world_size
    )
    wg_comm = wg_comm.wgb_comm

    single_rank_size = 1024 * 1024 * 1024
    malloc_size = single_rank_size * world_size
    granularity = 256

    print("")

    for mt in [
        wgb.WholeGraphMemoryType.MtContinuous,
        wgb.WholeGraphMemoryType.MtDistributed,
    ]:
        for ml in [
            wgb.WholeGraphMemoryLocation.MlHost,
            wgb.WholeGraphMemoryLocation.MlDevice,
        ]:
            if wg_comm.support_type_location(mt, ml):
                single_test_case(wg_comm, mt, ml, malloc_size, granularity)
    wgb.finalize()


def test_dlpack(torch):
    gpu_count = wgb.fork_get_gpu_count()
    assert gpu_count > 0
    multiprocess_run(gpu_count, routine_func)
