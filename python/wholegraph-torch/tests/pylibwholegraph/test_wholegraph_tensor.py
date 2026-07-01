# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pylibwholegraph.binding.wholegraph_binding as wgb
from pylibwholegraph.utils.multiprocess import multiprocess_run
from wholegraph_torch.initialize import init_torch_env_and_create_wg_comm
from test_utils.test_comm import random_partition


# Run with:
# python3 -m pytest ../tests/pylibwholegraph/test_wholegraph_tensor.py -s


def array_test_case(wg_comm, dt, mt, ml, size, entry_partition):
    world_rank = wg_comm.get_rank()
    print(
        "Rank=%d testing array size=%d dt=%s, mt=%s, ml=%s"
        % (world_rank, size, dt, mt, ml)
    )
    wg_array = wgb.create_wholegraph_array(dt, size, wg_comm, mt, ml, entry_partition)
    assert wg_array.dtype == dt
    assert wg_array.dim() == 1
    assert len(wg_array.shape) == 1
    assert wg_array.shape[0] == size
    assert len(wg_array.stride()) == 1
    assert wg_array.stride()[0] == 1
    assert wg_array.storage_offset() == 0

    wg_sub_array = wg_array.get_sub_tensor([size // 4], [-1])
    assert wg_sub_array.dtype == dt
    assert wg_sub_array.dim() == 1
    assert wg_sub_array.shape[0] == size - size // 4
    assert wg_sub_array.stride()[0] == 1
    assert wg_sub_array.storage_offset() == size // 4

    wgb.destroy_wholegraph_tensor(wg_sub_array)

    wgb.destroy_wholegraph_tensor(wg_array)


def matrix_test_case(wg_comm, dt, mt, ml, mat_size, entry_partition):
    world_rank = wg_comm.get_rank()
    print(
        "Rank=%d testing matrix size=%s dt=%s, mt=%s, ml=%s"
        % (world_rank, mat_size, dt, mt, ml)
    )
    wg_matrix = wgb.create_wholegraph_matrix(
        dt, mat_size[0], mat_size[1], -1, wg_comm, mt, ml, entry_partition
    )

    assert wg_matrix.dtype == dt
    assert wg_matrix.dim() == 2
    assert len(wg_matrix.shape) == 2
    assert wg_matrix.shape[0] == mat_size[0]
    assert wg_matrix.shape[1] == mat_size[1]
    assert len(wg_matrix.stride()) == 2
    assert wg_matrix.stride()[0] == mat_size[1]
    assert wg_matrix.stride()[1] == 1

    wg_sub_matrix = wg_matrix.get_sub_tensor(
        [mat_size[0] // 3, mat_size[1] // 5], [-1, mat_size[1] // 5 * 3]
    )
    assert wg_sub_matrix.dtype == dt
    assert wg_sub_matrix.dim() == 2
    assert wg_sub_matrix.shape[0] == mat_size[0] - mat_size[0] // 3
    assert wg_sub_matrix.shape[1] == mat_size[1] // 5 * 3 - mat_size[1] // 5
    assert wg_sub_matrix.stride()[0] == mat_size[1]
    assert wg_sub_matrix.stride()[1] == 1
    assert (
        wg_sub_matrix.storage_offset()
        == mat_size[1] // 5 + mat_size[0] // 3 * mat_size[1]
    )

    wgb.destroy_wholegraph_tensor(wg_sub_matrix)
    wgb.destroy_wholegraph_tensor(wg_matrix)


def routine_func(world_rank: int, world_size: int):
    wg_comm, _ = init_torch_env_and_create_wg_comm(
        world_rank, world_size, world_rank, world_size
    )
    wg_comm = wg_comm.wgb_comm

    single_array_size = 128 * 1024 * 1024 * world_size
    single_matrix_size = (1024 * 1024 * world_size, 128)
    dt = wgb.WholeGraphDataType.DtFloat
    array_entry_partition = random_partition(single_array_size, world_size)
    matrix_entry_partition = random_partition(single_matrix_size[0], world_size)
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
                array_test_case(
                    wg_comm, dt, mt, ml, single_array_size, array_entry_partition
                )
                matrix_test_case(
                    wg_comm, dt, mt, ml, single_matrix_size, matrix_entry_partition
                )
    wgb.finalize()


def test_wholegraph_tensor(torch):
    gpu_count = wgb.fork_get_gpu_count()
    assert gpu_count > 0
    multiprocess_run(gpu_count, routine_func)
