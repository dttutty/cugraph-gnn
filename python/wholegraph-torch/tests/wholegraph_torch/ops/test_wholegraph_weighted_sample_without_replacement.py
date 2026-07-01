# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pylibwholegraph.utils.multiprocess import multiprocess_run
from wholegraph_torch.initialize import init_torch_env_and_create_wg_comm
import pylibwholegraph.binding.wholegraph_binding as wgb
import random
from functools import partial
from test_utils.test_comm import (
    gen_csr_graph,
    copy_host_1D_tensor_to_wholegraph,
    host_get_sample_offset_tensor,
    host_sample_all_neighbors,
    int_to_wholegraph_datatype,
    int_to_wholegraph_location,
    int_to_wholegraph_type,
)
import wholegraph_torch.wholegraph_ops as wg_ops


def host_weighted_sample_without_replacement_func(
    host_csr_row_ptr,
    host_csr_col_ptr,
    host_csr_weight_ptr,
    center_nodes,
    output_sample_offset_tensor,
    col_id_dtype,
    csr_weight_dtype,
    total_sample_count,
    max_sample_count,
    random_seed,
):
    torch = pytest.importorskip("torch")
    output_dest_tensor = torch.empty((total_sample_count,), dtype=col_id_dtype)
    output_center_localid_tensor = torch.empty((total_sample_count,), dtype=torch.int32)
    output_edge_gid_tensor = torch.empty((total_sample_count,), dtype=torch.int64)
    center_nodes_count = center_nodes.size(0)
    block_size = 128 if max_sample_count <= 256 else 256

    for i in range(center_nodes_count):
        node_id = center_nodes[i]
        start = host_csr_row_ptr[node_id]
        end = host_csr_row_ptr[node_id + 1]
        neighbor_count = end - start
        output_id = output_sample_offset_tensor[i]
        gidx = i * block_size
        if neighbor_count <= max_sample_count:
            for j in range(end - start):
                output_dest_tensor[output_id + j] = host_csr_col_ptr[start + j]
                output_center_localid_tensor[output_id + j] = i
                output_edge_gid_tensor[output_id + j] = start + j
        else:
            total_neighbor_generated_weights = torch.tensor([], dtype=csr_weight_dtype)
            edge_weight_corresponding_ids = torch.tensor([], dtype=col_id_dtype)
            for j in range(block_size):
                local_gidx = gidx + j
                local_edge_weights = torch.tensor([], dtype=csr_weight_dtype)
                generated_edge_weight_count = 0
                for id in range(j, neighbor_count, block_size):
                    local_edge_weights = torch.cat(
                        (
                            local_edge_weights,
                            torch.tensor(
                                [host_csr_weight_ptr[start + id]],
                                dtype=csr_weight_dtype,
                            ),
                        )
                    )
                    generated_edge_weight_count += 1
                    edge_weight_corresponding_ids = torch.cat(
                        (
                            edge_weight_corresponding_ids,
                            torch.tensor([id], dtype=col_id_dtype),
                        )
                    )
                random_values = (
                    wg_ops.generate_exponential_distribution_negative_float_cpu(
                        random_seed, local_gidx, generated_edge_weight_count
                    )
                )
                generated_random_weight = torch.tensor(
                    [
                        (1.0 / local_edge_weights[i]) * random_values[i]
                        for i in range(generated_edge_weight_count)
                    ]
                )

                total_neighbor_generated_weights = torch.cat(
                    (total_neighbor_generated_weights, generated_random_weight)
                )
            assert total_neighbor_generated_weights.size(0) == neighbor_count
            _, sorted_weight_ids = torch.sort(
                total_neighbor_generated_weights, descending=True
            )
            sorted_top_m_weight_ids = edge_weight_corresponding_ids[
                sorted_weight_ids[0:max_sample_count]
            ]
            for sample_id in range(max_sample_count):
                output_dest_tensor[output_id + sample_id] = host_csr_col_ptr[
                    start + sorted_top_m_weight_ids[sample_id]
                ]
                output_center_localid_tensor[output_id + sample_id] = i
                output_edge_gid_tensor[output_id + sample_id] = (
                    start + sorted_top_m_weight_ids[sample_id]
                )
    return output_dest_tensor, output_center_localid_tensor, output_edge_gid_tensor


def host_weighted_sample_without_replacement(
    host_csr_row_ptr,
    host_csr_col_ptr,
    host_csr_weight_ptr,
    center_nodes,
    max_sample_count,
    col_id_dtype,
    random_seed,
):
    torch = pytest.importorskip("torch")
    center_nodes_count = center_nodes.size(0)
    output_sample_offset_tensor = host_get_sample_offset_tensor(
        host_csr_row_ptr, center_nodes, max_sample_count
    )
    total_sample_count = output_sample_offset_tensor[center_nodes_count]

    if max_sample_count <= 0:
        return host_sample_all_neighbors(
            host_csr_row_ptr,
            host_csr_col_ptr,
            center_nodes,
            output_sample_offset_tensor,
            col_id_dtype,
            total_sample_count,
        )
    if max_sample_count > 1024:
        raise ValueError(
            "invalid host_unweighted_sample_without_replacement test max_sample_count"
        )

    torch_col_id_dtype = torch.int32
    if col_id_dtype == wgb.WholeGraphDataType.DtInt64:
        torch_col_id_dtype = torch.int64

    (
        output_dest_tensor,
        output_center_localid_tensor,
        output_edge_gid_tensor,
    ) = host_weighted_sample_without_replacement_func(
        host_csr_row_ptr,
        host_csr_col_ptr,
        host_csr_weight_ptr,
        center_nodes,
        output_sample_offset_tensor,
        torch_col_id_dtype,
        host_csr_weight_ptr.dtype,
        total_sample_count,
        max_sample_count,
        random_seed,
    )

    return (
        output_sample_offset_tensor,
        output_dest_tensor,
        output_center_localid_tensor,
        output_edge_gid_tensor,
    )


def routine_func(world_rank: int, world_size: int, **kwargs):
    torch = pytest.importorskip("torch")
    wg_comm, _ = init_torch_env_and_create_wg_comm(
        world_rank, world_size, world_rank, world_size
    )
    wg_comm = wg_comm.wgb_comm
    host_csr_row_ptr = kwargs["host_csr_row_ptr"]
    host_csr_col_ptr = kwargs["host_csr_col_ptr"]
    host_csr_weight_ptr = kwargs["host_csr_weight_ptr"]
    graph_node_count = kwargs["graph_node_count"]
    graph_edge_count = kwargs["graph_edge_count"]
    max_sample_count = kwargs["max_sample_count"]
    center_node_count = kwargs["center_node_count"]
    center_node_dtype = kwargs["center_node_dtype"]
    int_col_id_dtype = kwargs["col_id_dtype"]
    int_csr_weight_dtype = kwargs["csr_weight_dtype"]
    int_wholegraph_location = kwargs["wholegraph_location"]
    int_wholegraph_type = kwargs["wholegraph_type"]
    need_center_local_output = kwargs["need_center_local_output"]
    need_edge_output = kwargs["need_edge_output"]

    world_rank = wg_comm.get_rank()
    world_size = wg_comm.get_size()

    col_id_dtype = int_to_wholegraph_datatype(int_col_id_dtype)
    csr_weight_dtype = int_to_wholegraph_datatype(int_csr_weight_dtype)
    wholegraph_location = int_to_wholegraph_location(int_wholegraph_location)
    wholegraph_type = int_to_wholegraph_type(int_wholegraph_type)

    if not wg_comm.support_type_location(wholegraph_type, wholegraph_location):
        wgb.finalize()
        return

    wg_csr_row_ptr = wgb.create_wholegraph_array(
        wgb.WholeGraphDataType.DtInt64,
        graph_node_count + 1,
        wg_comm,
        wholegraph_type,
        wholegraph_location,
    )
    wg_csr_col_ptr = wgb.create_wholegraph_array(
        col_id_dtype, graph_edge_count, wg_comm, wholegraph_type, wholegraph_location
    )
    wg_csr_weight_ptr = wgb.create_wholegraph_array(
        csr_weight_dtype,
        graph_edge_count,
        wg_comm,
        wholegraph_type,
        wholegraph_location,
    )

    copy_host_1D_tensor_to_wholegraph(
        wg_csr_row_ptr, host_csr_row_ptr, world_rank, world_size, wg_comm
    )
    copy_host_1D_tensor_to_wholegraph(
        wg_csr_col_ptr, host_csr_col_ptr, world_rank, world_size, wg_comm
    )
    copy_host_1D_tensor_to_wholegraph(
        wg_csr_weight_ptr, host_csr_weight_ptr, world_rank, world_size, wg_comm
    )

    wg_comm.barrier()

    center_node_tensor = torch.randint(
        0, graph_node_count, (center_node_count,), dtype=center_node_dtype
    )
    center_node_tensor_cuda = center_node_tensor.cuda()
    random_seed = random.randint(1, 10000)

    # output_sample_offset_tensor_cuda,
    # output_dest_tensor_cuda,
    # output_center_localid_tensor_cuda,
    # output_edge_gid_tensor_cuda =
    # torch.ops.wholegraph.weighted_sample_without_replacement(wg_csr_row_ptr.get_c_handle(),
    #                            wg_csr_col_ptr.get_c_handle(),
    #                            wg_csr_weight_ptr.get_c_handle(),
    #                            center_node_tensor_cuda,
    #                            max_sample_count,
    #                            random_seed)
    output_sample_offset_tensor = None
    output_dest_tensor = None
    output_center_localid_tensor = None
    output_edge_gid_tensor = None

    output_tensors = wg_ops.weighted_sample_without_replacement(
        wg_csr_row_ptr,
        wg_csr_col_ptr,
        wg_csr_weight_ptr,
        center_node_tensor_cuda,
        max_sample_count,
        random_seed,
        need_center_local_output=need_center_local_output,
        need_edge_output=need_edge_output,
    )
    output_cpu_tensors = tuple(tensor.cpu() for tensor in output_tensors)
    if need_edge_output and need_center_local_output:
        (
            output_sample_offset_tensor,
            output_dest_tensor,
            output_center_localid_tensor,
            output_edge_gid_tensor,
        ) = output_cpu_tensors
    elif need_center_local_output:
        (
            output_sample_offset_tensor,
            output_dest_tensor,
            output_center_localid_tensor,
        ) = output_cpu_tensors
    elif need_edge_output:
        (
            output_sample_offset_tensor,
            output_dest_tensor,
            output_edge_gid_tensor,
        ) = output_cpu_tensors
    else:
        output_sample_offset_tensor, output_dest_tensor = output_cpu_tensors

    (
        output_sample_offset_tensor_ref,
        output_dest_tensor_ref,
        output_center_localid_tensor_ref,
        output_edge_gid_tensor_ref,
    ) = host_weighted_sample_without_replacement(
        host_csr_row_ptr,
        host_csr_col_ptr,
        host_csr_weight_ptr,
        center_node_tensor,
        max_sample_count,
        col_id_dtype,
        random_seed,
    )

    assert torch.equal(output_sample_offset_tensor, output_sample_offset_tensor_ref)

    for i in range(center_node_count):
        start = output_sample_offset_tensor[i]
        end = output_sample_offset_tensor[i + 1]
        output_dest_tensor[start:end], sorted_ids = torch.sort(
            output_dest_tensor[start:end]
        )

        output_dest_tensor_ref[start:end], ref_sorted_ids = torch.sort(
            output_dest_tensor_ref[start:end]
        )
        output_center_localid_tensor_ref[start:end] = output_center_localid_tensor_ref[
            start:end
        ][ref_sorted_ids]
        output_edge_gid_tensor_ref[start:end] = output_edge_gid_tensor_ref[start:end][
            ref_sorted_ids
        ]
        if need_edge_output and need_center_local_output:
            output_center_localid_tensor[start:end] = output_center_localid_tensor[
                start:end
            ][sorted_ids]
            output_edge_gid_tensor[start:end] = output_edge_gid_tensor[start:end][
                sorted_ids
            ]
        elif need_center_local_output:
            output_center_localid_tensor[start:end] = output_center_localid_tensor[
                start:end
            ][sorted_ids]
        elif need_edge_output:
            output_edge_gid_tensor[start:end] = output_edge_gid_tensor[start:end][
                sorted_ids
            ]

    assert torch.equal(output_dest_tensor, output_dest_tensor_ref)
    if need_edge_output and need_center_local_output:
        assert torch.equal(
            output_center_localid_tensor, output_center_localid_tensor_ref
        )
        assert torch.equal(output_edge_gid_tensor, output_edge_gid_tensor_ref)
    elif need_center_local_output:
        assert torch.equal(
            output_center_localid_tensor, output_center_localid_tensor_ref
        )
    elif need_edge_output:
        assert torch.equal(output_edge_gid_tensor, output_edge_gid_tensor_ref)

    wgb.destroy_wholegraph_tensor(wg_csr_row_ptr)
    wgb.destroy_wholegraph_tensor(wg_csr_col_ptr)
    wgb.destroy_wholegraph_tensor(wg_csr_weight_ptr)
    wgb.finalize()


@pytest.mark.parametrize("graph_node_count", [113])
@pytest.mark.parametrize("graph_edge_count", [1043])
@pytest.mark.parametrize("max_sample_count", [11])
@pytest.mark.parametrize("center_node_count", [13])
@pytest.mark.parametrize("center_node_dtype", ["int32", "int64"])
@pytest.mark.parametrize("col_id_dtype", [0, 1])
@pytest.mark.parametrize("csr_weight_dtype", [2, 3])
@pytest.mark.parametrize("wholegraph_location", ([0, 1]))
@pytest.mark.parametrize("wholegraph_type", ([0]))
@pytest.mark.parametrize("need_center_local_output", [True, False])
@pytest.mark.parametrize("need_edge_output", [True, False])
def test_wholegraph_weighted_sample(
    graph_node_count,
    graph_edge_count,
    max_sample_count,
    center_node_count,
    center_node_dtype,
    col_id_dtype,
    csr_weight_dtype,
    wholegraph_location,
    wholegraph_type,
    need_center_local_output,
    need_edge_output,
    torch,
):
    gpu_count = wgb.fork_get_gpu_count()
    assert gpu_count > 0
    csr_col_dtype = torch.int32
    if col_id_dtype == 1:
        csr_col_dtype = torch.int64
    host_csr_row_ptr, host_csr_col_ptr, host_csr_weight_ptr = gen_csr_graph(
        graph_node_count,
        graph_edge_count,
        neighbor_node_count=None,
        csr_row_dtype=torch.int64,
        csr_col_dtype=csr_col_dtype,
        weight_dtype=torch.float32,
    )
    routine_func_partial = partial(
        routine_func,
        host_csr_row_ptr=host_csr_row_ptr,
        host_csr_col_ptr=host_csr_col_ptr,
        host_csr_weight_ptr=host_csr_weight_ptr,
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        max_sample_count=max_sample_count,
        center_node_count=center_node_count,
        center_node_dtype=getattr(torch, center_node_dtype),
        col_id_dtype=col_id_dtype,
        csr_weight_dtype=csr_weight_dtype,
        wholegraph_location=wholegraph_location,
        wholegraph_type=wholegraph_type,
        need_center_local_output=need_center_local_output,
        need_edge_output=need_edge_output,
    )
    multiprocess_run(gpu_count, routine_func_partial, True)
