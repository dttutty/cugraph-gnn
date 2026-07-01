# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np

from cugraph_pyg.data.graph_store import GraphStore
from cugraph_pyg.sampler import BaseSampler
from cugraph_pyg.sampler.distributed_sampler import DistributedNeighborSampler
from cugraph_pyg.utils.imports import import_optional

torch_geometric = import_optional("torch_geometric")


def require_cugraph_store(data):
    if (
        not isinstance(data, (list, tuple))
        or len(data) < 2
        or not isinstance(data[1], GraphStore)
    ):
        raise NotImplementedError("Currently can't accept non-cugraph graphs")
    return data


def require_cugraph_sampler(sampler):
    if not isinstance(sampler, BaseSampler):
        raise NotImplementedError("Must provide a cuGraph sampler")


def warn_ignored_loader_args(
    *,
    filter_per_worker=None,
    custom_cls=None,
    transform=None,
    transform_sampler_output=None,
):
    if filter_per_worker:
        warnings.warn("filter_per_worker is currently ignored")
    if custom_cls is not None:
        warnings.warn("custom_cls is currently ignored")
    if transform is not None:
        warnings.warn("transform is currently ignored.")
    if transform_sampler_output is not None:
        warnings.warn("transform_sampler_output is currently ignored.")


def normalize_neighbor_options(
    *,
    subgraph_type,
    temporal_strategy,
    neighbor_sampler,
    directed,
    is_sorted,
    temporal_comparison,
):
    subgraph_type = torch_geometric.sampler.base.SubgraphType(subgraph_type)
    temporal_comparison = temporal_comparison or "monotonically_decreasing"

    if not directed:
        subgraph_type = torch_geometric.sampler.base.SubgraphType.induced
        warnings.warn(
            "The 'directed' argument is deprecated. "
            "Use subgraph_type='induced' instead."
        )
    if subgraph_type != torch_geometric.sampler.base.SubgraphType.directional:
        raise ValueError("Only directional subgraphs are currently supported")
    if temporal_strategy != "uniform":
        warnings.warn("Only the uniform temporal strategy is currently supported")
    if neighbor_sampler is not None:
        raise ValueError("Passing a neighbor sampler is currently unsupported")
    if is_sorted:
        warnings.warn("The 'is_sorted' argument is ignored by cuGraph.")

    return temporal_comparison


def normalize_compression(graph_store, compression):
    if compression is None:
        compression = "CSR" if graph_store.is_homogeneous else "COO"
    elif compression not in ["CSR", "COO"]:
        raise ValueError("Invalid value for compression (expected 'CSR' or 'COO')")

    if not graph_store.is_homogeneous and compression != "COO":
        raise ValueError("Only COO format is supported for heterogeneous graphs!")

    return compression


def normalize_num_neighbors(graph_store, num_neighbors):
    if not isinstance(num_neighbors, dict):
        return num_neighbors

    sorted_keys, _, _ = graph_store._numeric_edge_types
    fanout_length = len(next(iter(num_neighbors.values())))
    normalized = np.zeros(fanout_length * len(sorted_keys), dtype="int32")

    for i, key in enumerate(sorted_keys):
        if key in num_neighbors:
            for hop in range(fanout_length):
                normalized[hop * len(sorted_keys) + i] = num_neighbors[key][hop]

    return normalized


def build_distributed_neighbor_sampler(
    *,
    feature_store,
    graph_store,
    num_neighbors,
    batch_size,
    compression,
    replace,
    disjoint,
    local_seeds_per_call,
    weight_attr,
    is_temporal,
    temporal_comparison,
):
    return BaseSampler(
        DistributedNeighborSampler(
            graph_store._graph,
            retain_original_seeds=True,
            fanout=normalize_num_neighbors(graph_store, num_neighbors),
            prior_sources_behavior="exclude",
            deduplicate_sources=True,
            compression=compression,
            compress_per_hop=False,
            with_replacement=replace,
            disjoint=disjoint,
            local_seeds_per_call=local_seeds_per_call,
            biased=(weight_attr is not None),
            heterogeneous=(not graph_store.is_homogeneous),
            temporal=is_temporal,
            temporal_comparison=temporal_comparison,
            vertex_type_offsets=graph_store._vertex_offset_array,
            num_edge_types=len(graph_store.get_all_edge_attrs()),
        ),
        (feature_store, graph_store),
        batch_size=batch_size,
    )
