# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from typing import Union, Tuple, Optional, Callable, List, Dict

from cugraph_pyg.loader import LinkLoader
from cugraph_pyg.utils.imports import import_optional
from ._validation import (
    build_distributed_neighbor_sampler,
    normalize_compression,
    normalize_neighbor_options,
    require_cugraph_store,
)

torch_geometric = import_optional("torch_geometric")


class LinkNeighborLoader(LinkLoader):
    """
    Duck-typed version of torch_geometric.loader.LinkNeighborLoader

    Link loader that implements the neighbor sampling
    algorithm used in GraphSAGE.
    """

    def __init__(
        self,
        data: Union[
            "torch_geometric.data.Data",
            "torch_geometric.data.HeteroData",
            Tuple[
                "torch_geometric.data.FeatureStore", "torch_geometric.data.GraphStore"
            ],
        ],
        num_neighbors: Union[
            List[int], Dict["torch_geometric.typing.EdgeType", List[int]]
        ],
        edge_label_index: "torch_geometric.typing.InputEdges" = None,
        edge_label: "torch_geometric.typing.OptTensor" = None,
        edge_label_time: "torch_geometric.typing.OptTensor" = None,
        replace: bool = False,
        subgraph_type: Union[
            "torch_geometric.typing.SubgraphType", str
        ] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        neg_sampling: Optional["torch_geometric.sampler.NegativeSampling"] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional["torch_geometric.sampler.NeighborSampler"] = None,
        directed: bool = True,  # Deprecated.
        batch_size: int = 16,  # Refers to number of edges per batch.
        compression: Optional[str] = None,
        local_seeds_per_call: Optional[int] = None,
        temporal_comparison: Optional[str] = None,
        **kwargs,
    ):
        """
        data: Data, HeteroData, or Tuple[FeatureStore, GraphStore]
            See torch_geometric.loader.LinkNeighborLoader.
        num_neighbors: List[int] or Dict[EdgeType, List[int]]
            Fanout values.
            See torch_geometric.loader.LinkNeighborLoader.
        edge_label_index: InputEdges
            Input edges for sampling.
            See torch_geometric.loader.LinkNeighborLoader.
        edge_label: OptTensor
            Labels for input edges.
            See torch_geometric.loader.LinkNeighborLoader.
        edge_label_time: OptTensor
            Time attribute for input edges.
            See torch_geometric.loader.LinkNeighborLoader.
        replace: bool (optional, default=False)
            Whether to sample with replacement.
            See torch_geometric.loader.LinkNeighborLoader.
        subgraph_type: Union[SubgraphType, str] (optional, default='directional')
            The type of subgraph to return.
            Currently only 'directional' is supported.
            See torch_geometric.loader.LinkNeighborLoader.
        disjoint: bool (optional, default=False)
            Whether to perform disjoint sampling.
            See torch_geometric.loader.LinkNeighborLoader.
        temporal_strategy: str (optional, default='uniform')
            Currently only 'uniform' is suppported.
            See torch_geometric.loader.LinkNeighborLoader.
        time_attr: str (optional, default=None)
            Used for temporal sampling.
            See torch_geometric.loader.LinkNeighborLoader.
        weight_attr: str (optional, default=None)
            Used for biased sampling.
            See torch_geometric.loader.LinkNeighborLoader.
        transform: Callable (optional, default=None)
            See torch_geometric.loader.LinkNeighborLoader.
        transform_sampler_output: Callable (optional, default=None)
            See torch_geometric.loader.LinkNeighborLoader.
        is_sorted: bool (optional, default=False)
            Ignored by cuGraph.
            See torch_geometric.loader.LinkNeighborLoader.
        filter_per_worker: bool (optional, default=False)
            Currently ignored by cuGraph, but this may
            change once in-memory sampling is implemented.
            See torch_geometric.loader.LinkNeighborLoader.
        neighbor_sampler: torch_geometric.sampler.NeighborSampler
            (optional, default=None)
            Not supported by cuGraph.
            See torch_geometric.loader.LinkNeighborLoader.
        directed: bool (optional, default=True)
            Deprecated.
            See torch_geometric.loader.LinkNeighborLoader.
        batch_size: int (optional, default=16)
            The number of input nodes per output minibatch.
            See torch.utils.dataloader.
        compression: str (optional, default=None)
            The compression type to use if writing samples to disk.
            If not provided, it is automatically chosen.
        local_seeds_per_call: int (optional, default=None)
            The number of seeds to process within a single sampling call.
            Manually tuning this parameter is not recommended but reducing
            it may conserve GPU memory.  The total number of seeds processed
            per sampling call is equal to the sum of this parameter across
            all workers.  If not provided, it will be automatically
            calculated.
            See cugraph_pyg.sampler.BaseDistributedSampler.
        temporal_comparison: str (optional, default='monotonically_decreasing')
            The comparison operator for temporal sampling
            ('strictly_increasing', 'monotonically_increasing',
            'strictly_decreasing', 'monotonically_decreasing', 'last').
            Note that this should be 'last' for temporal_strategy='last'.
            See cugraph_pyg.sampler.BaseDistributedSampler.
        **kwargs
            Other keyword arguments passed to the superclass.
        """

        temporal_comparison = normalize_neighbor_options(
            subgraph_type=subgraph_type,
            temporal_strategy=temporal_strategy,
            neighbor_sampler=neighbor_sampler,
            directed=directed,
            is_sorted=is_sorted,
            temporal_comparison=temporal_comparison,
        )
        require_cugraph_store(data)

        feature_store, graph_store = data
        compression = normalize_compression(graph_store, compression)

        is_temporal = (edge_label_time is not None) and (time_attr is not None)

        if not is_temporal and (edge_label_time is not None or time_attr is not None):
            warnings.warn(
                "Edge-based temporal sampling requires that both edge_label_time "
                "and time_attr are provided. Defaulting to non-temporal sampling."
            )

        if weight_attr is not None:
            graph_store._set_weight_attr((feature_store, weight_attr))
        if is_temporal:
            graph_store._set_time_attr((feature_store, time_attr))

        sampler = build_distributed_neighbor_sampler(
            feature_store=feature_store,
            graph_store=graph_store,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            compression=compression,
            replace=replace,
            disjoint=disjoint,
            local_seeds_per_call=local_seeds_per_call,
            weight_attr=weight_attr,
            is_temporal=is_temporal,
            temporal_comparison=temporal_comparison,
        )

        super().__init__(
            (feature_store, graph_store),
            sampler,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            edge_label_time=edge_label_time,
            neg_sampling=neg_sampling,
            neg_sampling_ratio=neg_sampling_ratio,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            batch_size=batch_size,
            **kwargs,
        )
