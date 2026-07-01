# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Tuple, Callable, Optional

import cugraph_pyg
from cugraph_pyg.utils.imports import import_optional

from ._validation import (
    require_cugraph_sampler,
    require_cugraph_store,
    warn_ignored_loader_args,
)
from .utils import (
    generate_seed,
    get_input_permutation,
    index_optional,
    make_input_id,
    validate_input_batch_size,
)

torch_geometric = import_optional("torch_geometric")


class NodeLoader:
    """
    Duck-typed version of torch_geometric.loader.NodeLoader.
    Loads samples from batches of input nodes using a
    `~cugraph_pyg.sampler.BaseSampler.sample_from_nodes`
    function.
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
        node_sampler: "cugraph_pyg.sampler.BaseSampler",
        input_nodes: "torch_geometric.typing.InputNodes" = None,
        input_time: "torch_geometric.typing.OptTensor" = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional["torch_geometric.data.HeteroData"] = None,
        input_id: "torch_geometric.typing.OptTensor" = None,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
            data: Data, HeteroData, or Tuple[FeatureStore, GraphStore]
                See torch_geometric.loader.NodeLoader.
            node_sampler: BaseSampler
                See torch_geometric.loader.NodeLoader.
            input_nodes: InputNodes
                See torch_geometric.loader.NodeLoader.
            input_time: OptTensor
                See torch_geometric.loader.NodeLoader.
            transform: Callable (optional, default=None)
                This argument currently has no effect.
            transform_sampler_output: Callable (optional, default=None)
                This argument currently has no effect.
            filter_per_worker: bool (optional, default=False)
                This argument currently has no effect.
            custom_cls: HeteroData
                This argument currently has no effect.  This loader will
                always return a Data or HeteroData object.
            input_id: OptTensor
                See torch_geometric.loader.NodeLoader.

        """
        require_cugraph_store(data)
        require_cugraph_sampler(node_sampler)
        warn_ignored_loader_args(
            filter_per_worker=filter_per_worker,
            custom_cls=custom_cls,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
        )

        (
            input_type,
            input_nodes,
            input_id,
        ) = torch_geometric.loader.utils.get_input_nodes(
            data,
            input_nodes,
            input_id,
        )
        input_nodes = input_nodes.detach().clone()

        validate_input_batch_size(
            input_nodes.numel(),
            batch_size,
            drop_last,
            input_kind="nodes",
            input_name="input_nodes",
        )

        if input_type is not None:
            input_nodes += data[1]._vertex_offsets[input_type]

        self.__input_data = torch_geometric.sampler.NodeSamplerInput(
            input_id=make_input_id(len(input_nodes), input_id),
            node=input_nodes,
            time=input_time,
            input_type=input_type,
        )

        self.__data = data

        self.__node_sampler = node_sampler

        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__drop_last = drop_last

    def __iter__(self):
        perm = get_input_permutation(
            self.__input_data.node.numel(),
            self.__shuffle,
            self.__batch_size,
            self.__drop_last,
        )

        input_data = torch_geometric.sampler.NodeSamplerInput(
            input_id=self.__input_data.input_id[perm],
            node=self.__input_data.node[perm],
            time=index_optional(self.__input_data.time, perm),
            input_type=self.__input_data.input_type,
        )

        return cugraph_pyg.sampler.SampleIterator(
            self.__data,
            self.__node_sampler.sample_from_nodes(
                input_data, random_state=generate_seed()
            ),
        )
