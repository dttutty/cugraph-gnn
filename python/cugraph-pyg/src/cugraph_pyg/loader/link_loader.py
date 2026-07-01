# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

import cugraph_pyg
from typing import Union, Tuple, Callable, Optional

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
torch = import_optional("torch")


class LinkLoader:
    """
    Duck-typed version of torch_geometric.loader.LinkLoader.
    Loads samples from batches of input nodes using a
    `~cugraph_pyg.sampler.BaseSampler.sample_from_edges`
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
        link_sampler: "cugraph_pyg.sampler.BaseSampler",
        edge_label_index: "torch_geometric.typing.InputEdges" = None,
        edge_label: "torch_geometric.typing.OptTensor" = None,
        edge_label_time: "torch_geometric.typing.OptTensor" = None,
        neg_sampling: Optional["torch_geometric.sampler.NegativeSampling"] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional["torch_geometric.data.HeteroData"] = None,
        input_id: "torch_geometric.typing.OptTensor" = None,
        batch_size: int = 1,  # refers to number of edges in batch
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
            data: Data, HeteroData, or Tuple[FeatureStore, GraphStore]
                See torch_geometric.loader.NodeLoader.
            link_sampler: BaseSampler
                See torch_geometric.loader.LinkLoader.
            edge_label_index: InputEdges
                See torch_geometric.loader.LinkLoader.
            edge_label: OptTensor
                See torch_geometric.loader.LinkLoader.
            edge_label_time: OptTensor
                See torch_geometric.loader.LinkLoader.
            neg_sampling: Optional[NegativeSampling]
                Type of negative sampling to perform, if desired.
                See torch_geometric.loader.LinkLoader.
            neg_sampling_ratio: Optional[Union[int, float]]
                Negative sampling ratio.  Affects how many negative
                samples are generated.
                See torch_geometric.loader.LinkLoader.
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
                See torch_geometric.loader.LinkLoader.

        """
        require_cugraph_store(data)
        require_cugraph_sampler(link_sampler)
        warn_ignored_loader_args(
            filter_per_worker=filter_per_worker,
            custom_cls=custom_cls,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
        )

        if neg_sampling_ratio is not None:
            warnings.warn(
                "The 'neg_sampling_ratio' argument is deprecated in PyG"
                " and is not supported in cuGraph-PyG."
            )

        neg_sampling = torch_geometric.sampler.NegativeSampling.cast(neg_sampling)

        (
            input_type,
            edge_label_index,
        ) = torch_geometric.loader.utils.get_edge_label_index(
            data,
            (None, edge_label_index)
            if isinstance(edge_label_index, torch.Tensor)
            else edge_label_index,
        )
        edge_label_index = edge_label_index.detach().clone()

        validate_input_batch_size(
            edge_label_index.shape[1],
            batch_size,
            drop_last,
            input_kind="edges",
            input_name="edge_label_index",
        )

        # Note reverse of standard convention here
        if input_type is not None:
            edge_label_index[0] += data[1]._vertex_offsets[input_type[0]]
            edge_label_index[1] += data[1]._vertex_offsets[input_type[2]]

        self.__input_data = torch_geometric.sampler.EdgeSamplerInput(
            input_id=make_input_id(edge_label_index[0].numel(), input_id),
            row=edge_label_index[0],
            col=edge_label_index[1],
            label=edge_label,
            time=edge_label_time,
            input_type=input_type,
        )

        # Edge label check from torch_geometric.loader.LinkLoader
        if (
            neg_sampling is not None
            and neg_sampling.is_binary()
            and edge_label is not None
            and edge_label.min() == 0
        ):
            edge_label = edge_label + 1

        if (
            neg_sampling is not None
            and neg_sampling.is_triplet()
            and edge_label is not None
        ):
            raise ValueError(
                "'edge_label' needs to be undefined for "
                "'triplet'-based negative sampling. Please use "
                "`src_index`, `dst_pos_index` and "
                "`neg_pos_index` of the returned mini-batch "
                "instead to differentiate between positive and "
                "negative samples."
            )

        self.__data = data

        self.__link_sampler = link_sampler
        self.__neg_sampling = neg_sampling

        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__drop_last = drop_last

    def __iter__(self):
        perm = get_input_permutation(
            self.__input_data.row.numel(),
            self.__shuffle,
            self.__batch_size,
            self.__drop_last,
        )

        input_data = torch_geometric.sampler.EdgeSamplerInput(
            input_id=self.__input_data.input_id[perm],
            row=self.__input_data.row[perm],
            col=self.__input_data.col[perm],
            label=index_optional(self.__input_data.label, perm),
            time=index_optional(self.__input_data.time, perm),
            input_type=self.__input_data.input_type,
        )

        return cugraph_pyg.sampler.SampleIterator(
            self.__data,
            self.__link_sampler.sample_from_edges(
                input_data,
                neg_sampling=self.__neg_sampling,
                random_state=generate_seed(),
            ),
        )
