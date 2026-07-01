# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cugraph_pyg.utils.imports import import_optional, MissingModule

from cugraph_pyg.data import GraphStore
from _graph_data import KARATE_NUM_NODES, karate_edgelist

torch = import_optional("torch")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_graph_store_basic_api(single_pytorch_worker):
    src, dst = karate_edgelist(torch)

    ei = torch.stack([dst, src])

    graph_store = GraphStore()
    graph_store.put_edge_index(
        ei,
        ("person", "knows", "person"),
        "coo",
        False,
        (KARATE_NUM_NODES, KARATE_NUM_NODES),
    )

    rei = graph_store.get_edge_index(("person", "knows", "person"), "coo")

    assert (ei == rei).all()

    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 1

    graph_store.remove_edge_index(("person", "knows", "person"), "coo")
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 0
