# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from ._compat import install_module_alias

install_module_alias(
    globals(),
    __name__,
    "wholegraph_torch.embedding",
    {
        "WholeMemoryCachePolicy": "WholeGraphCachePolicy",
        "WholeMemoryEmbedding": "WholeGraphEmbedding",
        "WholeMemoryEmbeddingModule": "WholeGraphEmbeddingModule",
        "WholeMemoryOptimizer": "WholeGraphOptimizer",
        "create_wholememory_cache_policy": "create_wholegraph_cache_policy",
        "create_wholememory_optimizer": "create_wholegraph_optimizer",
        "destroy_wholememory_cache_policy": "destroy_wholegraph_cache_policy",
        "destroy_wholememory_optimizer": "destroy_wholegraph_optimizer",
    },
)
