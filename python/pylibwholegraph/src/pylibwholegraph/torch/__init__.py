# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from ._compat import install_module_alias

install_module_alias(
    globals(),
    __name__,
    "wholegraph_torch",
    {
        "WholeMemoryCachePolicy": "WholeGraphCachePolicy",
        "WholeMemoryCommunicator": "WholeGraphCommunicator",
        "WholeMemoryEmbedding": "WholeGraphEmbedding",
        "WholeMemoryEmbeddingModule": "WholeGraphEmbeddingModule",
        "WholeMemoryOptimizer": "WholeGraphOptimizer",
        "WholeMemoryTensor": "WholeGraphTensor",
        "create_wholememory_cache_policy": "create_wholegraph_cache_policy",
        "create_wholememory_optimizer": "create_wholegraph_optimizer",
        "create_wholememory_tensor": "create_wholegraph_tensor",
        "create_wholememory_tensor_from_filelist": (
            "create_wholegraph_tensor_from_filelist"
        ),
        "destroy_wholememory_cache_policy": "destroy_wholegraph_cache_policy",
        "destroy_wholememory_optimizer": "destroy_wholegraph_optimizer",
        "destroy_wholememory_tensor": "destroy_wholegraph_tensor",
        "init_torch_env_and_create_wm_comm": "init_torch_env_and_create_wg_comm",
    },
)
