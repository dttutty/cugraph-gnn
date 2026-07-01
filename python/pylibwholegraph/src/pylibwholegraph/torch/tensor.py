# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from ._compat import install_module_alias

install_module_alias(
    globals(),
    __name__,
    "wholegraph_torch.tensor",
    {
        "WholeMemoryTensor": "WholeGraphTensor",
        "create_wholememory_tensor": "create_wholegraph_tensor",
        "create_wholememory_tensor_from_filelist": (
            "create_wholegraph_tensor_from_filelist"
        ),
        "destroy_wholememory_tensor": "destroy_wholegraph_tensor",
    },
)
