# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from wholegraph_torch.distributed_tensor_utils import (
    copy_host_global_tensor_to_local,
    create_wg_dist_tensor,
    create_wg_dist_tensor_from_files,
    empty,
    is_empty,
)
from wholegraph_torch.storage_backend import has_mapped_storage_support

warnings.warn(
    "cugraph_pyg.tensor.utils is deprecated; "
    "use wholegraph_torch.distributed_tensor_utils instead.",
    FutureWarning,
    stacklevel=2,
)


def has_nvlink_network():
    return has_mapped_storage_support()


__all__ = [
    "copy_host_global_tensor_to_local",
    "create_wg_dist_tensor",
    "create_wg_dist_tensor_from_files",
    "empty",
    "has_nvlink_network",
    "is_empty",
]
