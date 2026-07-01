# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from wholegraph_torch.distributed_matrix import DistMatrix

warnings.warn(
    "cugraph_pyg.tensor.dist_matrix is deprecated; "
    "use wholegraph_torch.distributed_matrix instead.",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["DistMatrix"]
