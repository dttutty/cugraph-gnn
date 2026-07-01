# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from wholegraph_torch.distributed_tensor import DistEmbedding, DistTensor

warnings.warn(
    "cugraph_pyg.tensor.dist_tensor is deprecated; "
    "use wholegraph_torch.distributed_tensor instead.",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["DistEmbedding", "DistTensor"]
