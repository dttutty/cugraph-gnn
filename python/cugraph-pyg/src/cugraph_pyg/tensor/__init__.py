# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from wholegraph_torch.distributed_matrix import DistMatrix
from wholegraph_torch.distributed_tensor import DistEmbedding, DistTensor
from wholegraph_torch.distributed_tensor_utils import empty, is_empty

warnings.warn(
    "cugraph_pyg.tensor is deprecated; use wholegraph_torch instead.",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["DistEmbedding", "DistMatrix", "DistTensor", "empty", "is_empty"]
