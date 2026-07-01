# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pylibwholegraph.binding.wholegraph_binding as wgb
from pylibwholegraph.utils.imports import import_optional
from .wholegraph_env import (
    get_stream,
    get_wholegraph_env_fns,
    wrap_torch_tensor,
)
from .utils import wholegraph_dtype_to_torch_dtype

torch = import_optional("torch")


def wholegraph_gather_forward_functor(
    wholegraph_tensor: wgb.PyWholeGraphTensor,
    indices_tensor: "torch.Tensor",
    requires_grad=False,
    torch_output_dtype=None,
):
    """
    Wrapper functor for gather op of WholeGraph tensor
    :param wholegraph_tensor: PyWholeGraphTensor
    :param indices_tensor: Indices to gather from
    :param requires_grad: if requires gradients
    :param torch_output_dtype: output dtype, None for same as wholegraph_tensor
    :return: Gathered tensor
    """
    assert indices_tensor.dim() == 1
    assert indices_tensor.dtype == torch.int32 or indices_tensor.dtype == torch.int64
    if torch_output_dtype is None:
        torch_output_dtype = wholegraph_dtype_to_torch_dtype(wholegraph_tensor.dtype)

    embedding_dim = wholegraph_tensor.shape[1] if wholegraph_tensor.dim() == 2 else 1
    output_tensor = torch.empty(
        [indices_tensor.shape[0], embedding_dim],
        device="cuda",
        dtype=torch_output_dtype,
        requires_grad=requires_grad,
    )
    wgb.wholegraph_gather_op(
        wholegraph_tensor,
        wrap_torch_tensor(indices_tensor),
        wrap_torch_tensor(output_tensor),
        get_wholegraph_env_fns(),
        get_stream(),
    )
    return output_tensor.view(-1) if wholegraph_tensor.dim() == 1 else output_tensor


def wholegraph_scatter_functor(
    input_tensor: "torch.Tensor",
    indices_tensor: "torch.Tensor",
    wholegraph_tensor: wgb.PyWholeGraphTensor,
):
    """
    Wrapper functor for scatter op of WholeGraph tensor
    :param input_tensor: Input tensor to scater to WholeGraph tensor
    :param indices_tensor: Indices to scatter to
    :param wholegraph_tensor: WholeGraph tensor
    :return: None
    """
    assert indices_tensor.dim() == 1
    assert indices_tensor.dtype == torch.int32 or indices_tensor.dtype == torch.int64
    wgb.wholegraph_scatter_op(
        wrap_torch_tensor(input_tensor),
        wrap_torch_tensor(indices_tensor),
        wholegraph_tensor,
        get_wholegraph_env_fns(),
        get_stream(),
    )
