# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import pylibwholegraph.binding.wholegraph_binding as wgb


def test_tensor_description_round_trips_shape_stride_and_storage_offset():
    desc = wgb.PyWholeGraphTensorDescription()

    desc.set_dtype(wgb.WholeGraphDataType.DtInt64)
    desc.set_shape([2, 3])
    desc.set_stride([3, 1])
    desc.set_storage_offset(5)

    assert desc.dtype == wgb.WholeGraphDataType.DtInt64
    assert desc.dim() == 2
    assert desc.shape == (2, 3)
    assert desc.stride() == (3, 1)
    assert desc.storage_offset() == 5


def test_tensor_description_rejects_empty_shape():
    desc = wgb.PyWholeGraphTensorDescription()

    with pytest.raises(ValueError, match="shape length must be in range"):
        desc.set_shape([])


def test_tensor_description_rejects_stride_before_shape():
    desc = wgb.PyWholeGraphTensorDescription()

    with pytest.raises(ValueError, match="shape must be set before stride"):
        desc.set_stride([])


def test_tensor_description_rejects_stride_length_mismatch():
    desc = wgb.PyWholeGraphTensorDescription()
    desc.set_shape([2, 3])

    with pytest.raises(ValueError, match="stride length must match"):
        desc.set_stride([3])
