# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import pylibwholegraph.binding.wholegraph_binding as wgb


def _create_context_args(global_context=None):
    return [
        lambda _global_context: object(),
        lambda _memory_context, _global_context: None,
        lambda _shape, _dtype, _malloc_type, _memory_context, _global_context: 0,
        lambda _memory_context, _global_context: None,
        global_context,
        lambda _shape, _dtype, _malloc_type, _memory_context, _global_context: 0,
        lambda _memory_context, _global_context: None,
        global_context,
    ]


@pytest.mark.parametrize("callback_index", [0, 1, 2, 3, 5, 6])
def test_global_context_wrapper_rejects_non_callable_callbacks(callback_index):
    args = _create_context_args()
    args[callback_index] = object()

    with pytest.raises(ValueError, match="callbacks must be callable"):
        wgb.GlobalContextWrapper().create_context(*args)


def test_global_context_wrapper_exports_env_function_pointer():
    context = wgb.GlobalContextWrapper()

    context.create_context(*_create_context_args())

    assert context.get_env_fns() > 0
