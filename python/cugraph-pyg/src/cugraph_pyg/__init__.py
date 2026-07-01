# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cugraph_pyg._version import __git_commit__, __version__

from importlib import import_module

__all__ = ["__git_commit__", "__version__", "data", "loader", "sampler", "tensor"]


def __getattr__(name):
    if name in {"data", "loader", "sampler", "tensor"}:
        module = import_module(f"cugraph_pyg.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
