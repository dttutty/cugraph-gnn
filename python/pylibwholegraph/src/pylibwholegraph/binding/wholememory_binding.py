# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from . import wholegraph_binding as _wholegraph_binding
from .wholegraph_binding import *  # noqa: F401,F403

warnings.warn(
    "pylibwholegraph.binding.wholememory_binding is deprecated; "
    "use pylibwholegraph.binding.wholegraph_binding instead.",
    FutureWarning,
    stacklevel=2,
)

_REPLACEMENTS = (
    ("WholeMemory", "WholeGraph"),
    ("PyWholeMemory", "PyWholeGraph"),
    ("wholememory", "wholegraph"),
)


def __getattr__(name):
    for old, new in _REPLACEMENTS:
        if old in name:
            candidate = name.replace(old, new)
            if hasattr(_wholegraph_binding, candidate):
                return getattr(_wholegraph_binding, candidate)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [name for name in dir(_wholegraph_binding) if not name.startswith("_")]
