# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
import warnings

_NAME_REPLACEMENTS = (
    ("WholeMemory", "WholeGraph"),
    ("wholememory", "wholegraph"),
    ("wm_", "wg_"),
    ("_wm_", "_wg_"),
)


def install_module_alias(namespace, old_module, new_module, explicit_aliases=None):
    module = import_module(new_module)
    explicit_aliases = explicit_aliases or {}

    warnings.warn(
        f"{old_module} is deprecated; use {new_module} instead.",
        FutureWarning,
        stacklevel=3,
    )

    exported = [name for name in dir(module) if not name.startswith("_")]
    namespace.update({name: getattr(module, name) for name in exported})

    for old_name, new_name in explicit_aliases.items():
        if hasattr(module, new_name):
            namespace[old_name] = getattr(module, new_name)

    def __getattr__(name):
        if name in explicit_aliases and hasattr(module, explicit_aliases[name]):
            return getattr(module, explicit_aliases[name])

        for old, new in _NAME_REPLACEMENTS:
            if old in name:
                candidate = name.replace(old, new)
                if hasattr(module, candidate):
                    return getattr(module, candidate)

        raise AttributeError(f"module {old_module!r} has no attribute {name!r}")

    namespace["__getattr__"] = __getattr__
    namespace["__all__"] = sorted(set(exported) | set(explicit_aliases))
