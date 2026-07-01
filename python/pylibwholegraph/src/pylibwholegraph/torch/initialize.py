# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from ._compat import install_module_alias

install_module_alias(
    globals(),
    __name__,
    "wholegraph_torch.initialize",
    {"init_torch_env_and_create_wm_comm": "init_torch_env_and_create_wg_comm"},
)
