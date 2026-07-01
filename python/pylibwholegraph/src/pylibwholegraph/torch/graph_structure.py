# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from ._compat import install_module_alias

install_module_alias(globals(), __name__, "wholegraph_torch.graph_structure")
