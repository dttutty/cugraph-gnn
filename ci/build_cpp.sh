#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin libwholegraph conda build"

sccache --stop-server 2>/dev/null || true

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

source rapids-rattler-channel-string

rattler-build build --recipe conda/recipes/libwholegraph \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_cpp libwholegraph cugraph --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
