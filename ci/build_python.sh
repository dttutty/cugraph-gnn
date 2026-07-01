#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libwholegraph cugraph --cuda "$RAPIDS_CUDA_VERSION")")

rapids-generate-version > ./VERSION

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

source rapids-rattler-channel-string

RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "${RATTLER_CHANNELS[@]}")

rapids-logger "Begin pylibwholegraph conda build"

sccache --stop-server 2>/dev/null || true

rattler-build build --recipe conda/recipes/pylibwholegraph \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python pylibwholegraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
