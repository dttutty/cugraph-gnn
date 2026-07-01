#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-date-string

rapids-print-env

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libwholegraph cugraph --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBWHOLEGRAPH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python pylibwholegraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-generate-version > ./VERSION

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

source rapids-rattler-channel-string

RATTLER_CHANNELS=(
  "--channel" "${CPP_CHANNEL}"
  "--channel" "${PYLIBWHOLEGRAPH_CHANNEL}"
  "${RATTLER_CHANNELS[@]}"
)

rapids-logger "Begin wholegraph-torch conda build"

rattler-build build --recipe conda/recipes/wholegraph-torch \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python wholegraph-torch cugraph --pure --arch any)"
export RAPIDS_PACKAGE_NAME
