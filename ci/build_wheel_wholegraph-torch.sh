#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_dir="python/wholegraph-torch"

./ci/build_wheel.sh wholegraph-torch ${package_dir}
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python wholegraph-torch cugraph --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
