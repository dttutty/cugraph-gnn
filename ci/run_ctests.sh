#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libwholegraph/"
build_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build"

if [[ -d "${installed_test_location}" ]]; then
    cd "${installed_test_location}"
elif [[ -d "${build_test_location}" ]]; then
    cd "${build_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${build_test_location}" >&2
    exit 1
fi

find . -type f -executable -print0 | xargs -0 -r -t -n1 -P1 sh -c 'exec "$0"'
