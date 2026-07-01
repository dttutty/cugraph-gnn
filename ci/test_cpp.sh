#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libwholegraph cugraph --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_wholegraph_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Run libwholegraph tests"
./ci/run_ctests.sh && EXITCODE=$? || EXITCODE=$?

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
