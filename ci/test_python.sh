#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libwholegraph cugraph --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBWHOLEGRAPH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python pylibwholegraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
WHOLEGRAPH_TORCH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python wholegraph-torch cugraph --pure --arch any)")
CUGRAPH_PYG_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cugraph-pyg cugraph --pure --arch any)")

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "(pylibwholegraph) Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_pylibwholegraph \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES};require_gpu=true" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYLIBWHOLEGRAPH_CHANNEL}" \
| tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test_pylibwholegraph

set +u
conda activate test_pylibwholegraph
set -u

rapids-print-env
nvidia-smi

rapids-logger "pytest pylibwholegraph"
./ci/run_pylibwholegraph_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-pylibwholegraph.xml" \
  --cov-config=../../.coveragerc \
  --cov=pylibwholegraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/pylibwholegraph-coverage.xml" \
  --cov-report=term

set +u
conda deactivate
set -u

rapids-logger "(wholegraph-torch) Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_wholegraph_torch \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES};require_gpu=true" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYLIBWHOLEGRAPH_CHANNEL}" \
  --prepend-channel "${WHOLEGRAPH_TORCH_CHANNEL}" \
| tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test_wholegraph_torch

set +u
conda activate test_wholegraph_torch
set -u

rapids-logger "Confirming that PyTorch is installed"
python -c "import torch; assert torch.cuda.is_available()"

rapids-print-env
nvidia-smi

rapids-logger "pytest wholegraph-torch"
./ci/run_wholegraph_torch_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-wholegraph-torch.xml" \
  --cov-config=../../.coveragerc \
  --cov=wholegraph_torch \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/wholegraph-torch-coverage.xml" \
  --cov-report=term

set +u
conda deactivate
set -u

rapids-logger "(cugraph-pyg) Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cugraph_pyg \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES};require_gpu=true" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYLIBWHOLEGRAPH_CHANNEL}" \
  --prepend-channel "${WHOLEGRAPH_TORCH_CHANNEL}" \
  --prepend-channel "${CUGRAPH_PYG_CHANNEL}" \
| tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test_cugraph_pyg

set +u
conda activate test_cugraph_pyg
set -u

rapids-logger "Confirming that PyTorch is installed"
python -c "import torch; assert torch.cuda.is_available()"

rapids-print-env
nvidia-smi

rapids-logger "pytest cugraph-pyg"
./ci/run_cugraph_pyg_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph-pyg.xml" \
  --cov-config=../../.coveragerc \
  --cov=cugraph_pyg \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-pyg-coverage.xml" \
  --cov-report=term

set +u
conda deactivate
set -u

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
