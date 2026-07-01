#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Delete system libnccl.so to ensure the wheel is used.
# (but only do this in CI, to avoid breaking local dev environments)
if [[ "${CI:-}" == "true" ]]; then
  rm -rf /usr/lib64/libnccl*
fi

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

LIBWHOLEGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libwholegraph cugraph --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBWHOLEGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibwholegraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
WHOLEGRAPH_TORCH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python wholegraph-torch cugraph --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")")

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-generate-pip-constraints test_wholegraph_torch "${PIP_CONSTRAINT}"

PIP_INSTALL_ARGS=(
  --prefer-binary
  --constraint "${PIP_CONSTRAINT}"
  "${LIBWHOLEGRAPH_WHEELHOUSE}"/*.whl
  "$(echo "${PYLIBWHOLEGRAPH_WHEELHOUSE}"/pylibwholegraph_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"
  "$(echo "${WHOLEGRAPH_TORCH_WHEELHOUSE}"/wholegraph_torch_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"
)

TORCH_WHEEL_DIR="$(mktemp -d)"
./ci/download-torch-wheels.sh "${TORCH_WHEEL_DIR}"

torch_downloaded=true
if [ -z "$(ls -A "${TORCH_WHEEL_DIR}" 2>/dev/null)" ]; then
  rapids-echo-stderr "No 'torch' wheels downloaded."
  torch_downloaded=false
else
  PIP_INSTALL_ARGS+=("${TORCH_WHEEL_DIR}"/torch-*.whl)
fi

rapids-logger "Installing Packages"
rapids-pip-retry install \
    "${PIP_INSTALL_ARGS[@]}"

if [[ "${torch_downloaded}" == "true" ]]; then
  rapids-logger "Confirming that PyTorch is installed"
  python -c "import torch; assert torch.cuda.is_available()"

  rapids-logger "pytest wholegraph-torch (with 'torch')"
  ./ci/run_wholegraph_torch_pytests.sh \
    --cov-config=../../.coveragerc \
    --cov=wholegraph_torch \
    --cov-fail-under=15
fi

rapids-logger "import wholegraph_torch (no 'torch')"
./ci/uninstall-torch-wheels.sh

python -c "import wholegraph_torch; print('wholegraph_torch imported')"
