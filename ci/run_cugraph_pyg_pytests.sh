#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_cugraph_pyg_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph-pyg

pytest -rs --cache-clear --benchmark-disable "$@" tests

# Used to skip certain examples in CI due to memory limitations
export CI=true

# Enable legacy behavior of torch.load for examples relying on ogb
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Test examples (disabled due to lack of memory)
#for e in "$(pwd)"/examples/*.py; do
#  echo "running example $e"
#  (yes || true) | torchrun --nnodes 1 --nproc_per_node 1 $e --dataset_root "./ogb_datasets"
#done

# echo "running bitcoin example"
# (yes || true) | torchrun --nnodes 1 --nproc_per_node 1 "$(pwd)"/examples/fraud/bitcoin_mnmg.py --dataset_root "./bitcoin" --embedding_dir "./bitcoin_embeddings"
