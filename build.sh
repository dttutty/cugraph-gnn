#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cugraph build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)

LIBWHOLEGRAPH_BUILD_DIR=${LIBWHOLEGRAPH_BUILD_DIR:=${REPODIR}/cpp/build}
PYLIBWHOLEGRAPH_BUILD_DIR=${PYLIBWHOLEGRAPH_BUILD_DIR:=${REPODIR}/python/pylibwholegraph/build}
WHOLEGRAPH_TORCH_BUILD_DIR=${WHOLEGRAPH_TORCH_BUILD_DIR:=${REPODIR}/python/wholegraph-torch/build}
CUGRAPH_PYG_BUILD_DIR=${CUGRAPH_PYG_BUILD_DIR:=${REPODIR}/python/cugraph-pyg/build}

# Valid args to this script (all possible targets and options) - only one per line
VALIDARGS="
   clean
   uninstall
   libwholegraph
   pylibwholegraph
   wholegraph-torch
   cugraph-pyg
   wholegraph-tests
   wholegraph-benchmarks
   all
   -v
   -g
   -n
   --pydevelop
   --allgpuarch
   --compile-cmd
   --cmake_default_generator
   --clean
   -h
   --help
"

HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean                      - remove all existing build artifacts and configuration (start over)
   uninstall                  - uninstall GNN-related cuGraph packages from a prior build/install (see also -n)
   libwholegraph              - build libwholegraph.so
   pylibwholegraph            - build the pylibwholegraph Python package
   wholegraph-torch           - build the WholeGraph PyTorch adapter package
   cugraph-pyg                - build the cugraph-pyg Python package
   wholegraph-tests           - build libwholegraph C++ tests
   wholegraph-benchmarks      - build libwholegraph C++ benchmarks
   all                        - build everything
 and <flag> is:
   -v                         - verbose build mode
   -g                         - build for debug
   -n                         - do not install after a successful build (does not affect Python packages)
   --pydevelop                - install the Python packages in editable mode
   --allgpuarch               - build for all supported GPU architectures
   --compile-cmd              - configure WholeGraph targets without building them
   --cmake_default_generator  - use the default cmake generator instead of ninja
   --clean                    - clean an individual target (note: to do a complete rebuild, use the clean target described above)
   -h                         - print this text

default action (no args) is to build and install the local GNN stack:
'libwholegraph', 'pylibwholegraph', 'wholegraph-torch', then 'cugraph-pyg'

 libwholegraph build dir is: ${LIBWHOLEGRAPH_BUILD_DIR}

Set env var LIBWHOLEGRAPH_BUILD_DIR to override libwholegraph build dir.
"
BUILD_DIRS="${LIBWHOLEGRAPH_BUILD_DIR}
            ${PYLIBWHOLEGRAPH_BUILD_DIR}
            ${WHOLEGRAPH_TORCH_BUILD_DIR}
            ${CUGRAPH_PYG_BUILD_DIR}
"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
CMAKE_VERBOSE_OPTION=()
BUILD_TYPE=Release
INSTALL_TARGET=(--target install)
BUILD_WHOLEGRAPH_TESTS=OFF
BUILD_WHOLEGRAPH_BENCHMARKS=OFF
BUILD_ALL_GPU_ARCH=0
CMAKE_GENERATOR_OPTION=(-G Ninja)
PYTHON_ARGS_FOR_INSTALL=(
   --no-build-isolation
   --no-deps
   --config-settings="rapidsai.disable-cuda=true"
)

# Set defaults for vars that may not have been defined externally
#  FIXME: if PREFIX is not set, check CONDA_PREFIX, but there is no fallback
#  from there!
INSTALL_PREFIX=${PREFIX:=${CONDA_PREFIX}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}
BUILD_ABI=${BUILD_ABI:=ON}

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function buildDefault {
    (( NUMARGS == 0 )) || ! (echo " ${ARGS} " | grep -q " [^-][a-zA-Z0-9\_\-]\+ ")
}

function cleanPythonDir {
    pushd "$1" > /dev/null
    rm -rf dist dask-worker-space cugraph/raft ./*.egg-info
    find . -type d -name __pycache__ -print0 | xargs -0 rm -rf
    find . -type d -name build -print0 | xargs -0 rm -rf
    find . -type d -name dist -print0 | xargs -0 rm -rf
    find . -type f -name "*.cpython*.so" -delete
    find . -type d -name _external_repositories -print0 | xargs -0 rm -rf
    popd > /dev/null
}

if (( NUMARGS == 0 )); then
    ARGS="libwholegraph pylibwholegraph wholegraph-torch cugraph-pyg"
    NUMARGS=4
fi

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    for a in ${ARGS}; do
        if ! (echo "${VALIDARGS}" | grep -q "^[[:blank:]]*${a}$"); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG="-v"
    CMAKE_VERBOSE_OPTION=(--log-level=VERBOSE)
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=()
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg wholegraph-tests; then
    BUILD_WHOLEGRAPH_TESTS=ON
fi
if hasArg wholegraph-benchmarks; then
    BUILD_WHOLEGRAPH_BENCHMARKS=ON
fi
if hasArg --cmake_default_generator; then
    CMAKE_GENERATOR_OPTION=()
fi
if hasArg --pydevelop; then
    PYTHON_ARGS_FOR_INSTALL+=(-e)
fi

SKBUILD_EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS}"

# Replace spaces with semicolons in SKBUILD_EXTRA_CMAKE_ARGS
SKBUILD_EXTRA_CMAKE_ARGS=${SKBUILD_EXTRA_CMAKE_ARGS// /;}

# If clean or uninstall targets given, run them prior to any other steps
if hasArg uninstall; then
    if [[ "$INSTALL_PREFIX" != "" ]]; then
        rm -rf "${INSTALL_PREFIX}/include/wholegraph"
        rm -f "${INSTALL_PREFIX}/lib/libwholegraph.so"
        rm -rf "${INSTALL_PREFIX}/lib/cmake/wholegraph"
    fi
    # This may be redundant given the above, but can also be used in case
    # there are other installed files outside of the locations above.
    if [ -e "${LIBWHOLEGRAPH_BUILD_DIR}/install_manifest.txt" ]; then
        xargs rm -f < "${LIBWHOLEGRAPH_BUILD_DIR}/install_manifest.txt" > /dev/null 2>&1
    fi
    # uninstall GNN-related cuGraph and WholeGraph packages installed from a prior install
    # FIXME: if multiple versions of these packages are installed, this only
    # removes the latest one and leaves the others installed. build.sh uninstall
    # can be run multiple times to remove all of them, but that is not obvious.
    pip uninstall -y libwholegraph pylibwholegraph wholegraph-torch cugraph-pyg
fi

if hasArg clean; then
    # Ignore errors for clean since missing files, etc. are not failures
    set +e
    # remove artifacts generated inplace
    if [[ -d ${REPODIR}/python ]]; then
        cleanPythonDir "${REPODIR}/python"
    fi

    # If the dirs to clean are mounted dirs in a container, the contents should
    # be removed but the mounted dirs will remain.  The find removes all
    # contents but leaves the dirs, the rmdir attempts to remove the dirs but
    # can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
        fi
    done
    # Go back to failing on first error for all other operations
    set -e
fi

################################################################################
# Configure, build, and install libwholegraph
if buildDefault || hasArg libwholegraph || hasArg wholegraph-tests || hasArg wholegraph-benchmarks || hasArg all; then
    if hasArg --clean; then
        if [ -d "${LIBWHOLEGRAPH_BUILD_DIR}" ]; then
            find "${LIBWHOLEGRAPH_BUILD_DIR}" -mindepth 1 -delete
            rmdir "${LIBWHOLEGRAPH_BUILD_DIR}" || true
        fi
    else
        if (( BUILD_ALL_GPU_ARCH == 0 )); then
            WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="NATIVE"
            echo "Building WholeGraph for the architecture of the GPU in the system..."
        else
            WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
            echo "Building WholeGraph for *ALL* supported GPU architectures..."
        fi
        mkdir -p "${LIBWHOLEGRAPH_BUILD_DIR}"
        cd "${LIBWHOLEGRAPH_BUILD_DIR}"
        cmake -B "${LIBWHOLEGRAPH_BUILD_DIR}" -S "${REPODIR}/cpp" \
              -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
              -DCMAKE_CUDA_ARCHITECTURES="${WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES}" \
              -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
              -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
              -DBUILD_TESTS=${BUILD_WHOLEGRAPH_TESTS} \
              -DBUILD_BENCHMARKS=${BUILD_WHOLEGRAPH_BENCHMARKS} \
              "${CMAKE_GENERATOR_OPTION[@]}" \
              "${CMAKE_VERBOSE_OPTION[@]}" \
              ${EXTRA_CMAKE_ARGS}

        if ! hasArg --compile-cmd; then
            cmake --build "${LIBWHOLEGRAPH_BUILD_DIR}" "-j${PARALLEL_LEVEL}" "${INSTALL_TARGET[@]}" "${VERBOSE_FLAG}"
        fi
    fi
fi

# If `RAPIDS_PY_VERSION` is set, use that as the lower-bound for the stable ABI CPython version
if [ -n "${RAPIDS_PY_VERSION:-}" ]; then
    RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
    PYTHON_ARGS_FOR_INSTALL+=("--config-settings" "skbuild.wheel.py-api=${RAPIDS_PY_API}")
fi

# Build and install pylibwholegraph
if buildDefault || hasArg pylibwholegraph || hasArg all; then
    if hasArg --clean; then
        cleanPythonDir "${REPODIR}/python/pylibwholegraph"
    else
        # setup.py and cmake reference LIBWHOLEGRAPH_DIR to find the libwholegraph package.
        LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR:=${LIBWHOLEGRAPH_BUILD_DIR}}
        if ! hasArg --compile-cmd; then
            LIBWHOLEGRAPH_DIR="${LIBWHOLEGRAPH_DIR}" \
            SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
                python -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" "${REPODIR}/python/pylibwholegraph"
        else
            LIBWHOLEGRAPH_DIR="${LIBWHOLEGRAPH_DIR}" \
                cmake -S "${REPODIR}/python/pylibwholegraph" -B "${PYLIBWHOLEGRAPH_BUILD_DIR}" \
                      -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
        fi
    fi
fi

# Build and install the WholeGraph PyTorch adapter Python package
if buildDefault || hasArg wholegraph-torch || hasArg all; then
    if hasArg --clean; then
        cleanPythonDir "${REPODIR}/python/wholegraph-torch"
    else
        python -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" "${REPODIR}/python/wholegraph-torch"
    fi
fi

# Build and install the cugraph-pyg Python package
if buildDefault || hasArg cugraph-pyg || hasArg all; then
    if hasArg --clean; then
        cleanPythonDir "${REPODIR}/python/cugraph-pyg"
    else
        python -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" "${REPODIR}/python/cugraph-pyg"
    fi
fi
