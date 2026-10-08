#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs INSIDE a conda container. Installs a single cuOpt component conda package into a fresh env,
# builds the sample gtest against it, runs it.
#
# Env:
#   COMPONENT      client|mathopt|routing (required)
#   CUDA_MAJOR     default 13
#   LOCAL_CHANNEL  optional, a local conda channel holding the freshly built packages (what PR CI
#                  uses). When unset, the rapidsai-nightly channel is used.
set -euo pipefail

: "${COMPONENT:?}"
CUDA_MAJOR="${CUDA_MAJOR:-13}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"

CHANNELS=()
[[ -n "${LOCAL_CHANNEL:-}" ]] && CHANNELS+=(-c "${LOCAL_CHANNEL}")
CHANNELS+=(-c rapidsai-nightly -c rapidsai -c conda-forge)

case "${COMPONENT}" in
  client)  PKG="libcuopt-client";  EXPECTED="libcuopt-client" ;;
  mathopt) PKG="libcuopt-mathopt"; EXPECTED="libcuopt-client libcuopt-mathopt" ;;
  routing) PKG="libcuopt-routing"; EXPECTED="libcuopt-client libcuopt-routing" ;;
  *) echo "bad COMPONENT=${COMPONENT}" >&2; exit 2 ;;
esac

# Build tooling is a test fixture, kept separate from the package under test so the
# cuopt package set in the env stays exactly what a user would get.
TOOLS=(cmake make gtest cxx-compiler)
[[ "${COMPONENT}" != "client" ]] && TOOLS+=("cuda-nvcc" "cuda-cudart-dev" "cuda-version=${CUDA_MAJOR}")

ENV_NAME="isolated_${COMPONENT}_$$"
CREATE=(conda create)
command -v rapids-mamba-retry >/dev/null 2>&1 && CREATE=(rapids-mamba-retry create)
"${CREATE[@]}" -y -q -n "${ENV_NAME}" --override-channels "${CHANNELS[@]}" "${PKG}" "${TOOLS[@]}"

# shellcheck disable=SC1091
. "${CONDA_ROOT}/etc/profile.d/conda.sh"
set +u  # conda activate scripts reference unset variables
conda activate "${ENV_NAME}"
set -u

echo "== installed cuopt packages"
INSTALLED="$(conda list --json | python3 -c 'import json,sys; print(" ".join(sorted(p["name"] for p in json.load(sys.stdin) if p["name"].startswith(("libcuopt","cuopt")))))')"
echo "${INSTALLED}"
WANT="$(echo "${EXPECTED}" | tr ' ' '\n' | sort | xargs)"
if [[ "${INSTALLED}" != "${WANT}" ]]; then
  echo "FAIL: expected exactly [${WANT}] but found [${INSTALLED}]" >&2
  exit 1
fi

WORK="$(mktemp -d)"
cmake -S "${HERE}/tests" -B "${WORK}/build" -DCOMPONENT="${COMPONENT}" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"
cmake --build "${WORK}/build" -j"$(nproc)"
"${WORK}/build/isolated_${COMPONENT}_test"
