#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs each libcuopt-* conda package on its own, in a fresh environment, and builds and runs a
# C++ binary against it, to catch a package that only works because another one happened to be
# installed.
#
# CUOPT_SMOKE_CHANNELS replaces the downloaded channel, e.g. to test the published nightlies.
# CUOPT_SMOKE_CUDA_VERSION is the CUDA major.minor to use when RAPIDS_CUDA_VERSION is not set.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPONENTS=${COMPONENTS:-"client mathopt routing"}

# shellcheck disable=SC1091
. "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -n "${CUOPT_SMOKE_CHANNELS:-}" ]]; then
  read -ra CHANNEL_ARGS <<< "${CUOPT_SMOKE_CHANNELS}"
  CUDA_VERSION=${CUOPT_SMOKE_CUDA_VERSION:?set CUOPT_SMOKE_CUDA_VERSION, e.g. 13.3}
  create_env() { mamba create --yes "$@"; }
else
  conda config --set channel_priority strict
  CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
  RAPIDS_CHANNEL=rapidsai-nightly
  rapids-is-release-build && RAPIDS_CHANNEL=rapidsai
  CHANNEL_ARGS=(--channel "${CPP_CHANNEL}" --channel "${RAPIDS_CHANNEL}" --channel conda-forge --channel nvidia)
  CUDA_VERSION=${RAPIDS_CUDA_VERSION%.*}
  create_env() { rapids-mamba-retry create --yes "$@"; }
fi

FAILED=()
for component in ${COMPONENTS}; do
  echo "=== conda: libcuopt-${component} alone ==="
  package="libcuopt-${component}"
  expected="libcuopt-client ${package}"
  [[ "${component}" == "client" ]] && expected="libcuopt-client"

  env_dir="$(mktemp -d)/env"
  (
    create_env --prefix "${env_dir}" "${CHANNEL_ARGS[@]}" \
      "${package}" "cuda-version=${CUDA_VERSION}" \
      cmake ninja cxx-compiler cuda-nvcc cuda-cudart-dev

    installed=$(conda list --prefix "${env_dir}" --json | python -c \
      "import json,sys; print(' '.join(sorted(p['name'] for p in json.load(sys.stdin) if p['name'].startswith(('libcuopt','cuopt')))))")
    wanted=$(echo "${expected}" | tr ' ' '\n' | sort | tr '\n' ' ' | sed 's/ $//')
    if [[ "${installed}" != "${wanted}" ]]; then
      echo "installing ${package} pulled in: ${installed} (expected: ${wanted})" >&2
      exit 1
    fi

    set +u
    conda activate "${env_dir}"
    set -u
    bash "${REPO_ROOT}/ci/split_package_smoke/build_and_run.sh" "${component}" "${env_dir}"
  ) || FAILED+=("${component}")
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED conda components: ${FAILED[*]}" >&2
  exit 1
fi
