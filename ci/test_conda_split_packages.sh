#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs each libcuopt-* conda package on its own, in a fresh environment, and builds and runs a
# C++ binary against it, to catch a package that only works because another one happened to be
# installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPONENTS="client mathopt routing"

# shellcheck disable=SC1091
. "$(conda info --base)/etc/profile.d/conda.sh"

conda config --set channel_priority strict
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
RAPIDS_CHANNEL=rapidsai-nightly
rapids-is-release-build && RAPIDS_CHANNEL=rapidsai
CUDA_VERSION=${RAPIDS_CUDA_VERSION%.*}

FAILED=()
for component in ${COMPONENTS}; do
  echo "=== conda: libcuopt-${component} alone ==="
  package="libcuopt-${component}"
  expected="libcuopt-client ${package}"
  [[ "${component}" == "client" ]] && expected="libcuopt-client"

  env_dir="$(mktemp -d)/env"
  (
    rapids-mamba-retry create --yes --prefix "${env_dir}" \
      --channel "${CPP_CHANNEL}" --channel "${RAPIDS_CHANNEL}" --channel conda-forge --channel nvidia \
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
