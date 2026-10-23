#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs each libcuopt-* wheel on its own, in a fresh venv, and builds and runs a C++ binary
# against it, to catch a wheel that only works because another one happened to be installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPONENTS="client mathopt routing"

source rapids-init-pip
bash "${REPO_ROOT}/ci/utils/install_openssl3_runtime.sh"

SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CLIENT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_client cuopt)")
MATHOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_mathopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
ROUTING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_routing cuopt --cuda "$RAPIDS_CUDA_VERSION")")

cat > "${PIP_CONSTRAINT}" <<CONSTRAINTS
libcuopt-client @ file://$(echo "${CLIENT_WHEELHOUSE}"/libcuopt_client-*.whl)
libcuopt-mathopt-${SUFFIX} @ file://$(echo "${MATHOPT_WHEELHOUSE}"/libcuopt_mathopt_*.whl)
libcuopt-routing-${SUFFIX} @ file://$(echo "${ROUTING_WHEELHOUSE}"/libcuopt_routing_*.whl)
CONSTRAINTS

FAILED=()
for component in ${COMPONENTS}; do
  echo "=== wheel: libcuopt-${component} alone ==="

  if [[ "${component}" == "client" ]]; then
    package=libcuopt-client
    expected="libcuopt-client"
  else
    package="libcuopt-${component}-${SUFFIX}"
    expected="libcuopt-client ${package}"
  fi

  venv="$(mktemp -d)/venv"
  python -m venv "${venv}"
  (
    # shellcheck disable=SC1091
    . "${venv}/bin/activate"
    pip install --upgrade pip cmake ninja
    rapids-pip-retry install --prefer-binary --constraint "${PIP_CONSTRAINT}" "${package}"

    installed=$(pip list --format=freeze | sed -E 's/==.*//' | grep -E '^libcuopt' | sort | tr '\n' ' ')
    wanted=$(echo "${expected}" | tr ' ' '\n' | sort | tr '\n' ' ')
    if [[ "${installed}" != "${wanted}" ]]; then
      echo "installing ${package} pulled in: ${installed} (expected: ${wanted})" >&2
      exit 1
    fi

    prefixes=$(python - <<'PY'
import importlib, importlib.metadata as md, os
print(";".join(os.path.dirname(importlib.import_module(ep.value.split(":")[0]).__file__)
               for ep in md.entry_points(group="cmake.prefix")))
PY
)
    bash "${REPO_ROOT}/ci/split_package_smoke/build_and_run.sh" "${component}" "${prefixes}"
  ) || FAILED+=("${component}")
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED wheel components: ${FAILED[*]}" >&2
  exit 1
fi
