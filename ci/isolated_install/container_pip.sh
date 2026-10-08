#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs INSIDE the container (Ubuntu + CUDA devel image). Installs a single nightly
# cuOpt component wheel into a clean venv, builds the sample gtest against it, runs it.
# Env: COMPONENT (client|mathopt|routing), CUDA_MAJOR (default 13)
set -euo pipefail

: "${COMPONENT:?}"
CUDA_MAJOR="${CUDA_MAJOR:-13}"
INDEX="${NIGHTLY_WHEEL_INDEX:-https://pypi.anaconda.org/rapidsai-wheels-nightly/simple}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${COMPONENT}" in
  client)  PKG="libcuopt-client"; EXPECTED="libcuopt-client" ;;
  mathopt) PKG="libcuopt-mathopt-cu${CUDA_MAJOR}"; EXPECTED="libcuopt-client libcuopt-mathopt-cu${CUDA_MAJOR}" ;;
  routing) PKG="libcuopt-routing-cu${CUDA_MAJOR}"; EXPECTED="libcuopt-client libcuopt-routing-cu${CUDA_MAJOR}" ;;
  *) echo "bad COMPONENT=${COMPONENT}" >&2; exit 2 ;;
esac

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3-venv python3-pip g++ make libgtest-dev ca-certificates >/dev/null

python3 -m venv /tmp/venv
# shellcheck disable=SC1091
. /tmp/venv/bin/activate
pip install -q cmake   # build tool for the sample test, not part of the package under test
pip install -q --pre --extra-index-url "${INDEX}" "${PKG}"

echo "== installed cuopt packages"
INSTALLED="$(pip list --format=freeze | grep -i '^libcuopt\|^cuopt' | cut -d= -f1 | tr '[:upper:]_' '[:lower:]-' | sort | xargs)"
echo "${INSTALLED}"
WANT="$(echo "${EXPECTED}" | tr ' ' '\n' | sort | xargs)"
if [[ "${INSTALLED}" != "${WANT}" ]]; then
  echo "FAIL: expected exactly [${WANT}] but found [${INSTALLED}]" >&2
  exit 1
fi

SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# Wheels install headers + CMake configs under <pkg>/lib64/cmake/<name>/. Third-party configs
# (rmm, raft, ...) are only found via their config directory; cuopt's config finds its sibling
# component targets (mathopt/routing) through each wheel's root, so pass both.
ROOTS="$(find "${SP}" -mindepth 3 -maxdepth 3 -path '*/lib64/cmake' -printf '%h\n' | xargs -r -n1 dirname | sort -u | tr '\n' ';')"
CONFIG_DIRS="$(find "${SP}" \( -name '*-config.cmake' -o -name '*Config.cmake' \) -printf '%h\n' | sort -u | tr '\n' ';')"
PREFIXES="${ROOTS}${CONFIG_DIRS}"
CUDA_LIBS="$(find "${SP}/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':' || true)"
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:-}"

cmake -S "${HERE}/tests" -B /tmp/build -DCOMPONENT="${COMPONENT}" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${PREFIXES}"
cmake --build /tmp/build -j"$(nproc)"
"/tmp/build/isolated_${COMPONENT}_test"
