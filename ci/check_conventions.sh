#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Warn about convention violations on the lines a change ADDS (never on existing code).
# The rules and their IDs live in skills/cuopt-developer/references/review_rules.md.
#
# Usage: ci/check_conventions.sh [--strict] [base_ref]
#   base_ref   ref to diff against (default: origin/main); changes are base_ref...HEAD
#              plus uncommitted changes to tracked files
#   --strict   exit non-zero when anything is reported (default: warn only, exit 0)
#
# Suppress one finding by putting "conventions-ok: <RULE-ID>" on the same line.

set -euo pipefail

STRICT=0
BASE="origin/main"
for arg in "$@"; do
  case "${arg}" in
    --strict) STRICT=1 ;;
    *) BASE="${arg}" ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

MERGE_BASE="$(git merge-base "${BASE}" HEAD)"

# Files that describe the rules and so quote the patterns they forbid.
SKIP_RE='^(skills/cuopt-developer/references/review_rules\.md|ci/check_conventions\.sh|\.coderabbit\.yaml|\.github/\.coderabbit_review_guide\.md|thirdparty/|RELEASE-NOTES\.md|CHANGELOG\.md)'

FINDINGS=0

# report <file> <line> <rule-id> <message> <text>
report() {
  if [[ "$5" == *"conventions-ok: $3"* ]]; then
    return
  fi
  printf '%s:%s: [%s] %s\n' "$1" "$2" "$3" "$4"
  FINDINGS=$((FINDINGS + 1))
}

# check <regex> <rule-id> <message> <file-regex> <comment-only: 0|1>
# Reads "file<TAB>line<TAB>text" records of added lines from ADDED.
check() {
  local regex="$1" rule="$2" msg="$3" files="$4" comment_only="$5"
  local file line text
  while IFS=$'\t' read -r file line text; do
    [[ "${file}" =~ ${files} ]] || continue
    if [[ "${comment_only}" == 1 ]] && ! [[ "${text}" =~ (//|#|/\*|^[[:space:]]*\*) ]]; then
      continue
    fi
    if [[ "${text}" =~ ${regex} ]]; then
      report "${file}" "${line}" "${rule}" "${msg}" "${text}"
    fi
  done <<< "${ADDED}"
}

# Added lines as "file<TAB>line<TAB>text".
ADDED="$(git diff -U0 --no-color --diff-filter=AM "${MERGE_BASE}" -- . \
  | awk -v skip="${SKIP_RE}" '
      /^\+\+\+ b\// { file = substr($0, 7); next }
      /^@@/ {
        match($0, /\+[0-9]+/); line = substr($0, RSTART + 1, RLENGTH - 1) + 0; next
      }
      /^\+/ && !/^\+\+\+/ {
        if (file !~ skip) { printf "%s\t%d\t%s\n", file, line, substr($0, 2) }
        line++
      }')"

CODE='\.(cpp|hpp|cu|cuh|h|pyx|pxd|py|java)$'
CPP='\.(cpp|hpp|cu|cuh|h)$'
SRC_CPP='^cpp/src/.*\.(cpp|hpp|cu|cuh|h)$'
DOCS='\.(rst|md)$'

check 'used to be|previously|earlier (version|revision)|load[- ]bearing|no longer (needed|used) (because|since)' \
  COMMENT-1 'comment narrates history; describe the current code' "${CODE}" 1
check '(PR|pull request) ?#[0-9]+|[(,;] ?(see )?lines? [0-9]+' \
  COMMENT-2 'comment has a PR number or line number; use a stable name instead' "${CODE}" 1
check '\bIPM\b' TERM-2 'say "barrier", not "IPM"' "${CODE}|${DOCS}" 0
check 'stationary_residual' TERM-2 'say "dual residual"' "${CPP}" 0
check '\bMILP\b' TERM-2 'say "MIP", not "MILP"' "${DOCS}|\.java$" 0
check '\[ ?static_cast<(std::)?size_t>\(' TYPE-2 'no static_cast<size_t> in a subscript' "${CPP}" 0
check 'std::shared_ptr|std::make_shared' TYPE-3 'prefer unique_ptr; explain why sharing is essential' "${CPP}" 0
check 'std::(j?thread)\b|pthread_create' THREAD-1 'use OpenMP, not another threading model' "${SRC_CPP}" 0
check 'std::mt19937' THREAD-1 'use pcgenerator_t' "${SRC_CPP}" 0

# PR-1: build output must not be committed.
while IFS= read -r path; do
  [[ -n "${path}" ]] || continue
  if [[ "${path}" =~ (^|/)target/ ]] || [[ "${path}" =~ \.class$|hs_err.*\.log$ ]]; then
    report "${path}" 1 PR-1 'build output added to the repository' ''
  fi
done < <(git diff --name-only --diff-filter=A "${MERGE_BASE}" -- .)

if [[ "${FINDINGS}" -gt 0 ]]; then
  echo "${FINDINGS} convention finding(s). Rules: skills/cuopt-developer/references/review_rules.md"
  [[ "${STRICT}" == 1 ]] && exit 1
else
  echo "No convention findings on added lines."
fi
exit 0
