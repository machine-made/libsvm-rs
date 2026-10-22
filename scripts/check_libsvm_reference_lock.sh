#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="${PROJ_ROOT}/reference/libsvm_upstream_lock.json"
VENDOR_HEADER="${PROJ_ROOT}/vendor/libsvm/svm.h"

if [[ ! -f "${LOCK_FILE}" ]]; then
    echo "ERROR: lock file not found: ${LOCK_FILE}" >&2
    exit 1
fi
if [[ ! -f "${VENDOR_HEADER}" ]]; then
    echo "ERROR: vendor header not found: ${VENDOR_HEADER}" >&2
    exit 1
fi

LOCK_TSV="$(
    python3 - "${LOCK_FILE}" <<'PY'
import json
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text())
required = [
    "upstream_url",
    "upstream_tag",
    "upstream_commit",
    "libsvm_version",
]
for key in required:
    if key not in lock:
        raise SystemExit(f"missing key in lock file: {key}")

print(
    lock["upstream_url"],
    lock["upstream_tag"],
    lock.get("upstream_tag_object", ""),
    lock["upstream_commit"],
    lock["libsvm_version"],
    sep="\t",
)
PY
)"

IFS=$'\t' read -r UPSTREAM_URL UPSTREAM_TAG UPSTREAM_TAG_OBJECT UPSTREAM_COMMIT LOCKED_VERSION <<< "${LOCK_TSV}"

VENDOR_VERSION="$(
    grep -E '^#define LIBSVM_VERSION[[:space:]]+[0-9]+' "${VENDOR_HEADER}" \
        | awk '{print $3}'
)"

if [[ -z "${VENDOR_VERSION}" ]]; then
    echo "ERROR: could not read LIBSVM_VERSION from ${VENDOR_HEADER}" >&2
    exit 1
fi

TAG_VERSION="${UPSTREAM_TAG#v}"
if [[ "${TAG_VERSION}" != "${LOCKED_VERSION}" ]]; then
    echo "ERROR: lock mismatch: upstream_tag=${UPSTREAM_TAG} implies ${TAG_VERSION}, but libsvm_version=${LOCKED_VERSION}" >&2
    exit 1
fi

if [[ "${VENDOR_VERSION}" != "${LOCKED_VERSION}" ]]; then
    echo "ERROR: vendor/libsvm/svm.h has LIBSVM_VERSION=${VENDOR_VERSION}, expected ${LOCKED_VERSION}" >&2
    exit 1
fi

RESOLVED_TAG_OBJECT="$(
    git ls-remote --tags --refs "${UPSTREAM_URL}" "refs/tags/${UPSTREAM_TAG}" \
        | awk '{print $1}'
)"

if [[ -z "${RESOLVED_TAG_OBJECT}" ]]; then
    echo "ERROR: could not resolve ${UPSTREAM_TAG} from ${UPSTREAM_URL}" >&2
    exit 1
fi

if [[ -n "${UPSTREAM_TAG_OBJECT}" && "${RESOLVED_TAG_OBJECT}" != "${UPSTREAM_TAG_OBJECT}" ]]; then
    echo "ERROR: upstream tag object mismatch: ${UPSTREAM_TAG} resolved to ${RESOLVED_TAG_OBJECT}, expected ${UPSTREAM_TAG_OBJECT}" >&2
    exit 1
fi

RESOLVED_COMMIT="$(
    git ls-remote --tags "${UPSTREAM_URL}" "refs/tags/${UPSTREAM_TAG}^{}" \
        | awk '{print $1}'
)"
if [[ -z "${RESOLVED_COMMIT}" ]]; then
    echo "ERROR: could not resolve peeled commit for ${UPSTREAM_TAG} from ${UPSTREAM_URL}" >&2
    exit 1
fi

if [[ "${RESOLVED_COMMIT}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "ERROR: upstream tag peeled-commit mismatch: ${UPSTREAM_TAG} resolved to ${RESOLVED_COMMIT}, expected ${UPSTREAM_COMMIT}" >&2
    exit 1
fi

echo "LIBSVM reference lock OK"
echo "  upstream: ${UPSTREAM_URL}"
echo "  tag:      ${UPSTREAM_TAG}"
if [[ -n "${UPSTREAM_TAG_OBJECT}" ]]; then
    echo "  tag obj:  ${UPSTREAM_TAG_OBJECT}"
fi
echo "  commit:   ${UPSTREAM_COMMIT}"
echo "  version:  ${LOCKED_VERSION}"
