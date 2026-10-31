#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="${PROJ_ROOT}/reference/libsvm_upstream_lock.json"

# Resolve a path (symlinks followed, missing parts allowed) and require it to be
# inside the repository. Print the resolved path. Exit with an error otherwise.
# Python's realpath works the same on Linux and macOS; BSD realpath has no -m.
resolve_path() {
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}
PROJ_ROOT_REAL="$(resolve_path "${PROJ_ROOT}")"
safe_output() {
    local resolved
    resolved="$(resolve_path "$1")"
    case "${resolved}" in
        "${PROJ_ROOT_REAL}"/*) printf '%s\n' "${resolved}" ;;
        *)
            echo "ERROR: $1 resolves to ${resolved}, outside the repository (${PROJ_ROOT_REAL})." >&2
            echo "Remove symlinks that point outside the repository, then run the script again." >&2
            exit 1
            ;;
    esac
}

# Check every output path before the first write. Keep the resolved paths.
WORK_ROOT="$(safe_output "${PROJ_ROOT}/.tmp/reference_upstream")"
SRC_DIR="$(safe_output "${WORK_ROOT}/libsvm")"
PROVENANCE_FILE="$(safe_output "${PROJ_ROOT}/reference/reference_provenance.json")"
REPORT_FILE="$(safe_output "${PROJ_ROOT}/reference/reference_build_report.md")"
BUILD_LOG="$(safe_output "${PROJ_ROOT}/reference/reference_build.log")"

mkdir -p "${WORK_ROOT}"

if [[ ! -f "${LOCK_FILE}" ]]; then
    echo "ERROR: lock file not found: ${LOCK_FILE}" >&2
    exit 1
fi

LOCK_TSV="$(
    python3 - "${LOCK_FILE}" <<'PY'
import json
import sys
from pathlib import Path

lock = json.loads(Path(sys.argv[1]).read_text())
print(
    lock["upstream_url"],
    lock["upstream_tag"],
    lock.get("upstream_tag_object", ""),
    lock["upstream_commit"],
    lock["libsvm_version"],
    lock.get("release_date", ""),
    sep="\t",
)
PY
)"

IFS=$'\t' read -r UPSTREAM_URL UPSTREAM_TAG UPSTREAM_TAG_OBJECT UPSTREAM_COMMIT LOCKED_VERSION RELEASE_DATE <<< "${LOCK_TSV}"

if [[ -d "${SRC_DIR}/.git" ]]; then
    git -C "${SRC_DIR}" fetch --tags --force origin
else
    git clone "${UPSTREAM_URL}" "${SRC_DIR}"
fi

git -C "${SRC_DIR}" fetch --tags --force origin
git -C "${SRC_DIR}" checkout --detach "${UPSTREAM_COMMIT}"

HEAD_COMMIT="$(git -C "${SRC_DIR}" rev-parse HEAD)"
if [[ "${HEAD_COMMIT}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "ERROR: checked-out commit ${HEAD_COMMIT} does not match locked ${UPSTREAM_COMMIT}" >&2
    exit 1
fi

TAG_COMMIT="$(git -C "${SRC_DIR}" rev-list -n 1 "${UPSTREAM_TAG}")"
if [[ "${TAG_COMMIT}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "ERROR: local tag ${UPSTREAM_TAG} points to ${TAG_COMMIT}, expected ${UPSTREAM_COMMIT}" >&2
    exit 1
fi

if [[ -n "${UPSTREAM_TAG_OBJECT}" ]]; then
    TAG_OBJECT="$(git -C "${SRC_DIR}" rev-parse "${UPSTREAM_TAG}")"
    if [[ "${TAG_OBJECT}" != "${UPSTREAM_TAG_OBJECT}" ]]; then
        echo "ERROR: local tag object ${UPSTREAM_TAG} is ${TAG_OBJECT}, expected ${UPSTREAM_TAG_OBJECT}" >&2
        exit 1
    fi
fi

VENDOR_VERSION="$(
    grep -E '^#define LIBSVM_VERSION[[:space:]]+[0-9]+' "${SRC_DIR}/svm.h" \
        | awk '{print $3}'
)"
if [[ "${VENDOR_VERSION}" != "${LOCKED_VERSION}" ]]; then
    echo "ERROR: upstream source LIBSVM_VERSION=${VENDOR_VERSION}, expected ${LOCKED_VERSION}" >&2
    exit 1
fi

if ! make -C "${SRC_DIR}" clean >"${BUILD_LOG}" 2>&1; then
    echo "ERROR: failed to clean upstream LIBSVM build. See ${BUILD_LOG}" >&2
    tail -n 40 "${BUILD_LOG}" >&2 || true
    exit 1
fi
if ! make -C "${SRC_DIR}" >>"${BUILD_LOG}" 2>&1; then
    echo "ERROR: failed to build upstream LIBSVM. See ${BUILD_LOG}" >&2
    tail -n 80 "${BUILD_LOG}" >&2 || true
    exit 1
fi

CC_VERSION="$(cc --version | head -n 1 || true)"

python3 - \
    "${PROVENANCE_FILE}" \
    "${UPSTREAM_URL}" \
    "${UPSTREAM_TAG}" \
    "${UPSTREAM_TAG_OBJECT}" \
    "${UPSTREAM_COMMIT}" \
    "${LOCKED_VERSION}" \
    "${RELEASE_DATE}" \
    "${CC_VERSION}" \
    "${SRC_DIR}/svm-train" \
    "${SRC_DIR}/svm-predict" \
    "${PROJ_ROOT}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
upstream_url = sys.argv[2]
tag = sys.argv[3]
tag_object = sys.argv[4]
commit = sys.argv[5]
version = int(sys.argv[6])
release_date = sys.argv[7]
cc_version = sys.argv[8]
train_bin = Path(sys.argv[9])
predict_bin = Path(sys.argv[10])
proj_root = Path(sys.argv[11])


def repo_relative(path: Path) -> str:
    # Keep committed provenance free of local paths.
    try:
        return path.resolve().relative_to(proj_root.resolve()).as_posix()
    except ValueError:
        sys.exit(f"error: {path} is outside the repository ({proj_root}); keep the work directory inside it.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "upstream": {
        "url": upstream_url,
        "tag": tag,
        "tag_object": tag_object,
        "commit": commit,
        "libsvm_version": version,
        "release_date": release_date,
    },
    "build": {
        "cc_version": cc_version,
        "svm_train_sha256": sha256(train_bin),
        "svm_predict_sha256": sha256(predict_bin),
        "svm_train_path": repo_relative(train_bin),
        "svm_predict_path": repo_relative(predict_bin),
    },
}

out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

python3 - "${PROVENANCE_FILE}" "${REPORT_FILE}" "${PROJ_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

provenance_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
proj_root = Path(sys.argv[3])
data = json.loads(provenance_path.read_text())

up = data["upstream"]
build = data["build"]

lines = [
    "# Reference Build Report",
    "",
    f"Generated: {data['generated_utc']}",
    "",
    "## Upstream Lock",
    "",
    f"- URL: `{up['url']}`",
    f"- Tag: `{up['tag']}`",
    f"- Tag object: `{up.get('tag_object', '')}`",
    f"- Commit: `{up['commit']}`",
    f"- LIBSVM_VERSION: `{up['libsvm_version']}`",
    f"- Release date: `{up.get('release_date', '')}`",
    "",
    "## Build Environment",
    "",
    f"- Compiler: `{build['cc_version']}`",
    f"- `svm-train` SHA256: `{build['svm_train_sha256']}`",
    f"- `svm-predict` SHA256: `{build['svm_predict_sha256']}`",
    f"- `svm-train` path: `{build['svm_train_path']}`",
    f"- `svm-predict` path: `{build['svm_predict_path']}`",
    "",
    "## Artifacts",
    "",
    f"- Provenance JSON: `{provenance_path.resolve().relative_to(proj_root.resolve()).as_posix()}`",
]

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Reference LIBSVM setup complete"
echo "  source: ${SRC_DIR}"
echo "  svm-train: ${SRC_DIR}/svm-train"
echo "  svm-predict: ${SRC_DIR}/svm-predict"
echo "  build log: ${BUILD_LOG}"
echo "  provenance: ${PROVENANCE_FILE}"
echo "  report: ${REPORT_FILE}"
