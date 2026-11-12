#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Measure the distance from the Rust tool to the C tool.
# DIFF_SCOPE=full selects every case.
# The suite keeps its own numeric tolerances.
# The old suite folder is removed so an old file cannot enter the JSON.
# The suite also rewrites the differential report under reference/.

set -euo pipefail

# macOS realpath has no -m.
# Python realpath resolves the path on Linux and on macOS.
ROOT="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$(dirname "$0")/..")"
# The suite calls cargo in the current folder, and the report records
# its input paths: run from the root so both work and the paths stay relative.
cd "${ROOT}"
SUITE=".tmp/differential_suite"
EXPORT="scripts/parity_export.py"
COMPARE="scripts/parity_compare.py"
CONFIG="parity/tolerances.json"
REPORT="parity/parity_ulp.json"
REF="${SUITE}/parity_c.json"
CAND="${SUITE}/parity_r.json"

rm -rf "${SUITE}"
mkdir -p "${SUITE}"

# env sets the limits on the suite process.
# jobscope runs that process when the command is installed.
if command -v jobscope >/dev/null 2>&1; then
    jobscope env RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 DIFF_SCOPE=full \
        python3 scripts/run_differential_suite.py
else
    env RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 DIFF_SCOPE=full \
        python3 scripts/run_differential_suite.py
fi

python3 "${EXPORT}" "${SUITE}"

set +e
python3 "${COMPARE}" \
    --ref "${REF}" \
    --cand "${CAND}" \
    --config "${CONFIG}" \
    --report "${REPORT}"
status=$?
set -e

if [[ -f "${REPORT}" ]]; then
    python3 - "${REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print("summary " + json.dumps(report["summary"], sort_keys=True))
kind_ulp = {}
kind_rel = {}
for field in report["fields"]:
    name = field["name"]
    ulp = field["max_ulp"]
    rel = field["max_rel_err"]
    print(f"{name} max_ulp {ulp} max_rel_err {rel}")
    kind = name.split(".", 1)[0]
    if isinstance(ulp, int):
        kind_ulp[kind] = max(kind_ulp.get(kind, 0), ulp)
    if isinstance(rel, (int, float)):
        kind_rel[kind] = max(kind_rel.get(kind, 0.0), rel)
print("kind_max")
for kind in kind_ulp:
    print(f"{kind} max_ulp {kind_ulp[kind]} max_rel_err {kind_rel.get(kind, 0.0)}")
PY
fi

exit "${status}"
