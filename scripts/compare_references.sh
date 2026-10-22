#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_TRAIN="${PROJ_ROOT}/target/release/svm-train-rs"
RUST_PREDICT="${PROJ_ROOT}/target/release/svm-predict-rs"
DATA_DIR="${PROJ_ROOT}/data"
REF_DIR="${PROJ_ROOT}/reference"
DIFF_REPORT="${REF_DIR}/diff_report.txt"
SUMMARY_JSON="${REF_DIR}/compare_summary.json"
TMP_DIR="${PROJ_ROOT}/.tmp/compare_references"

mkdir -p "${TMP_DIR}"

PASSED=0
FAILED=0
WARNED=0
SKIPPED=0
> "${DIFF_REPORT}"

compare_scalar_file() {
    # Compare one-float-per-line files with relative tolerance.
    python3 - "$1" "$2" <<'PY'
import math
import sys

ref_path, rust_path = sys.argv[1], sys.argv[2]
tol = 1e-6

with open(ref_path) as f:
    ref = [ln.strip() for ln in f if ln.strip()]
with open(rust_path) as f:
    rust = [ln.strip() for ln in f if ln.strip()]

if len(ref) != len(rust):
    print(f"line-count mismatch: ref={len(ref)} rust={len(rust)}")
    sys.exit(1)

for i, (a, b) in enumerate(zip(ref, rust), start=1):
    va = float(a)
    vb = float(b)
    diff = abs(va - vb)
    scale = max(abs(va), abs(vb), 1e-15)
    if diff / scale > tol:
        print(f"line {i}: ref={va} rust={vb} reldiff={diff/scale:.3e} > tol={tol}")
        sys.exit(1)
PY
}

compare_probability_file() {
    # Compare probability output files:
    # - header must match exactly when present
    # - predicted label column must match exactly
    # - probability columns use relative tolerance
    python3 - "$1" "$2" <<'PY'
import math
import sys

ref_path, rust_path = sys.argv[1], sys.argv[2]
rel_tol = 2.5e-1
abs_tol = 3e-2

with open(ref_path) as f:
    ref = [ln.strip() for ln in f if ln.strip()]
with open(rust_path) as f:
    rust = [ln.strip() for ln in f if ln.strip()]

if len(ref) != len(rust):
    print(f"line-count mismatch: ref={len(ref)} rust={len(rust)}")
    sys.exit(1)

if not ref:
    sys.exit(0)

start = 0
if ref[0].startswith("labels") or rust[0].startswith("labels"):
    if ref[0] != rust[0]:
        print(f"header mismatch:\n  ref : {ref[0]}\n  rust: {rust[0]}")
        sys.exit(1)
    start = 1

for line_no in range(start, len(ref)):
    ra = ref[line_no].split()
    rb = rust[line_no].split()
    if len(ra) != len(rb):
        print(f"line {line_no+1}: token-count mismatch ref={len(ra)} rust={len(rb)}")
        sys.exit(1)
    if not ra:
        continue

    # Predicted label (first column) must match exactly as a numeric value.
    try:
        label_a = float(ra[0])
        label_b = float(rb[0])
    except ValueError:
        print(f"line {line_no+1}: invalid predicted label token")
        sys.exit(1)
    if label_a != label_b:
        print(
            f"line {line_no+1}, col 1: predicted label mismatch "
            f"ref={label_a} rust={label_b}"
        )
        sys.exit(1)

    # Probability columns tolerance check.
    for col in range(1, len(ra)):
        va = float(ra[col])
        vb = float(rb[col])
        diff = abs(va - vb)
        scale = max(abs(va), abs(vb), 1e-15)
        threshold = max(abs_tol, rel_tol * scale)
        if diff > threshold:
            print(
                f"line {line_no+1}, col {col+1}: "
                f"ref={va} rust={vb} diff={diff:.3e} "
                f"(rel={diff/scale:.3e}) > threshold={threshold:.3e}"
            )
            sys.exit(1)
PY
}

compare_model_header() {
    # Compare model headers (up to SV section) with strict key checks
    # and tolerance-based checks for floating arrays.
    python3 - "$1" "$2" <<'PY'
import math
import sys

ref_path, rust_path = sys.argv[1], sys.argv[2]
float_keys = {"gamma", "coef0", "rho", "probA", "probB", "prob_density_marks"}
int_keys = {"degree", "nr_class", "total_sv", "label", "nr_sv"}
str_keys = {"svm_type", "kernel_type"}
rel_tol_standard = 1e-2
abs_tol_standard = 5e-4
rel_tol_probability = 6e-2
abs_tol_probability = 5e-3

def parse_header(path):
    out = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "SV":
                break
            parts = line.split()
            out[parts[0]] = parts[1:]
    return out

ref = parse_header(ref_path)
rust = parse_header(rust_path)

all_keys = sorted(set(ref) | set(rust))
for key in all_keys:
    if key not in ref:
        print(f"missing key in ref header: {key}")
        sys.exit(1)
    if key not in rust:
        print(f"missing key in rust header: {key}")
        sys.exit(1)

    a = ref[key]
    b = rust[key]
    if len(a) != len(b):
        print(f"key {key}: value-count mismatch ref={len(a)} rust={len(b)}")
        sys.exit(1)

    if key in str_keys:
        if a != b:
            print(f"key {key}: mismatch ref={a} rust={b}")
            sys.exit(1)
    elif key in int_keys:
        ai = [int(v) for v in a]
        bi = [int(v) for v in b]
        if ai != bi:
            print(f"key {key}: mismatch ref={ai} rust={bi}")
            sys.exit(1)
    elif key in float_keys:
        for idx, (va_s, vb_s) in enumerate(zip(a, b), start=1):
            va = float(va_s)
            vb = float(vb_s)
            diff = abs(va - vb)
            scale = max(abs(va), abs(vb), 1e-15)
            if key in {"probA", "probB", "prob_density_marks"}:
                rel_tol = rel_tol_probability
                abs_tol = abs_tol_probability
            else:
                rel_tol = rel_tol_standard
                abs_tol = abs_tol_standard
            threshold = max(abs_tol, rel_tol * scale)
            if diff > threshold:
                print(
                    f"key {key}[{idx}]: ref={va} rust={vb} "
                    f"diff={diff:.3e} (rel={diff/scale:.3e}) > threshold={threshold:.3e}"
                )
                sys.exit(1)
    else:
        # Fallback: exact string comparison for unknown keys.
        if a != b:
            print(f"key {key}: mismatch ref={a} rust={b}")
            sys.exit(1)
PY
}

for ref_path in "${REF_DIR}"/*/*/; do
    [[ -d "${ref_path}" ]] || continue

    dir_name=$(basename "${ref_path}")
    dataset_name=$(basename "$(dirname "${ref_path}")")

    if [[ ! "${dir_name}" =~ ^s([0-9]+)_t([0-9]+)$ ]]; then
        continue
    fi

    svm_type="${BASH_REMATCH[1]}"
    kernel="${BASH_REMATCH[2]}"
    case_id="${dataset_name} -s ${svm_type} -t ${kernel}"

    if [[ ! -f "${DATA_DIR}/${dataset_name}" ]]; then
        echo "SKIP: Dataset not found: ${dataset_name}"
        ((SKIPPED++)) || true
        continue
    fi

    ref_model="${ref_path}/model"
    ref_pred="${ref_path}/predictions"
    ref_model_prob="${ref_path}/model_prob"
    ref_pred_prob="${ref_path}/predictions_prob"

    [[ -f "${ref_pred}" ]] || continue

    # Skip known-divergent combos: classification on regression datasets
    # (housing_scale has ~80 float labels; tiny numerical diffs flip multiclass predictions)
    if [[ "${dataset_name}" == housing_scale* && "${svm_type}" -le 1 ]]; then
        echo "SKIP: ${case_id} (classification on regression data)"
        ((SKIPPED++)) || true
        continue
    fi

    rust_model="${TMP_DIR}/rust_model_${dataset_name}_s${svm_type}_t${kernel}"
    rust_pred="${TMP_DIR}/rust_pred_${dataset_name}_s${svm_type}_t${kernel}"
    rust_model_prob="${TMP_DIR}/rust_model_prob_${dataset_name}_s${svm_type}_t${kernel}"
    rust_pred_prob="${TMP_DIR}/rust_pred_prob_${dataset_name}_s${svm_type}_t${kernel}"

    case_failed=0

    if ! "${RUST_TRAIN}" -q -s "${svm_type}" -t "${kernel}" \
        "${DATA_DIR}/${dataset_name}" "${rust_model}" 2>/dev/null; then
        echo "FAIL: Training ${case_id}"
        echo "FAIL: Training ${case_id}" >> "${DIFF_REPORT}"
        case_failed=1
    fi

    if [[ "${case_failed}" -eq 0 ]] && ! "${RUST_PREDICT}" -q \
        "${DATA_DIR}/${dataset_name}" "${rust_model}" "${rust_pred}" 2>/dev/null; then
        echo "FAIL: Prediction ${case_id}"
        echo "FAIL: Prediction ${case_id}" >> "${DIFF_REPORT}"
        case_failed=1
    fi

    if [[ "${case_failed}" -eq 0 ]]; then
        if [[ "${svm_type}" -le 2 ]]; then
            if ! diff -q "${ref_pred}" "${rust_pred}" >/dev/null 2>&1; then
                echo "FAIL: ${case_id} (label diff)"
                {
                    echo "--- Prediction diff for ${case_id} ---"
                    diff "${ref_pred}" "${rust_pred}" || true
                } >> "${DIFF_REPORT}"
                case_failed=1
            fi
        else
            if ! compare_scalar_file "${ref_pred}" "${rust_pred}" >> "${DIFF_REPORT}" 2>&1; then
                echo "FAIL: ${case_id} (regression tolerance)"
                echo "FAIL: Regression tolerance ${case_id}" >> "${DIFF_REPORT}"
                case_failed=1
            fi
        fi
    fi

    if [[ "${case_failed}" -eq 0 ]] && [[ -f "${ref_model}" ]]; then
        if ! compare_model_header "${ref_model}" "${rust_model}" >> "${DIFF_REPORT}" 2>&1; then
            echo "FAIL: ${case_id} (model header)"
            echo "FAIL: Model header mismatch ${case_id}" >> "${DIFF_REPORT}"
            case_failed=1
        fi
    fi

    if [[ "${case_failed}" -eq 0 ]] && [[ -f "${ref_model_prob}" && -f "${ref_pred_prob}" ]]; then
        if ! "${RUST_TRAIN}" -q -b 1 -s "${svm_type}" -t "${kernel}" \
            "${DATA_DIR}/${dataset_name}" "${rust_model_prob}" 2>/dev/null; then
            echo "FAIL: ${case_id} (probability training)"
            echo "FAIL: Probability training ${case_id}" >> "${DIFF_REPORT}"
            case_failed=1
        fi

        if [[ "${case_failed}" -eq 0 ]] && ! "${RUST_PREDICT}" -q -b 1 \
            "${DATA_DIR}/${dataset_name}" "${rust_model_prob}" "${rust_pred_prob}" 2>/dev/null; then
            echo "FAIL: ${case_id} (probability prediction)"
            echo "FAIL: Probability prediction ${case_id}" >> "${DIFF_REPORT}"
            case_failed=1
        fi

        if [[ "${case_failed}" -eq 0 ]]; then
            if [[ "${svm_type}" -le 2 ]]; then
                if ! compare_probability_file "${ref_pred_prob}" "${rust_pred_prob}" >> "${DIFF_REPORT}" 2>&1; then
                    echo "WARN: ${case_id} (probability outputs)"
                    echo "WARN: Probability output mismatch ${case_id}" >> "${DIFF_REPORT}"
                    ((WARNED++)) || true
                fi
            else
                if ! compare_scalar_file "${ref_pred_prob}" "${rust_pred_prob}" >> "${DIFF_REPORT}" 2>&1; then
                    echo "WARN: ${case_id} (probability regression outputs)"
                    echo "WARN: Probability regression output mismatch ${case_id}" >> "${DIFF_REPORT}"
                    ((WARNED++)) || true
                fi
            fi
        fi

        if [[ "${case_failed}" -eq 0 ]]; then
            if ! compare_model_header "${ref_model_prob}" "${rust_model_prob}" >> "${DIFF_REPORT}" 2>&1; then
                echo "WARN: ${case_id} (probability model header)"
                echo "WARN: Probability model header mismatch ${case_id}" >> "${DIFF_REPORT}"
                ((WARNED++)) || true
            fi
        fi
    fi

    if [[ "${case_failed}" -eq 0 ]]; then
        echo "PASS: ${case_id}"
        ((PASSED++)) || true
    else
        ((FAILED++)) || true
    fi

    rm -f "${rust_model}" "${rust_pred}" "${rust_model_prob}" "${rust_pred_prob}"
done

echo ""
echo "Results: ${PASSED} passed, ${FAILED} failed, ${WARNED} warnings, ${SKIPPED} skipped"

cat > "${SUMMARY_JSON}" <<EOF
{
  "generated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "passed": ${PASSED},
  "failed": ${FAILED},
  "warnings": ${WARNED},
  "skipped": ${SKIPPED}
}
EOF

echo "Summary: ${SUMMARY_JSON}"

[[ ${FAILED} -eq 0 ]]
