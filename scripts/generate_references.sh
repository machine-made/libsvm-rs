#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBSVM_TRAIN="${PROJ_ROOT}/vendor/libsvm/svm-train"
LIBSVM_PREDICT="${PROJ_ROOT}/vendor/libsvm/svm-predict"
DATA_DIR="${PROJ_ROOT}/data"
REF_DIR="${PROJ_ROOT}/reference"

mkdir -p "${REF_DIR}"

python3 "${PROJ_ROOT}/scripts/generate_precomputed_datasets.py"

DATASETS=("heart_scale" "iris.scale" "housing_scale")
PRECOMPUTED_DATASETS=("heart_scale.precomputed" "iris.scale.precomputed" "housing_scale.precomputed")
SVM_TYPES=(0 1 2 3 4)
KERNELS=(0 1 2 3)

for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "${DATA_DIR}/${dataset}" ]]; then
        echo "Dataset not found: ${DATA_DIR}/${dataset}"
        continue
    fi

    for svm_type in "${SVM_TYPES[@]}"; do
        for kernel in "${KERNELS[@]}"; do
            echo "Testing ${dataset} -s ${svm_type} -t ${kernel}..."

            ref_subdir="${REF_DIR}/${dataset}/s${svm_type}_t${kernel}"
            mkdir -p "${ref_subdir}"

            "${LIBSVM_TRAIN}" -s "${svm_type}" -t "${kernel}" \
                "${DATA_DIR}/${dataset}" "${ref_subdir}/model" >/dev/null 2>&1 || continue

            "${LIBSVM_PREDICT}" "${DATA_DIR}/${dataset}" "${ref_subdir}/model" \
                "${ref_subdir}/predictions" >/dev/null 2>&1 || continue

            # Probability estimates (not applicable to one-class, type 2)
            if [[ "${svm_type}" != "2" ]]; then
                "${LIBSVM_TRAIN}" -b 1 -s "${svm_type}" -t "${kernel}" \
                    "${DATA_DIR}/${dataset}" "${ref_subdir}/model_prob" >/dev/null 2>&1 || true

                if [[ -f "${ref_subdir}/model_prob" ]]; then
                    "${LIBSVM_PREDICT}" -b 1 "${DATA_DIR}/${dataset}" "${ref_subdir}/model_prob" \
                        "${ref_subdir}/predictions_prob" >/dev/null 2>&1 || true
                fi
            fi
        done
    done
done

for dataset in "${PRECOMPUTED_DATASETS[@]}"; do
    if [[ ! -f "${DATA_DIR}/${dataset}" ]]; then
        echo "Dataset not found: ${DATA_DIR}/${dataset}"
        continue
    fi

    for svm_type in "${SVM_TYPES[@]}"; do
        kernel=4
        echo "Testing ${dataset} -s ${svm_type} -t ${kernel}..."

        ref_subdir="${REF_DIR}/${dataset}/s${svm_type}_t${kernel}"
        mkdir -p "${ref_subdir}"

        "${LIBSVM_TRAIN}" -s "${svm_type}" -t "${kernel}" \
            "${DATA_DIR}/${dataset}" "${ref_subdir}/model" >/dev/null 2>&1 || continue

        "${LIBSVM_PREDICT}" "${DATA_DIR}/${dataset}" "${ref_subdir}/model" \
            "${ref_subdir}/predictions" >/dev/null 2>&1 || continue

        # Probability estimates (not applicable to one-class, type 2)
        if [[ "${svm_type}" != "2" ]]; then
            "${LIBSVM_TRAIN}" -b 1 -s "${svm_type}" -t "${kernel}" \
                "${DATA_DIR}/${dataset}" "${ref_subdir}/model_prob" >/dev/null 2>&1 || true

            if [[ -f "${ref_subdir}/model_prob" ]]; then
                "${LIBSVM_PREDICT}" -b 1 "${DATA_DIR}/${dataset}" "${ref_subdir}/model_prob" \
                    "${ref_subdir}/predictions_prob" >/dev/null 2>&1 || true
            fi
        fi
    done
done

echo "Reference generation complete."
