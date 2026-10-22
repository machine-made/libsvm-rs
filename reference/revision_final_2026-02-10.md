# Full Revision Final Report

Date: 2026-02-10

## Scope Completed

This report closes the full-revision request across parity, compatibility, coverage, and benchmarks.

## 1) Function-by-Function Parity

Source matrix: `reference/function_parity_matrix.md`

Final status:
- `Equivalent`: `80`
- `Equivalent (API shape)`: `21`
- `Intentional divergence`: `10`
- `Missing`: `0`

Notable closure items:
- Added direct library helpers matching C API names:
  - `svm_get_svm_type`
  - `svm_get_nr_class`
  - `svm_get_labels`
  - `svm_get_sv_indices`
  - `svm_get_nr_sv`
  - `svm_get_svr_probability`
  - `svm_check_probability_model`
- Added model accessor methods to `SvmModel` and test coverage for helper behavior.

## 2) Compatibility Matrix (Rust vs C LIBSVM)

Command:
- `bash scripts/compare_references.sh`

Artifacts:
- `reference/diff_report.txt`
- `reference/compare_summary.json`

Latest result:
- Passed: `65`
- Failed: `0`
- Warnings: `20`
- Skipped: `5`

Coverage notes:
- Includes kernels `-t 0..4` (precomputed included).
- Includes datasets:
  - `heart_scale`
  - `iris.scale`
  - `housing_scale`
  - `*.precomputed` variants
- Machine-readable summary now emitted to `reference/compare_summary.json`.

Warning profile:
- Warning-only differences are probability calibration / probability-model-header deltas for selected cases.
- No hard-fail prediction or model-header mismatches in required non-probability paths.

## 3) Coverage Metrics and Gates

Command:
- `bash scripts/check_coverage_thresholds.sh`

Artifact:
- `reference/coverage_report.md`

Thresholds enforced:
- `libsvm-rs` line coverage `>= 92%`
- `libsvm-rs` function coverage `>= 90%`
- Workspace line coverage `>= 85%`

Latest measured values:
- `libsvm-rs` line coverage: `93.19%`
- `libsvm-rs` function coverage: `92.86%`
- Workspace line coverage: `88.77%`

CI integration:
- Coverage job now runs `scripts/check_coverage_thresholds.sh` and fails if thresholds regress.

## 4) Benchmarks (Rust vs C)

Command:
- `python3 scripts/benchmark_compare.py`

Artifacts:
- `reference/benchmark_results.json` (raw)
- `reference/benchmark_report.md` (summary)

Benchmark scope:
- All 5 SVM types (`-s 0..4`)
- All 5 kernels (`-t 0..4`)
- Classification and regression datasets, including precomputed datasets
- Train + predict
- Probability train + probability predict where applicable (`-s 0,1,3,4`)

Aggregate summary (latest run):
- `predict`: Rust/C median ratio `1.002`
- `predict_probability`: Rust/C median ratio `0.972`
- `train`: Rust/C median ratio `1.039`
- `train_probability`: Rust/C median ratio `1.096`

CI integration:
- Benchmark job now runs `python3 scripts/benchmark_compare.py` and uploads benchmark artifacts.

## 5) Acceptance Check

Against plan criteria in `plan_2026-02-10.md`:
- Full function matrix completed: `PASS`
- Full compatibility harness pass on valid cases: `PASS` (0 failures)
- Coverage thresholds met and enforced in CI: `PASS`
- Benchmark report generated with raw data and Rust-vs-C comparison: `PASS`

## 6) Remaining Intentional Divergences

These are documented and retained by design:
- Rust ownership/RAII replaces C explicit free/destroy APIs.
- `set_quiet` is supported; arbitrary print callback injection (`svm_set_print_string_function`) is not exposed.
