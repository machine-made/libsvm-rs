# Revision Baseline Report

Date: 2026-02-10

This report captures the baseline metrics used for the full revision effort.

## Commands Run

```bash
# crate-only coverage
PATH="$HOME/.cargo/bin:$PATH" cargo llvm-cov -p libsvm-rs --all-features --summary-only

# workspace coverage (includes binaries)
PATH="$HOME/.cargo/bin:$PATH" cargo llvm-cov --workspace --all-features --summary-only

# benchmark sample run
cargo bench -p libsvm-rs --bench svm_bench -- --warm-up-time 0.5 --measurement-time 0.5 --sample-size 10

# C-reference parity comparison
./scripts/compare_references.sh
```

## Coverage (`libsvm-rs` crate)

Totals:
- Regions: `88.40%`
- Functions: `85.97%`
- Lines: `88.79%`

Lowest-covered modules:
- `crates/libsvm/src/probability.rs`: `75.73%` regions, `73.03%` lines
- `crates/libsvm/src/io.rs`: `76.83%` regions, `76.83%` lines
- `crates/libsvm/src/solver.rs`: `84.60%` regions, `85.34%` lines

## Coverage (workspace)

Totals:
- Regions: `75.11%`
- Functions: `73.64%`
- Lines: `74.65%`

Primary reason for lower workspace totals:
- CLI binaries currently have `0` direct unit/integration tests:
  - `bins/svm-train-rs/src/main.rs`
  - `bins/svm-predict-rs/src/main.rs`
  - `bins/svm-scale-rs/src/main.rs`

## Benchmarks (criterion quick baseline)

Measured intervals from sample run:
- `train_rbf`: `[1.0490 ms, 1.3150 ms, 1.5363 ms]`
- `train_linear`: `[1.6807 ms, 1.8769 ms, 2.3256 ms]`
- `predict_all`: `[656.43 us, 660.83 us, 666.69 us]`
- `train_with_probability`: `[4.8233 ms, 5.0087 ms, 5.4353 ms]`

Notes:
- Criterion reported relative change vs prior local baseline; this reflects local machine/run variance unless pinned against stable artifacts.

## C-Reference Compatibility

Result from `scripts/compare_references.sh`:
- Passed: `52`
- Failed: `0`
- Warnings: `16` (probability calibration/model-header deltas beyond warning thresholds)
- Skipped: `4` (documented classification-on-regression skip cases for `housing_scale` with `-s 0/1`)

Datasets covered by current script:
- `heart_scale`
- `iris.scale`
- `housing_scale`

Kernels covered by current script:
- `-t 0..3` (precomputed `-t 4` not yet included in script matrix)

Additional checks now performed by the script:
- Base predictions (hard-fail): exact labels for classification/one-class, tolerance for regression.
- Model header parity (hard-fail for non-probability models).
- Probability outputs and probability-model header parity (warning-only; reported in `reference/diff_report.txt`).

## Related Artifacts

- Function-level parity matrix: `reference/function_parity_matrix.md`
- Existing compare script: `scripts/compare_references.sh`
- Existing reference generation script: `scripts/generate_references.sh`
