# Differential Tolerance Policy

Date: 2026-02-11
Policy ID: `differential-v3`
Applies to: `scripts/run_differential_suite.py`

## Purpose

Define auditable comparison rules for Rust-vs-C LIBSVM differential verification.
Differential baselines are recorded per platform; libc `rand()` replication exists
for macOS and Linux.

## Current Per-Platform Baselines

Full-scope suite (250 configurations), post CV-shuffle alignment to the
process-global libc `rand()` replica (2026-06-10):

| Platform | Pass | Warn | Fail | Skip | Notes |
|---|---:|---:|---:|---:|---|
| macOS (arm64, Apple clang 21) | 237 | 3 | 0 | 10 | committed artifacts (`differential_report.md`, `differential_results.json`) |
| Linux (x86_64, pop-os) | 240 | 0 | 0 | 10 | counts recorded from the same-day full run; artifacts not committed |

Benchmark baselines are also per platform: `benchmark_report.md` /
`benchmark_results.json` are macOS-generated. The Linux (pop-os) full bench
reference point: `train_probability` ratio median 1.074 / worst 2.058
(small cases).

## Hard-Fail Rules

1. Any training/prediction command success mismatch between C and Rust.
2. Any non-probability classification label mismatch, except the one-class near-boundary rule below.
3. Any probability classification predicted-label mismatch (first token per row).
4. Any model-header mismatch beyond configured tolerances, except the rho-only near-equivalence rule below.

## Warning Rules

1. Probability numeric-value drift (with predicted label still matching) above tolerance.
2. Probability metadata drift in model header above probability-header tolerance.
3. Cases where both C and Rust probability training fail for the same combo.
4. One-class near-boundary label drift when all of the following are true:
   - label mismatch count is <= `5`
   - `rho` relative drift is <= `1e-6`
   - support-vector rows are identical and max `sv_coef` absolute drift is <= `1e-8`
5. Rho-only header drift for classification models when all of the following are true:
   - the header mismatch is only `rho`
   - `rho` relative drift is <= `5e-2`
   - support-vector rows are identical and max `sv_coef` absolute drift is <= `1e-8`
6. Targeted epsilon-SVR near-parity drift for `housing_scale_s3_t2_tuned` when all of the following are true:
   - non-probability max relative drift is <= `6e-5`
   - non-probability max absolute drift is <= `6e-4`
   - `rho` relative drift is <= `1e-5`
   - support-vector rows are identical and max `sv_coef` absolute drift is <= `4e-3`
   - cross-predict parity holds:
     - Rust predictor on C model matches C predictor on C model (strict scalar tolerance)
     - C predictor on Rust model matches Rust predictor on Rust model (strict scalar tolerance)
   - rule is enabled (`DIFF_ENABLE_TARGETED_SVR_WARN=1`, default)

## Numeric Tolerances

### Non-probability scalar outputs

- Relative tolerance: `1.5e-5`
- Absolute tolerance: `1e-8`

Used for regression outputs and scalar probability-path outputs.
The relative tolerance can be overridden for sensitivity studies with `DIFF_NONPROB_REL_TOL=<float>`.
Targeted SVR warning can be disabled for strict-only runs with `DIFF_ENABLE_TARGETED_SVR_WARN=0`.

### Probability classification value columns

- Relative tolerance: `2.5e-1`
- Absolute tolerance: `3e-2`

Applies only to probability components (not predicted label column).

### Model header floating values (standard)

- Relative tolerance: `1e-2`
- Absolute tolerance: `5e-4`

Keys: `gamma`, `coef0`, `rho`.

### Model header floating values (probability metadata)

- Relative tolerance: `6e-2`
- Absolute tolerance: `5e-3`

Keys: `probA`, `probB`, `prob_density_marks`.

## Exact-Match Rules

1. Header structure/keys in model files.
2. Integer model fields (`degree`, `nr_class`, `total_sv`, `label`, `nr_sv`).
3. String model fields (`svm_type`, `kernel_type`).
4. Classification predicted labels (non-probability and probability modes), except one-class near-boundary warning rule above.

## Governance

1. Any tolerance change must update this file and be called out in commit message.
2. Differential report consumers must record policy ID used for the run.
