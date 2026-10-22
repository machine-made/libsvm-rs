# Benchmark Report

Date: 2026-10-08 22:37:10Z

This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).

## Method

- Warmup runs per command: `3`
- Measured runs per command: `30`
- Timing metric: wall clock (`perf_counter_ns`) per command invocation
- Summary metric: per-case median and p95 from repeated runs

## Aggregate Results

| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |
|---|---:|---:|---:|---:|---:|---:|
| `predict` | 40 | 2.081 | 2.437 | 0.828 | 0.918 | 0.996 |
| `predict_probability` | 30 | 2.314 | 2.635 | 0.838 | 0.921 | 1.005 |
| `train` | 40 | 2.688 | 2.900 | 0.907 | 1.007 | 1.022 |
| `train_probability` | 30 | 6.185 | 6.442 | 0.979 | 1.081 | 1.112 |

## Highest Rust/C Ratios

| Case | Operation | Rust/C median ratio |
|---|---|---:|
| `s3_t2_housing_scale` | `train_probability` | 1.112 |
| `s3_t0_housing_scale` | `train_probability` | 1.083 |
| `s3_t4_housing_scale_precomputed` | `train_probability` | 1.078 |
| `s3_t3_housing_scale` | `train_probability` | 1.068 |
| `s3_t1_housing_scale` | `train_probability` | 1.061 |
| `s4_t2_housing_scale` | `train_probability` | 1.056 |
| `s0_t0_heart_scale` | `train_probability` | 1.049 |
| `s4_t3_housing_scale` | `train_probability` | 1.041 |
| `s3_t3_housing_scale` | `train` | 1.022 |
| `s0_t2_heart_scale` | `train_probability` | 1.021 |

Raw data: `reference/benchmark_results.json`

