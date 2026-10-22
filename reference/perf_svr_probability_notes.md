# SVR probability training performance notes (libsvm-rs-jeh)

## Profiling attempt before code changes

Command built with release optimizations and debug symbols:

```sh
RUSTFLAGS='-C debuginfo=2' cargo build --locked --release -p svm-train-rs
```

`perf record` is unavailable for this unprivileged user on this host:

```text
Error:
Failure to open event 'cpu/cycles/Pu' on PMU 'cpu' which will be removed.
Access to performance monitoring and observability operations is limited.
perf_event_paranoid setting is 4
Error:
Failure to open any events for recording.
```

Fallback profiler used before any code changes:

```sh
valgrind --tool=callgrind --callgrind-out-file=.sc/jeh-callgrind.before -- \
  target/release/svm-train-rs -s 3 -t 2 -b 1 data/housing_scale
```

Callgrind summary (`callgrind_annotate --threshold=0.1 .sc/jeh-callgrind.before`):

```text
708,580,123 (100.0%)  PROGRAM TOTALS
301,865,020 (42.60%)  ???:libsvm_rs::kernel::Kernel::evaluate
```

Observation: the only symbol above the annotation threshold is kernel evaluation, which is outside this bead's always-allowed scope and is explicitly ask-first (`kernel.rs`). I did not make code changes.

## Benchmark ratios

Not run: the differential verification gate failed before benchmarking, so the stop condition applies.

## Verification status

Differential suite command:

```sh
python3 scripts/run_differential_suite.py 2>&1 | tee .sc/jeh-differential.log | tail -5
```

Observed output:

```text
[044/45] housing_scale_precomputed_s3_t4_default
[045/45] housing_scale_precomputed_s4_t4_default
Differential suite complete: 25 pass, 14 warn, 6 fail, 0 skip
Wrote /home/rfrantz/Projects/libsvm-rs/reference/differential_results.json
Wrote /home/rfrantz/Projects/libsvm-rs/reference/differential_report.md
```

Expected by goal: `236 pass / 4 warn / 0 fail / 10 skip`. Because the differential counts include failures, the goal's stop condition applies. Generated tracked reference/data artifacts from the failed run were reverted; `.sc/jeh-differential.log` remains untracked as requested.

## Round 2 bitwise guard and benchmark

Baseline release binary/model capture before edits:

```sh
cargo build --locked --release -p svm-train-rs
mkdir -p .sc/jeh-models-before
# saved housing_scale (-s 3 -t 2 -b 1), heart_scale (-s 0 -t 2 -b 1), iris.scale (-s 0 -t 2 -b 1)
```

Post-change bitwise guard:

```text
cmp housing_scale: identical
cmp heart_scale: identical
cmp iris.scale: identical
```

Optimization attempted: a surgical `kernel.rs` hot-path cleanup that caches sparse slice lengths/node references in `dot` and caches `self.x[i]`/`self.x[j]` once in `Kernel::evaluate`. This preserves arithmetic, operation order, and iteration order; no unsafe/SIMD/Qfloat changes were used.

Benchmark command:

```sh
BENCH_WARMUP=3 BENCH_RUNS=30 python3 scripts/benchmark_compare.py 2>&1 | tee .sc/jeh-bench.log | tail -20
```

Observed train_probability summary from the generated benchmark report before reverting the forbidden generated report artifacts:

```text
| `train_probability` | 30 | 7.328 | 7.001 | 1.074 | 1.581 | 2.058 |
```

Worst train_probability cases observed:

```text
| `s1_t3_iris_scale` | `train_probability` | 2.058 |
| `s3_t0_housing_scale` | `train_probability` | 1.730 |
| `s4_t0_housing_scale` | `train_probability` | 1.399 |
```

Result: the safe kernel-only cleanup preserved bitwise output but did not close the worst-case benchmark target (≤1.15). The new worst cases are tiny/non-RBF cases where process/noise overhead dominates the median ratio, and the original housing_scale ε-SVR RBF probability case is no longer the worst item in the benchmark report. Further changes likely require broader algorithmic/cache strategy work or ask-first areas, so this round stops on the documented negative-result path rather than forcing a non-bitwise-safe change.

Differential suite:

```text
Differential suite complete: 240 pass, 0 warn, 0 fail, 10 skip
Wrote /home/rfrantz/Projects/libsvm-rs/reference/differential_results.json
Wrote /home/rfrantz/Projects/libsvm-rs/reference/differential_report.md
```

## 2026-10-09: SVR training gap closed (libsvm-rs-eaz)

Machine: Linux 6.18 (WSL2), AMD Ryzen 9 9900X, rustc 1.93.1 (pinned), C reference
`vendor/libsvm` built with `g++ -O3`.

### What changed

1. `solver.rs` and `qmatrix.rs`: the hot loops (working-set selection for both solver
   variants, the gradient update, and the sign and index copy in `SvrQ::get_q`) work on
   slices cut to `active_size` or `len`. The compiler can then drop the per-element bounds
   checks. Arithmetic, operation order and iteration order are unchanged.
2. Workspace `Cargo.toml`: `[profile.release]` with `lto = "fat"` and `codegen-units = 1`.
   This applies to the CLI binaries and benchmarks built in this workspace. It does not
   apply to crates that depend on `libsvm-rs`.

### Output identity

- 74 model files (every train configuration of `scripts/benchmark_compare.py`, with and
  without `-b 1`, plus `-s 3/4 -t 0/2` on `gen_regression_sparse.scale`), trained by a
  release build of the unchanged code and of the changed code: 74/74 byte-identical.
- `DIFF_SCOPE=full scripts/run_differential_suite.py` run on the unchanged and on the
  changed code on this machine: all 250 cases identical in every field
  (240 pass / 0 warn / 0 fail / 10 skip). The committed `reference/differential_report.md`
  (237 / 3 / 0 / 10) was made on macOS with datasets generated there; regenerating the
  datasets and the upstream build on Linux changes the counts for both versions alike, so
  the committed report was left as it is.

### Instruction counts (callgrind, plain `svm-train -q`, total Ir)

| Command | C | Rust before | LTO only | Rust after | before/C | after/C |
|---|---:|---:|---:|---:|---:|---:|
| `-s 3 -t 0 housing_scale` | 141,928,298 | 203,086,725 | 189,980,506 | 158,918,463 | 1.431 | 1.120 |
| `-s 4 -t 0 housing_scale` | 148,051,370 | 234,636,440 | 213,365,046 | 165,097,332 | 1.585 | 1.115 |
| `-s 4 -t 2 housing_scale` | 100,497,742 | 137,502,038 | 129,033,292 | 111,993,367 | 1.368 | 1.114 |
| `-s 3 -t 0 gen_regression_sparse` | 95,491,756 | 142,834,155 | 131,906,173 | 98,492,828 | 1.496 | 1.031 |
| `-s 4 -t 2 gen_regression_sparse` | 14,495,955 | 16,067,644 | 15,189,623 | 13,683,940 | 1.108 | 0.944 |

The `gen_regression_sparse` rows are a held-out dataset that the benchmark does not use.

### In-process training time (criterion, `housing_scale`, median)

| Benchmark | Before (ms) | After (ms) | Change |
|---|---:|---:|---:|
| `svr_epsilon_linear` | 5.632 | 5.206 | -7.6 % |
| `svr_epsilon_rbf` | 4.686 | 4.390 | -6.3 % |
| `svr_nu_linear` | 6.754 | 5.610 | -16.9 % |
| `svr_nu_rbf` | 3.874 | 3.457 | -10.8 % |

### CLI benchmark (`BENCH_WARMUP=3 BENCH_RUNS=30 scripts/benchmark_compare.py`)

Rust/C median ratio, aggregate (worst case in brackets):

| Operation | Before | After, run 1 | After, run 2 |
|---|---:|---:|---:|
| `train` | 0.904 (1.23) | 0.893 (1.03) | 0.907 (1.02) |
| `train_probability` | 1.007 (1.29) | 0.969 (1.12) | 0.979 (1.11) |
| `predict` | 0.886 (1.00) | 0.824 (1.01) | 0.828 (1.00) |
| `predict_probability` | 0.905 (1.05) | 0.842 (1.01) | 0.838 (1.01) |

Target cases (before / after run 1 / after run 2): `s3_t0_housing_scale` train
1.105 / 0.993 / 1.013 and train_probability 1.237 / 1.084 / 1.083; `s4_t0_housing_scale`
train 1.233 / 0.951 / 0.932 and train_probability 1.288 / 0.928 / 0.950.

One case rose by more than 0.05: `s0_t0_iris_scale` (train 0.794 to 0.919). Its Rust time
did not change (1.50 ms to 1.49 ms) and its Rust instruction count fell (2.92 M to 2.73 M);
the C median was 1.89 ms in the before run and 1.62 ms in the after runs. The rise comes
from the C measurement, not from the Rust change.

`reference/benchmark_report.md` and `reference/benchmark_results.json` are from run 2.
