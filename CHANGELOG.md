# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.9.1] - 2026-10-09

### Added

- `crates/libsvm/LICENSE` and `crates/libsvm/NOTICE` went into the published
  crate. The `libsvm-rs` package on crates.io now carries the BSD-3-Clause
  license and the LIBSVM copyright notice. Releases up to 0.9.0 carried them
  only in the GitHub repository.
- `README.md` gained a Disclaimer section. It states that the software comes
  "as is", without warranty, and that it is not validated for engineering
  design, certification, or safety-critical use.
- `SECURITY.md` named the supported release and the GitHub private
  vulnerability-reporting link. GitHub issues are disabled for this repository.
- `crates/libsvm/benches/svm_bench.rs` gained four in-process SVR training
  benchmarks on `housing_scale`: `svr_epsilon_linear`, `svr_epsilon_rbf`,
  `svr_nu_linear`, and `svr_nu_rbf`.

### Changed

- `Solver` working-set selection, the gradient update, and `SvrQ::get_q` now
  index slices cut to the active length, so the compiler drops the per-element
  bounds checks. Arithmetic, operation order, and iteration order did not
  change: 74 model files from the benchmark configurations stayed
  byte-identical, and the 250-case differential suite gave identical results
  for every case. On `housing_scale`, SVR training fell from 1.37–1.59 to
  1.13–1.14 times the C instruction count (callgrind, without LTO). On one
  machine (Ryzen 9 9900X), the CLI `train_probability` median Rust/C time ratio
  went from 1.007 to 0.979, and the worst case from 1.29 to 1.11.
- `[profile.release]` in the workspace `Cargo.toml` now sets `lto = "fat"` and
  `codegen-units = 1`. It applies to the CLI binaries and benchmarks built from
  this repository. Crates that depend on `libsvm-rs` keep their own profile. On
  top of the solver change, it removed 1.4–2.2 % more SVR training instructions
  on `housing_scale`.
- The library and the three CLI crates now declare `#![forbid(unsafe_code)]`.
  The code had no `unsafe` before; the compiler now rejects any.

### Fixed

- The `MSRV (1.80.0)` CI job and the beta build column ran Rust 1.93.1, because
  `rust-toolchain.toml` overrode the toolchain that the job installed. They now
  call `cargo +1.80.0` and `cargo +beta`, and Rust 1.80.0 builds the workspace.
- The `predict.rs` test that compares Rust predictions with C `svm-predict`
  returned early in CI, because CI never built the C tools. The Test and Code
  coverage jobs now build them first. Library line coverage in CI now reads
  93.24 % (was 92.73 %).

### Security

- `Cargo.lock` now pins `crossbeam-epoch` 0.9.21 (was 0.9.18) for
  RUSTSEC-2026-0204, an invalid pointer dereference in `fmt::Pointer`. The
  crate comes in through the optional `rayon` feature. Projects that depend on
  `libsvm-rs` resolve their own lockfile; `cargo update -p crossbeam-epoch`
  moves them to the fixed version.
- `Cargo.lock` now pins `anyhow` 1.0.104 (was 1.0.102) for RUSTSEC-2026-0190.
  `anyhow` is a development dependency only.

### Documentation

- README numbers now match the committed reports: test count 202 (was "~124"),
  the performance table, coverage, and the parity tolerance (relative `1.5e-5`
  and absolute `1e-8`, instead of "~1e-8"). The README now states that the
  differential counts depend on the platform: 237/3/0/10 on macOS and
  240/0/0/10 on Linux for the same code.
- Implementation internals moved from the README to `docs/ARCHITECTURE.md`.

## [0.9.0] - 2026-06-11

### Added

- `SvmParameterBuilder` provides idiomatic construction-time validation for
  parameters, using the same checks as `svm_check_parameter`.
- Optional `serde` support for `SvmModel` and `SvmParameter`, with LIBSVM's
  integer enum codes preserved and the text model format remaining canonical.
- Optional `rayon` support for parallel cross-validation folds and parallel
  probability-estimation fold training.
- CI now verifies that the core library builds for `wasm32-unknown-unknown`.

### Changed

- MSRV raised from 1.75 to 1.80, required by rayon-core 1.13; previously the
  1.75 claim was unsatisfiable with the `rayon` feature.
- Linux probability estimation and probability cross-validation now replicate
  glibc `rand()` for the stratified shuffle, restoring C LIBSVM parity; Linux
  users of probability estimation or cross-validation should expect outputs to
  differ from 0.8.1 by design.
- The default repository branch is now `main`; downstream links, badges, clone
  commands, and automation that referenced `master` should be updated.
- Differential reference artifacts were re-baselined on macOS, and benchmark and
  integration demo artifacts were refreshed.
- Development tooling and CI pins were refreshed, including the Node 24 GitHub
  Actions runtime and current benchmark/dependency updates.

### Fixed

- The WebAssembly core-library CI job now installs the `wasm32-unknown-unknown`
  target for the pinned Rust toolchain before building.
- Benchmark CI now uses static, least-privilege permissions and the Criterion
  benchmark harness uses `std::hint::black_box` with current Criterion releases.

### Security

- Reject negative `gamma` for polynomial/RBF/sigmoid kernels at model load
  (text and serde paths); refreshed security audit covering the serde,
  rayon, builder, and PRNG surfaces (`SECURITY_AUDIT.md`).

### Documentation

- Added a migration guide for moving C LIBSVM and `libsvm-sys2` workflows to
  `libsvm-rs`.
- Added `VISION.md` and refreshed README, reference, and performance notes for
  the 0.9.0 release line.

## [0.8.1] - 2026-06-02

### Security

- Reject malformed classification models that omit the `nr_sv`/`label` lines.
  Such files previously loaded with empty vectors and then panicked (index out
  of bounds) during prediction; `validate_model_header` now requires
  `label.len() == nr_class` and `n_sv.len() == nr_class` for `c_svc`/`nu_svc`
  and returns a `ModelFormatError` instead.
- Guard the command-line tools against a bare `-` argument, which previously
  panicked (`svm-train-rs`, `svm-predict-rs`, `svm-scale-rs`); they now print
  usage and exit cleanly.
- Reject non-finite (`NaN`/`Inf`) values in model `gamma`, `coef0`, `rho`,
  support-vector coefficients, and feature values, and reject a negative
  `degree`, at load time.
- `cargo-deny` advisory policy now fails CI on yanked (`yanked = "deny"`) and
  unmaintained (`unmaintained = "all"`) crates.

### Changed

- Retain the upstream LIBSVM copyright (Chih-Chung Chang and Chih-Jen Lin)
  alongside the Rust port's. Restore the upstream BSD-3-Clause `COPYRIGHT` file
  for the vendored source and add a project `NOTICE` documenting provenance.

## [0.8.0] - 2026-04-22

### Added

- `LoadOptions` and explicit `load_*_from_reader_with_options` entrypoints for
  bounded problem/model loading.
- Fuzz corpus seeds and property tests for parser and model serialization
  stability.
- Supply-chain policy checks with `cargo-deny` and a pinned release
  reproduction guide.

### Changed

- Default problem/model loaders now treat input as untrusted and apply byte,
  line-length, support-vector, class-count, and feature-index caps.
- Root Rust toolchain is pinned to `1.93.1`; MSRV remains `1.75.0`.

### Security

- Hardened model parsing against header-driven allocation and inconsistent
  model metadata before support-vector storage is allocated.
- Model files now validate `rho`, `label`, `nr_sv`, probability metadata,
  precomputed-kernel rows, and support-vector feature ordering.
- Problem and model loaders reject over-limit input, oversized lines, embedded
  NUL bytes, malformed `index:value` tokens, non-ascending feature indices, and
  over-limit feature indices.
- Production library code now denies `unwrap`, `expect`, `panic!`, and
  `unreachable!` outside tests.
- No breaking API changes are expected; `LoadOptions` is additive and existing
  loader entrypoints delegate to `LoadOptions::default()`.

## [0.7.0] - 2026-02-12

### Added

- `metrics` module: `accuracy_percentage()`, `regression_metrics()` (public API)
- `util` module: `parse_feature_index()`, `MAX_FEATURE_INDEX` (public API)
- CLI integration tests — flag permutation and edge-case coverage
- Shared `cli_flag_helpers.rs` for property-based CLI testing

### Changed

- Deduplicated `group_classes` from train.rs + cross_validation.rs into util module
- Collapsed `parse_multiple_f64`/`parse_multiple_i32` into generic `parse_multiple<T>` in io.rs
- CLI arg parsing simplified via `parse_flag_arg()` helper in all 3 binaries
- svm-predict collects predictions into Vec before computing metrics (cleaner flow)

### Fixed

- `.tmp/` added to `.gitignore`

## [0.6.0] - 2026-02-11

### Added

- Testing and validation infrastructure:
  - Differential verification suite against upstream LIBSVM (250 test configurations)
  - Upstream lock file and CI validation (`reference/libsvm_upstream_lock.json`)
  - Deterministic synthetic dataset generation (6 families: binary, multiclass, one-class, regression, sparse, extreme scale)
  - Reference build pipeline with provenance tracking (`scripts/setup_reference_libsvm.sh`)
  - Coverage threshold checking with CI enforcement (93.19% line coverage, 92.86% function coverage)
  - Benchmark comparison framework (`scripts/benchmark_compare.py`)
  - Tolerance policy documentation (`reference/tolerance_policy.md`)
- Precomputed kernel support:
  - Full training and prediction support for precomputed kernels (kernel_type=4)
  - Validation and reference data for heart_scale, iris.scale, housing_scale
- CLI integration tests:
  - `svm-train-rs`: model file output, cross-validation, quiet mode
  - `svm-predict-rs`: prediction output, probability mode rejection for non-prob models, quiet mode
  - `svm-scale-rs`: scaling output, save/restore parameters, negative index hardening, inconsistent bounds checking
- Library enhancements:
  - Precomputed kernel evaluation path in `Kernel`
  - Extended probability module with NaN/Inf guards
  - Parameter validation for precomputed kernels
  - Helper functions for querying model properties by SVM type

### Changed

- README: comprehensive Phase 6 status update with verification metrics
- CI workflow: added upstream lock validation, coverage gates
- Solver output: additional stability checks for edge cases
- Probability estimation: improved numerical stability for one-class and SVR

### Fixed

- Model I/O: robust header parsing with oversized count guards (continued from v0.5.1)
- Kernel evaluation: correct precomputed kernel access and bounds checking
- Probability estimation: handle edge cases with insufficient or degenerate samples
- Scale CLI: prevent panic on negative feature indices

### Security

- Added SECURITY_AUDIT.md with RustSec audit results (zero findings)
- Hardened parsing against malicious inputs (oversized headers, negative indices)

## [0.5.1] - 2026-02-09

### Fixed

- Model loading: harden against oversized header counts that could cause memory allocation failures
- Cross-validation: fix probability prediction output and zero-fold edge case (now clamps to leave-one-out)
- Code quality: resolve 43 clippy warnings (collapsible_else_if, needless_range_loop, excessive_precision, field_reassign_with_default, manual_memcpy, etc.)

## [0.5.0] - 2026-02-09

### Added

- CLI tools: `svm-train-rs`, `svm-predict-rs`, `svm-scale-rs` matching C LIBSVM interface
- Quiet mode: `set_quiet(true)` suppresses all solver output; `-q` flag in all CLIs
- Reference comparison pipeline: scripts to generate and compare outputs against C LIBSVM
- Criterion benchmarks for training and prediction
- Property tests for determinism and label-range validation
- Fuzz targets for problem and model file parsing
- 4 examples: basic_train, predict_from_model, cross_validation, probability
- CI: build matrix (3 OS × 2 toolchains), MSRV check, Miri, security audit, coverage, benchmarks
- Release workflow with prebuilt binaries for Linux, macOS (x86+arm), Windows

### Changed

- Solver output formatting: `obj`, `rho`, `mae` now use `{:.6}` (matches C's `%f`)
- All internal `eprintln!` calls replaced with `info()` respecting quiet mode

## [0.4.0] - 2026-02-09

### Added

- Probability estimates: Sigmoid probability model training and prediction
- Cross-validation: k-fold cross-validation with stratified splits
- `probability` module: `SigmoidTrainer`, `sigmoid_predict`, multiclass calibration
- `cross_validation` module: `StratifiedKFold` for proper class distribution
- 20 new unit tests covering probability and CV workflows

### Changed

- `SvmModel::predict_probability_multiclass()` now uses trained sigmoid probabilities
- Solver returns `alpha_sum` for probability fitting (one-vs-rest framework)

### Fixed

- Multiclass probability predictions now sum correctly to 1.0

## [0.3.0] - 2026-02-09

### Added

- Full SMO solver for all 5 SVM types: C-SVC, ν-SVC, one-class, ε-SVR, ν-SVR
- WSS3 working-set selection (second-order heuristic, Fan et al. JMLR 2005)
- Shrinking heuristic with gradient reconstruction
- `QMatrix` trait with `SvcQ`, `OneClassQ`, `SvrQ` implementations
- `svm_train` function producing `SvmModel` compatible with C LIBSVM
- Multiclass support via one-vs-one with class grouping and sv_coef assembly
- 50 tests (12 new), verified against C LIBSVM reference outputs

### Fixed

- `Cache::swap_index` — added column swap loop (critical for shrinking correctness)
- Kernel refactored to `Vec<&[SvmNode]>` for swappable data point references

## [0.2.0] - 2026-02-09

### Added

- Core types: `SvmNode`, `SvmProblem`, `SvmParameter`, `SvmModel`
- All 5 kernel functions (linear, polynomial, RBF, sigmoid, precomputed)
- LRU kernel cache
- Model and problem I/O (LIBSVM text format, byte-exact roundtrip)
- Prediction (zero mismatches against C `svm-predict` on heart_scale)
- Parameter validation with ν-SVC feasibility check
- 38 tests

[0.9.1]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ricardofrantz/libsvm-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ricardofrantz/libsvm-rs/commits/v0.2.0
