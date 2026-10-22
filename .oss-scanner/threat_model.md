# Threat model: libsvm-rs

## What this project does and where untrusted input enters
libsvm-rs is a pure Rust reimplementation of LIBSVM. It reads and writes the LIBSVM
text formats and gives the same numbers as the C++ reference. Users load files they
did not write: models downloaded from a colleague or the web, and data files from
public datasets.

Untrusted input enters through:
- `io::load_model*` and `io::load_problem*` (`crates/libsvm/src/io.rs`). The text
  loaders are the main attack surface.
- The `serde` feature: `Deserialize` for `SvmModel` and `SvmParameter` (`types.rs`).
  It must route through the same `validate_model()` as the text loader.
- The command-line tools `svm-train-rs`, `svm-predict-rs` and `svm-scale-rs`
  (`bins/`), which pass file paths from the user to the loaders.

Training parameters set in code by the caller are trusted. A bad parameter that
`SvmParameter::validate()` should reject but does not is still a bug worth reporting.

## Components that matter most / least
- Most: `io.rs` (both loaders, `LoadOptions` limits, `validate_model`), `types.rs`
  (serde path), `predict.rs` (anything a loaded model can make it index out of bounds).
- In scope: `train.rs`, `solver.rs`, `kernel.rs`, `cache.rs`, `probability.rs`,
  `cross_validation.rs` (rayon parallel paths).
- Out of scope: `vendor/` and `reference/` (upstream C++ LIBSVM, kept only to compare
  numbers), `examples/`, `scripts/`, benchmarks.

## How to exercise it
- `cargo test --workspace --all-features` runs the full suite.
  `crates/libsvm/tests/malicious_input.rs` with fixtures in
  `crates/libsvm/tests/malicious/` is the regression corpus for hostile files.
- Fuzz targets are built in the image:
  `cd crates/libsvm && cargo +nightly-2026-03-08 fuzz run parse_model` (or
  `parse_problem`). Seed corpora are in `crates/libsvm/fuzz/corpus/`.
- CLI: `target/debug/svm-predict-rs <test_file> <model_file> <output>`. Sample data
  and models are in `data/` (for example `data/heart_scale`, `data/heart_scale.model`).
- The C++ reference tools are built in the image at `vendor/libsvm/svm-train`,
  `svm-predict` and `svm-scale`. Use them to check whether libsvm-rs accepts or predicts
  differently from LIBSVM on the same file.
- `SECURITY_AUDIT.md` lists earlier findings and fixes. Please do not re-report those
  unless the fix is incomplete.

## How we rate severity
The crate has no `unsafe` code, so memory corruption is not expected. The realistic
failures are panics, unbounded memory or CPU use, and silent wrong results.
- Critical: memory unsafety reached from a model or data file (for example through a
  dependency), or any way for a file to run code.
- High: a model file that passes loading and then makes `predict` return wrong
  results silently, or makes training or prediction read out of bounds (a panic in
  `predict` on a loaded model counts here); a bypass of the `LoadOptions` limits
  (`max_bytes`, `max_line_len`, `max_sv`, `max_nr_class`, `max_feature_index`) under
  the default options.
- Medium: a panic or abort while loading a file under the default options; memory or
  CPU use far larger than the input size allows (more than about 10x the default
  `max_bytes`) under the default options; differences from the C++ reference on
  valid input that change predictions.
- Low: the same problems only with `LoadOptions::trusted_input()` or other limits the
  caller raised on purpose; misleading error messages.

## Anything to leave alone
- Slow training on large but valid data is expected (it is an SVM solver), not DoS.
- Floating-point differences in the last bits against C++ LIBSVM, when the predicted
  labels match.
- Out-of-memory when the caller chose `LoadOptions::trusted_input()`.
