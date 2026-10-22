# Migration guide: C LIBSVM to libsvm-rs

This guide maps an existing C LIBSVM workflow to the Rust-native `libsvm-rs` crate and CLI tools. It is for users of upstream LIBSVM, direct `svm.h` calls, or Rust FFI wrappers such as `libsvm-sys2`.

Parity here means numerical equivalence within the project tolerance policy, not bitwise identity. The same framing is used in the README: the differential suite has no hard failures under the default policy, with a small set of documented warnings; residual drift comes from training-side floating-point numerics, not prediction logic. See [`../reference/tolerance_policy.md`](../reference/tolerance_policy.md) and [`../reference/differential_report.md`](../reference/differential_report.md).

## 1. CLI mapping

The Rust binaries keep the upstream command shape and flag spelling, with `-rs` added to the binary names.

| C LIBSVM command | libsvm-rs command | Migration note |
|---|---|---|
| `svm-train [options] training_set_file [model_file]` | `svm-train-rs [options] training_set_file [model_file]` | Same positional arguments. If `model_file` is omitted, writes `<training_set_file>.model`. |
| `svm-predict [options] test_file model_file output_file` | `svm-predict-rs [options] test_file model_file output_file` | Same positional arguments. |
| `svm-scale [options] data_filename` | `svm-scale-rs [options] data_filename` | Same positional argument; scaled data is written to stdout. |

| Tool | Flag | C meaning | libsvm-rs equivalent / difference |
|---|---|---|---|
| train | `-s svm_type` | SVM type: `0` C-SVC, `1` nu-SVC, `2` one-class, `3` epsilon-SVR, `4` nu-SVR | Same values, mapped to `SvmType::{CSvc, NuSvc, OneClass, EpsilonSvr, NuSvr}`. |
| train | `-t kernel_type` | Kernel: `0` linear, `1` polynomial, `2` RBF, `3` sigmoid, `4` precomputed | Same values, mapped to `KernelType::{Linear, Polynomial, Rbf, Sigmoid, Precomputed}`. |
| train | `-d degree` | Polynomial degree, default `3` | Same; sets `SvmParameter::degree`. |
| train | `-g gamma` | Gamma, default `1/num_features` | Same; `gamma = 0` means auto-detect from the maximum feature index. |
| train | `-r coef0` | Polynomial/sigmoid `coef0`, default `0` | Same; sets `SvmParameter::coef0`. |
| train | `-c cost` | Cost `C`, default `1` | Same; sets `SvmParameter::c`. |
| train | `-n nu` | `nu`, default `0.5` | Same; sets `SvmParameter::nu`. |
| train | `-p epsilon` | Epsilon-SVR loss epsilon, default `0.1` | Same; sets `SvmParameter::p`. |
| train | `-m cachesize` | Kernel cache size in MB, default `100` | Same unit and default; sets `SvmParameter::cache_size`. |
| train | `-e epsilon` | Solver termination tolerance, default `0.001` | Same; sets `SvmParameter::eps`. |
| train | `-h shrinking` | Shrinking heuristic `0`/`1`, default `1` | Same; sets `SvmParameter::shrinking`. |
| train, predict | `-b probability_estimates` | Train or use probability estimates, `0`/`1`, default `0` | Same flag. In prediction, one-class SVM supports probability output with the Rust model's density marks; the CLI rejects a request if the loaded model lacks probability data. |
| train | `-wi weight` | Set class `i` weight to `weight*C` for C-SVC | Same composite flag syntax, for example `-w1 2.0` or `-w-1 0.5`; stored as `SvmParameter::weight`. |
| train | `-v n` | `n`-fold cross-validation | Same; `n` must be at least `2`; calls `cross_validation::svm_cross_validation`. |
| train, predict | `-q` | Quiet mode | Same; calls `libsvm_rs::set_quiet(true)` and suppresses solver/progress output. |
| scale | `-l lower` | Feature scaling lower limit, default `-1` | Same. |
| scale | `-u upper` | Feature scaling upper limit, default `+1` | Same. |
| scale | `-y y_lower y_upper` | Target scaling limits | Same. |
| scale | `-s save_filename` | Save scaling parameters | Same scale-parameter text format. |
| scale | `-r restore_filename` | Restore scaling parameters | Same scale-parameter text format; cannot be combined with `-s`. |

## 2. C API to Rust API

These rows are verified against `vendor/libsvm/svm.h` and the public exports in `crates/libsvm/src/lib.rs` / `types.rs`.

| C LIBSVM (`svm.h`) | libsvm-rs | Migration note |
|---|---|---|
| `struct svm_node { int index; double value; }` | `SvmNode { index: i32, value: f64 }` | Same sparse `index:value` idea. C rows end with an `index = -1` sentinel; Rust rows are `Vec<SvmNode>` slices and do not store a sentinel. |
| `struct svm_problem { int l; double *y; struct svm_node **x; }` | `SvmProblem { labels: Vec<f64>, instances: Vec<Vec<SvmNode>> }` | `labels.len()` / `instances.len()` replace `l`; Rust owns the vectors. |
| `struct svm_parameter` | `SvmParameter` | Fields map directly: `svm_type`, `kernel_type`, `degree`, `gamma`, `coef0`, `cache_size`, `eps`, `C`→`c`, `nu`, `p`, `shrinking`, `probability`; `nr_weight`, `weight_label`, `weight` become `weight: Vec<(i32, f64)>`. |
| `enum` integer constants `C_SVC`, `NU_SVC`, `ONE_CLASS`, `EPSILON_SVR`, `NU_SVR` | `SvmType::{CSvc, NuSvc, OneClass, EpsilonSvr, NuSvr}` | Same discriminant values `0..4`. |
| `enum` integer constants `LINEAR`, `POLY`, `RBF`, `SIGMOID`, `PRECOMPUTED` | `KernelType::{Linear, Polynomial, Rbf, Sigmoid, Precomputed}` | Same discriminant values `0..4`. |
| `struct svm_model *svm_train(...)` | `train::svm_train(&SvmProblem, &SvmParameter) -> SvmModel` | Returns an owned `SvmModel`, not a pointer. |
| `void svm_cross_validation(..., int nr_fold, double *target)` | `cross_validation::svm_cross_validation(&SvmProblem, &SvmParameter, usize) -> Vec<f64>` | Return vector replaces caller-allocated output buffer. |
| `int svm_save_model(const char *, const struct svm_model *)` | `io::save_model(&Path, &SvmModel) -> Result<(), SvmError>` | Structured `Result` replaces integer status. |
| `struct svm_model *svm_load_model(const char *)` | `io::load_model(&Path) -> Result<SvmModel, SvmError>` | Returns an owned model or structured parse/I/O error. |
| `int svm_get_svm_type(const struct svm_model *)` | `svm_get_svm_type(&SvmModel) -> SvmType` or `model.svm_type()` | Rust returns the enum. |
| `int svm_get_nr_class(const struct svm_model *)` | `svm_get_nr_class(&SvmModel) -> usize` or `model.class_count()` | Same concept, Rust size type. |
| `void svm_get_labels(const struct svm_model *, int *label)` | `svm_get_labels(&SvmModel) -> &[i32]` or `model.labels()` | Borrowed slice replaces caller-allocated buffer. |
| `void svm_get_sv_indices(const struct svm_model *, int *sv_indices)` | `svm_get_sv_indices(&SvmModel) -> &[usize]` or `model.support_vector_indices()` | Borrowed slice replaces caller-allocated buffer. |
| `int svm_get_nr_sv(const struct svm_model *)` | `svm_get_nr_sv(&SvmModel) -> usize` or `model.support_vector_count()` | Same concept. |
| `double svm_get_svr_probability(const struct svm_model *)` | `svm_get_svr_probability(&SvmModel) -> Option<f64>` or `model.svr_probability()` | `None` represents unavailable probability information. |
| `double svm_predict_values(..., double *dec_values)` | `predict::predict_values(&SvmModel, &[SvmNode], &mut [f64]) -> f64` | Caller still provides the decision-value buffer. |
| `double svm_predict(...)` | `predict::predict(&SvmModel, &[SvmNode]) -> f64` | Same returned label/value. |
| `double svm_predict_probability(..., double *prob_estimates)` | `predict::predict_probability(&SvmModel, &[SvmNode]) -> Option<(f64, Vec<f64>)>` | `None` when the model lacks probability data; probability vector is returned owned. |
| `const char *svm_check_parameter(...)` | `check_parameter(&SvmProblem, &SvmParameter) -> Result<(), SvmError>` | `Ok(())` replaces null; `Err` carries the reason. `SvmParameter::validate()` is available for data-independent checks. |
| `int svm_check_probability_model(...)` | `svm_check_probability_model(&SvmModel) -> bool` or `model.has_probability_model()` | Same boolean meaning. |
| `void svm_set_print_string_function(...)` | `set_quiet(bool)` | The current public Rust API exposes quiet-mode suppression, not an arbitrary print callback. |
| `svm_free_model_content`, `svm_free_and_destroy_model`, `svm_destroy_param` | No direct call | Rust ownership and `Drop` free the model/parameter contents automatically. |

Minimal Rust equivalent of a C train/predict/save/load flow:

```rust
use libsvm_rs::io::{load_model, load_problem, save_model};
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::{check_parameter, SvmParameter};
use std::path::Path;

let problem = load_problem(Path::new("train.libsvm"))?;
let param = SvmParameter::default();
check_parameter(&problem, &param)?;
let model = svm_train(&problem, &param);
save_model(Path::new("model.txt"), &model)?;
let loaded = load_model(Path::new("model.txt"))?;
let y = predict(&loaded, &problem.instances[0]);
# Ok::<(), libsvm_rs::SvmError>(())
```

## 3. Memory-model differences

C LIBSVM exposes pointer ownership in the API. You allocate `svm_problem`, arrays of `svm_node`, parameter arrays, and sometimes a loaded model, then clean them up with `svm_free_model_content`, `svm_free_and_destroy_model`, and `svm_destroy_param`. The C `svm_model` also carries `free_sv` to distinguish loaded support-vector storage from training-data-backed support vectors.

`libsvm-rs` uses normal Rust ownership instead:

- `SvmProblem`, `SvmParameter`, and `SvmModel` own their vectors.
- Borrowed prediction inputs are `&[SvmNode]`; there is no `-1` terminator.
- There is no `free_sv` flag and no public free/destroy function.
- Model and parameter memory is released when the owning value is dropped.
- Loader and saver APIs return `Result<_, SvmError>` instead of null pointers or integer status codes.

## 4. Format compatibility and parity

Problem files, model files, and scale-parameter files keep the LIBSVM text formats so existing data pipelines can move one piece at a time.

- Problem files use standard LIBSVM sparse rows: `label index:value index:value ...`; missing features are implicit zeroes.
- Model files are read and written in LIBSVM text format. Floating-point model values are formatted with `%.17g`-equivalent precision for round-trip fidelity.
- `svm-scale-rs` reads and writes the same scale-parameter file shape used by `svm-scale` (`y` section when target scaling is enabled, then `x` bounds and feature ranges).
- Precomputed-kernel problem rows follow the upstream convention: feature index `0` stores the sample id, and positive indices hold kernel values.

The compatibility claim is not bitwise identity. It is numerical equivalence within the documented tolerances, with the same wording as the README parity claim: no hard differential failures under the default policy, documented warnings where training numerics drift slightly, and prediction logic cross-checked. Use upstream LIBSVM directly if your process requires bit-for-bit identity.

## 5. Migrating from `libsvm-sys2` or the `libsvm` crate

If your Rust code currently goes through `libsvm-sys2`, migrate one layer at a time: replace raw pointer construction with owned Rust data, then replace FFI calls with crate APIs.

| FFI / binding pattern | libsvm-rs replacement |
|---|---|
| Build `svm_node` arrays and append `index = -1` | Build `Vec<SvmNode>` without a sentinel. |
| Fill `svm_problem.l`, `y`, and `x` manually | Use `SvmProblem { labels, instances }` or `io::load_problem`. |
| Fill `svm_parameter` and weight arrays manually | Use `SvmParameter::default()` and set fields; use `weight: Vec<(i32, f64)>` for class weights. |
| Call unsafe `svm_train` through FFI | Call safe `train::svm_train`. |
| Call unsafe `svm_predict`, `svm_predict_values`, `svm_predict_probability` | Call `predict::predict`, `predict::predict_values`, `predict::predict_probability`. |
| Call `svm_load_model` / `svm_save_model` | Call `io::load_model` / `io::save_model`. |
| Call `svm_check_parameter` and inspect a nullable C string | Call `check_parameter` and handle `Result<(), SvmError>`. |
| Call C free functions in `Drop` wrappers | Remove that code; owned Rust values drop automatically. |

The higher-level `libsvm` crate wraps the C implementation. Its exact type names differ from the raw FFI layer, but the migration target is the same: represent rows as `Vec<SvmNode>`, configure `SvmParameter`, train with `train::svm_train`, persist with `io::{load_model, save_model}`, and predict with `predict::*`.

## 6. Untrusted-input behavior

`libsvm-rs` treats problem and model files as untrusted text input by default. This is intentionally stricter than traditional C LIBSVM parsing.

Default loaders enforce `LoadOptions` caps for bytes, line length, support-vector count, class count, and feature indices. Problem files reject malformed sparse tokens, non-ascending feature indices, over-limit feature indices, oversized input, oversized lines, and embedded NUL bytes. Model loading also validates header consistency before allocating support-vector storage.

Use the default APIs for external files:

```rust
use libsvm_rs::io::{load_model, load_problem};
use std::path::Path;

let problem = load_problem(Path::new("input.libsvm"))?;
let model = load_model(Path::new("model.txt"))?;
# Ok::<(), libsvm_rs::SvmError>(())
```

For files whose source and size are already controlled, opt into relaxed limits explicitly with `LoadOptions::trusted_input()` and the `*_with_options` reader APIs:

```rust
use libsvm_rs::io::{load_model_from_reader_with_options, LoadOptions};
use std::fs::File;
use std::io::BufReader;

let reader = BufReader::new(File::open("model.txt")?);
let model = load_model_from_reader_with_options(reader, &LoadOptions::trusted_input())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The loader checks bound parsing work and memory use. They do not authenticate a model, prove dataset provenance, or make model outputs safe for regulated decisions.
