# Function Parity Matrix (LIBSVM C/C++ vs libsvm-rs)

Date: 2026-02-10

Legend:
- `Equivalent`: behavior exists with matching semantics.
- `Equivalent (API shape)`: behavior exists but Rust API is idiomatic, not C-style.
- `Intentional divergence`: deliberate difference from C API/model due Rust design.
- `Missing`: no direct helper/API equivalent currently exposed.

## 1) Public C API (`vendor/libsvm/svm.h`)

| Original function | C location | Rust equivalent | Rust location | Status | Notes |
|---|---|---|---|---|---|
| `svm_train` | `vendor/libsvm/svm.cpp:2179` | `svm_train` | `crates/libsvm/src/train.rs:292` | Equivalent | Core training behavior is implemented for all 5 SVM types. |
| `svm_cross_validation` | `vendor/libsvm/svm.cpp:2438` | `svm_cross_validation` | `crates/libsvm/src/cross_validation.rs:42` | Equivalent | Stratified classification and regression/one-class CV both implemented. |
| `svm_save_model` | `vendor/libsvm/svm.cpp:2760` | `save_model` / `save_model_to_writer` | `crates/libsvm/src/io.rs:221`, `crates/libsvm/src/io.rs:228` | Equivalent (API shape) | Same model format semantics with Rust `Result` return type. |
| `svm_load_model` | `vendor/libsvm/svm.cpp:3015` | `load_model` / `load_model_from_reader` | `crates/libsvm/src/io.rs:333`, `crates/libsvm/src/io.rs:340` | Equivalent (API shape) | Same model parsing semantics with Rust `Result`. |
| `svm_get_svm_type` | `vendor/libsvm/svm.cpp:2559` | `svm_get_svm_type` | `crates/libsvm/src/types.rs:298` | Equivalent | Direct C-style helper now exposed in library API. |
| `svm_get_nr_class` | `vendor/libsvm/svm.cpp:2564` | `svm_get_nr_class` | `crates/libsvm/src/types.rs:303` | Equivalent | Direct C-style helper now exposed in library API. |
| `svm_get_labels` | `vendor/libsvm/svm.cpp:2569` | `svm_get_labels` | `crates/libsvm/src/types.rs:308` | Equivalent | Direct C-style helper now exposed in library API. |
| `svm_get_sv_indices` | `vendor/libsvm/svm.cpp:2576` | `svm_get_sv_indices` | `crates/libsvm/src/types.rs:313` | Equivalent | Direct C-style helper now exposed in library API. |
| `svm_get_nr_sv` | `vendor/libsvm/svm.cpp:2583` | `svm_get_nr_sv` | `crates/libsvm/src/types.rs:318` | Equivalent | Direct C-style helper now exposed in library API. |
| `svm_get_svr_probability` | `vendor/libsvm/svm.cpp:2588` | `svm_get_svr_probability` | `crates/libsvm/src/types.rs:323` | Equivalent | Returns `Option<f64>` for SVR sigma metadata. |
| `svm_predict_values` | `vendor/libsvm/svm.cpp:2600` | `predict_values` | `crates/libsvm/src/predict.rs:21` | Equivalent | Decision-value API exists for all SVM types. |
| `svm_predict` | `vendor/libsvm/svm.cpp:2682` | `predict` | `crates/libsvm/src/predict.rs:105` | Equivalent | Direct prediction API exists. |
| `svm_predict_probability` | `vendor/libsvm/svm.cpp:2697` | `predict_probability` | `crates/libsvm/src/predict.rs:131` | Equivalent (API shape) | Returns `Option<(label, probs)>` rather than writing into buffer. |
| `svm_free_model_content` | `vendor/libsvm/svm.cpp:3124` | Rust drop semantics | `crates/libsvm/src/types.rs:226` | Intentional divergence | Explicit free function not needed with ownership/RAII. |
| `svm_free_and_destroy_model` | `vendor/libsvm/svm.cpp:3162` | Rust drop semantics | `crates/libsvm/src/types.rs:226` | Intentional divergence | Explicit destroy function not needed in Rust. |
| `svm_destroy_param` | `vendor/libsvm/svm.cpp:3172` | Rust drop semantics | `crates/libsvm/src/types.rs:68` | Intentional divergence | `Vec` cleanup automatic. |
| `svm_check_parameter` | `vendor/libsvm/svm.cpp:3178` | `check_parameter` | `crates/libsvm/src/types.rs:179` | Equivalent | Includes nu-SVC feasibility check. |
| `svm_check_probability_model` | `vendor/libsvm/svm.cpp:3296` | `svm_check_probability_model` | `crates/libsvm/src/types.rs:328` | Equivalent | Direct helper now exposed in library API. |
| `svm_set_print_string_function` | `vendor/libsvm/svm.cpp:3306` | `set_quiet` | `crates/libsvm/src/lib.rs:26` | Intentional divergence | Quiet toggle exists; arbitrary callback hook is not exposed. |

## 2) `svm.cpp` utility/helper functions

| Original function | C location | Rust equivalent | Rust location | Status | Notes |
|---|---|---|---|---|---|
| `min` (template) | `vendor/libsvm/svm.cpp:19` | `f64::min`, `std::cmp::min` | `crates/libsvm/src/*` | Equivalent (API shape) | Uses std methods rather than custom template. |
| `max` (template) | `vendor/libsvm/svm.cpp:22` | `f64::max`, `std::cmp::max` | `crates/libsvm/src/*` | Equivalent (API shape) | Uses std methods rather than custom template. |
| `swap` (template) | `vendor/libsvm/svm.cpp:24` | `.swap(...)` | `crates/libsvm/src/*` | Equivalent (API shape) | Native Rust swap on vectors/slices. |
| `clone` (template helper) | `vendor/libsvm/svm.cpp:25` | `.to_vec()` / `clone()` | `crates/libsvm/src/solver.rs:88` | Equivalent (API shape) | Rust owned buffers replace raw alloc/memcpy. |
| `powi` | `vendor/libsvm/svm.cpp:30` | `powi` | `crates/libsvm/src/kernel.rs:16` | Equivalent | Same fast exponentiation behavior. |
| `print_string_stdout` | `vendor/libsvm/svm.cpp:45` | `info` -> `eprint!` when not quiet | `crates/libsvm/src/lib.rs:31` | Equivalent (API shape) | Rust prints to stderr for diagnostics. |
| `info` | `vendor/libsvm/svm.cpp:52` | `info` | `crates/libsvm/src/lib.rs:31` | Equivalent (API shape) | No varargs formatting callback; formatting done at call sites. |
| `readline` (model loader helper) | `vendor/libsvm/svm.cpp:2864` | `BufRead::lines()` parsing | `crates/libsvm/src/io.rs:340` | Equivalent (API shape) | Rust line iteration replaces manual growable C buffer. |
| `read_model_header` | `vendor/libsvm/svm.cpp:2894` | `load_model_from_reader` header parsing branch | `crates/libsvm/src/io.rs:340` | Equivalent | Header keys mapped with structured error handling. |

## 3) Cache, kernel, solver, and Q-matrix internals (`svm.cpp`)

| Original function | C location | Rust equivalent | Rust location | Status | Notes |
|---|---|---|---|---|---|
| `Cache::Cache` | `vendor/libsvm/svm.cpp:98` | `Cache::new` | `crates/libsvm/src/cache.rs:47` | Equivalent | Same cache-size normalization and LRU sentinel strategy. |
| `Cache::~Cache` | `vendor/libsvm/svm.cpp:107` | Rust drop on `Cache` fields | `crates/libsvm/src/cache.rs:38` | Intentional divergence | Explicit destructor unnecessary. |
| `Cache::lru_delete` | `vendor/libsvm/svm.cpp:114` | `lru_delete` | `crates/libsvm/src/cache.rs:73` | Equivalent | Same linked-list unlink semantics via indices. |
| `Cache::lru_insert` | `vendor/libsvm/svm.cpp:121` | `lru_insert` | `crates/libsvm/src/cache.rs:84` | Equivalent | Same list tail insertion behavior. |
| `Cache::get_data` | `vendor/libsvm/svm.cpp:130` | `get_data` | `crates/libsvm/src/cache.rs:106` | Equivalent | Returns cached prefix length (`start`) for fill. |
| `Cache::swap_index` | `vendor/libsvm/svm.cpp:160` | `swap_index` | `crates/libsvm/src/cache.rs:148` | Equivalent | Includes row swap, column swap, and partial-row eviction logic. |
| `Kernel::swap_index` (inline) | `vendor/libsvm/svm.cpp:215` | `Kernel::swap_index` | `crates/libsvm/src/kernel.rs:182` | Equivalent | Swaps data references and precomputed RBF squares. |
| `Kernel::kernel_linear` (inline) | `vendor/libsvm/svm.cpp:235` | `Kernel::evaluate` linear branch | `crates/libsvm/src/kernel.rs:157` | Equivalent | |
| `Kernel::kernel_poly` (inline) | `vendor/libsvm/svm.cpp:239` | `Kernel::evaluate` polynomial branch | `crates/libsvm/src/kernel.rs:157` | Equivalent | |
| `Kernel::kernel_rbf` (inline) | `vendor/libsvm/svm.cpp:243` | `Kernel::evaluate` RBF branch | `crates/libsvm/src/kernel.rs:157` | Equivalent | Uses precomputed `x_square`. |
| `Kernel::kernel_sigmoid` (inline) | `vendor/libsvm/svm.cpp:247` | `Kernel::evaluate` sigmoid branch | `crates/libsvm/src/kernel.rs:157` | Equivalent | |
| `Kernel::kernel_precomputed` (inline) | `vendor/libsvm/svm.cpp:251` | `Kernel::evaluate` precomputed branch | `crates/libsvm/src/kernel.rs:172` | Equivalent | |
| `Kernel::Kernel` | `vendor/libsvm/svm.cpp:257` | `Kernel::new` | `crates/libsvm/src/kernel.rs:137` | Equivalent | Kernel dispatch and optional RBF cache setup aligned. |
| `Kernel::~Kernel` | `vendor/libsvm/svm.cpp:292` | Rust drop on `Kernel` fields | `crates/libsvm/src/kernel.rs:127` | Intentional divergence | Explicit delete not needed. |
| `Kernel::dot` | `vendor/libsvm/svm.cpp:298` | `dot` | `crates/libsvm/src/kernel.rs:36` | Equivalent | Sparse dot walk semantics preserved. |
| `Kernel::k_function` | `vendor/libsvm/svm.cpp:320` | `k_function` | `crates/libsvm/src/kernel.rs:94` | Equivalent | Single-evaluation path matches C behavior. |
| `Solver::get_C` (inline) | `vendor/libsvm/svm.cpp:430` | `get_c` | `crates/libsvm/src/solver.rs:224` | Equivalent | |
| `Solver::update_alpha_status` (inline) | `vendor/libsvm/svm.cpp:434` | `update_alpha_status` | `crates/libsvm/src/solver.rs:229` | Equivalent | |
| `Solver::is_upper_bound` (inline) | `vendor/libsvm/svm.cpp:442` | `is_upper_bound` | `crates/libsvm/src/solver.rs:240` | Equivalent | |
| `Solver::is_lower_bound` (inline) | `vendor/libsvm/svm.cpp:443` | `is_lower_bound` | `crates/libsvm/src/solver.rs:245` | Equivalent | |
| `Solver::is_free` (inline) | `vendor/libsvm/svm.cpp:444` | `is_free` | `crates/libsvm/src/solver.rs:250` | Equivalent | |
| `Solver::swap_index` | `vendor/libsvm/svm.cpp:454` | `swap_index` | `crates/libsvm/src/solver.rs:254` | Equivalent | |
| `Solver::reconstruct_gradient` | `vendor/libsvm/svm.cpp:466` | `reconstruct_gradient` | `crates/libsvm/src/solver.rs:266` | Equivalent | |
| `Solver::Solve` | `vendor/libsvm/svm.cpp:508` | `Solver::solve` | `crates/libsvm/src/solver.rs:77` | Equivalent (API shape) | Variant enum unifies standard and NU flows. |
| `Solver::select_working_set` | `vendor/libsvm/svm.cpp:790` | `select_working_set_standard` | `crates/libsvm/src/solver.rs:321` | Equivalent | |
| `Solver::be_shrunk` | `vendor/libsvm/svm.cpp:889` | `be_shrunk_standard` | `crates/libsvm/src/solver.rs:611` | Equivalent | |
| `Solver::do_shrinking` | `vendor/libsvm/svm.cpp:909` | `do_shrinking_standard` | `crates/libsvm/src/solver.rs:629` | Equivalent | |
| `Solver::calculate_rho` | `vendor/libsvm/svm.cpp:970` | `calculate_rho_standard` | `crates/libsvm/src/solver.rs:741` | Equivalent | |
| `Solver_NU::Solve` (inline) | `vendor/libsvm/svm.cpp:1017` | `Solver::solve` with `SolverVariant::Nu` | `crates/libsvm/src/solver.rs:77` | Equivalent (API shape) | NU state handled without subclass in Rust. |
| `Solver_NU::select_working_set` | `vendor/libsvm/svm.cpp:1033` | `select_working_set_nu` | `crates/libsvm/src/solver.rs:394` | Equivalent | |
| `Solver_NU::be_shrunk` | `vendor/libsvm/svm.cpp:1145` | `be_shrunk_nu` | `crates/libsvm/src/solver.rs:673` | Equivalent | |
| `Solver_NU::do_shrinking` | `vendor/libsvm/svm.cpp:1165` | `do_shrinking_nu` | `crates/libsvm/src/solver.rs:691` | Equivalent | |
| `Solver_NU::calculate_rho` | `vendor/libsvm/svm.cpp:1217` | `calculate_rho_nu` | `crates/libsvm/src/solver.rs:775` | Equivalent | |
| `SVC_Q::SVC_Q` | `vendor/libsvm/svm.cpp:1273` | `SvcQ::new` | `crates/libsvm/src/qmatrix.rs:43` | Equivalent | |
| `SVC_Q::get_Q` | `vendor/libsvm/svm.cpp:1283` | `SvcQ::get_q` | `crates/libsvm/src/qmatrix.rs:55` | Equivalent | |
| `SVC_Q::get_QD` | `vendor/libsvm/svm.cpp:1298` | `SvcQ::get_qd` | `crates/libsvm/src/qmatrix.rs:67` | Equivalent | |
| `SVC_Q::swap_index` | `vendor/libsvm/svm.cpp:1303` | `SvcQ::swap_index` | `crates/libsvm/src/qmatrix.rs:71` | Equivalent | |
| `SVC_Q::~SVC_Q` | `vendor/libsvm/svm.cpp:1311` | Rust drop on owned fields | `crates/libsvm/src/qmatrix.rs:31` | Intentional divergence | |
| `ONE_CLASS_Q::ONE_CLASS_Q` | `vendor/libsvm/svm.cpp:1326` | `OneClassQ::new` | `crates/libsvm/src/qmatrix.rs:91` | Equivalent | |
| `ONE_CLASS_Q::get_Q` | `vendor/libsvm/svm.cpp:1335` | `OneClassQ::get_q` | `crates/libsvm/src/qmatrix.rs:102` | Equivalent | |
| `ONE_CLASS_Q::get_QD` | `vendor/libsvm/svm.cpp:1347` | `OneClassQ::get_qd` | `crates/libsvm/src/qmatrix.rs:112` | Equivalent | |
| `ONE_CLASS_Q::swap_index` | `vendor/libsvm/svm.cpp:1352` | `OneClassQ::swap_index` | `crates/libsvm/src/qmatrix.rs:116` | Equivalent | |
| `ONE_CLASS_Q::~ONE_CLASS_Q` | `vendor/libsvm/svm.cpp:1359` | Rust drop on owned fields | `crates/libsvm/src/qmatrix.rs:79` | Intentional divergence | |
| `SVR_Q::SVR_Q` | `vendor/libsvm/svm.cpp:1372` | `SvrQ::new` | `crates/libsvm/src/qmatrix.rs:148` | Equivalent | |
| `SVR_Q::swap_index` | `vendor/libsvm/svm.cpp:1394` | `SvrQ::swap_index` | `crates/libsvm/src/qmatrix.rs:211` | Equivalent | |
| `SVR_Q::get_Q` | `vendor/libsvm/svm.cpp:1401` | `SvrQ::get_q` | `crates/libsvm/src/qmatrix.rs:184` | Equivalent | |
| `SVR_Q::get_QD` | `vendor/libsvm/svm.cpp:1423` | `SvrQ::get_qd` | `crates/libsvm/src/qmatrix.rs:207` | Equivalent | |
| `SVR_Q::~SVR_Q` | `vendor/libsvm/svm.cpp:1428` | Rust drop on owned fields | `crates/libsvm/src/qmatrix.rs:127` | Intentional divergence | |

## 4) Formulation, probability, and grouping internals (`svm.cpp`)

| Original function | C location | Rust equivalent | Rust location | Status | Notes |
|---|---|---|---|---|---|
| `solve_c_svc` | `vendor/libsvm/svm.cpp:1450` | `solve_c_svc` | `crates/libsvm/src/train.rs:18` | Equivalent | |
| `solve_nu_svc` | `vendor/libsvm/svm.cpp:1485` | `solve_nu_svc` | `crates/libsvm/src/train.rs:45` | Equivalent | |
| `solve_one_class` | `vendor/libsvm/svm.cpp:1540` | `solve_one_class` | `crates/libsvm/src/train.rs:88` | Equivalent | |
| `solve_epsilon_svr` | `vendor/libsvm/svm.cpp:1572` | `solve_epsilon_svr` | `crates/libsvm/src/train.rs:116` | Equivalent | |
| `solve_nu_svr` | `vendor/libsvm/svm.cpp:1610` | `solve_nu_svr` | `crates/libsvm/src/train.rs:148` | Equivalent | |
| `svm_train_one` | `vendor/libsvm/svm.cpp:1657` | `svm_train_one` | `crates/libsvm/src/train.rs:189` | Equivalent | |
| `sigmoid_train` | `vendor/libsvm/svm.cpp:1715` | `sigmoid_train` | `crates/libsvm/src/probability.rs:27` | Equivalent | |
| `sigmoid_predict` | `vendor/libsvm/svm.cpp:1828` | `sigmoid_predict` | `crates/libsvm/src/probability.rs:141` | Equivalent | |
| `multiclass_probability` | `vendor/libsvm/svm.cpp:1839` | `multiclass_probability` | `crates/libsvm/src/probability.rs:160` | Equivalent | |
| `svm_binary_svc_probability` | `vendor/libsvm/svm.cpp:1903` | `svm_binary_svc_probability` | `crates/libsvm/src/probability.rs:228` | Equivalent | |
| `predict_one_class_probability` | `vendor/libsvm/svm.cpp:1990` | `predict_one_class_probability` | `crates/libsvm/src/probability.rs:311` | Equivalent | |
| `compare_double` | `vendor/libsvm/svm.cpp:2011` | `sort_by(partial_cmp)` | `crates/libsvm/src/probability.rs:354` | Equivalent (API shape) | Rust sorting closure replaces C comparator callback. |
| `svm_one_class_probability` | `vendor/libsvm/svm.cpp:2021` | `svm_one_class_probability` | `crates/libsvm/src/probability.rs:340` | Equivalent | |
| `svm_svr_probability` | `vendor/libsvm/svm.cpp:2067` | `svm_svr_probability` | `crates/libsvm/src/probability.rs:404` | Equivalent | |
| `svm_group_classes` | `vendor/libsvm/svm.cpp:2101` | `svm_group_classes` | `crates/libsvm/src/train.rs:236` | Equivalent | |

## 5) CLI parity: `svm-train.c` ↔ `svm-train-rs`

| Original function | C location | Rust equivalent | Rust location | Status | Notes |
|---|---|---|---|---|---|
| `print_null` | `vendor/libsvm/svm-train.c:9` | `set_quiet(true)` path | `bins/svm-train-rs/src/main.rs:188`, `crates/libsvm/src/lib.rs:26` | Equivalent (API shape) | Quiet mode implemented without callback pointer. |
| `exit_with_help` | `vendor/libsvm/svm-train.c:11` | `exit_with_help` | `bins/svm-train-rs/src/main.rs:8` | Equivalent | |
| `exit_input_error` | `vendor/libsvm/svm-train.c:45` | parse errors via `load_problem` + `SvmError` | `crates/libsvm/src/io.rs:155` | Equivalent (API shape) | Structured parse errors replace direct line-exit helper. |
| `readline` | `vendor/libsvm/svm-train.c:65` | reader lines iteration | `crates/libsvm/src/io.rs:155` | Equivalent (API shape) | |
| `main` | `vendor/libsvm/svm-train.c:83` | `main` | `bins/svm-train-rs/src/main.rs:43` | Equivalent | |
| `do_cross_validation` | `vendor/libsvm/svm-train.c:122` | `do_cross_validation` | `bins/svm-train-rs/src/main.rs:209` | Equivalent | |
| `parse_command_line` | `vendor/libsvm/svm-train.c:161` | option parsing in `main` | `bins/svm-train-rs/src/main.rs:51` | Equivalent | |
| `read_problem` | `vendor/libsvm/svm-train.c:278` | `load_problem` | `crates/libsvm/src/io.rs:148` | Equivalent (API shape) | |

## 6) CLI parity: `svm-predict.c` ↔ `svm-predict-rs`

| Original function | C location | Rust equivalent | Rust location | Status | Notes |
|---|---|---|---|---|---|
| `print_null` | `vendor/libsvm/svm-predict.c:8` | `set_quiet(true)` path | `bins/svm-predict-rs/src/main.rs:76`, `crates/libsvm/src/lib.rs:26` | Equivalent (API shape) | |
| `readline` | `vendor/libsvm/svm-predict.c:21` | `load_problem` line parsing | `crates/libsvm/src/io.rs:155` | Equivalent (API shape) | |
| `exit_input_error` | `vendor/libsvm/svm-predict.c:39` | parse errors via `load_problem` | `crates/libsvm/src/io.rs:155` | Equivalent (API shape) | |
| `predict` | `vendor/libsvm/svm-predict.c:45` | prediction loop in `main` | `bins/svm-predict-rs/src/main.rs:134` | Equivalent | |
| `exit_with_help` | `vendor/libsvm/svm-predict.c:165` | `exit_with_help` | `bins/svm-predict-rs/src/main.rs:9` | Equivalent | |
| `main` | `vendor/libsvm/svm-predict.c:176` | `main` | `bins/svm-predict-rs/src/main.rs:31` | Equivalent | |

## 7) CLI parity: `svm-scale.c` ↔ `svm-scale-rs`

| Original function | C location | Rust equivalent | Rust location | Status | Notes |
|---|---|---|---|---|---|
| `exit_with_help` | `vendor/libsvm/svm-scale.c:7` | `exit_with_help` | `bins/svm-scale-rs/src/main.rs:6` | Equivalent | |
| `main` | `vendor/libsvm/svm-scale.c:42` | `main` | `bins/svm-scale-rs/src/main.rs:21` | Equivalent | |
| `readline` | `vendor/libsvm/svm-scale.c:335` | `BufRead::lines()` loops in pass1/2/3 | `bins/svm-scale-rs/src/main.rs:121`, `bins/svm-scale-rs/src/main.rs:158`, `bins/svm-scale-rs/src/main.rs:337` | Equivalent (API shape) | |
| `output_target` | `vendor/libsvm/svm-scale.c:358` | y-scaling print block in pass3 | `bins/svm-scale-rs/src/main.rs:341` | Equivalent | |
| `output` | `vendor/libsvm/svm-scale.c:372` | `output_feature` | `bins/svm-scale-rs/src/main.rs:393` | Equivalent | |
| `clean_up` | `vendor/libsvm/svm-scale.c:394` | Rust automatic resource cleanup + early exits | `bins/svm-scale-rs/src/main.rs:21` | Intentional divergence | Rust RAII replaces explicit cleanup helper. |

## 8) Parity Summary

- `Equivalent`: 80
- `Equivalent (API shape)`: 21
- `Intentional divergence`: 10
- `Missing`: 0

Notes:
- There are no unresolved algorithmic gaps in core training/prediction/probability paths.
- Remaining differences are primarily RAII memory management and print-callback model differences.
