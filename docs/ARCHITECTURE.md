# Architecture

This page describes how `libsvm-rs` is built: the source layout, each module,
the design decisions, and the notes on numerical equivalence with C LIBSVM.
For installation and usage, see the [README](../README.md).

## Source layout

```
crates/libsvm/src/
  lib.rs              — Module declarations and re-exports
  types.rs            — SvmNode, SvmProblem, SvmParameter, SvmModel, enums
  error.rs            — SvmError enum (thiserror)
  io.rs               — Problem/model file I/O (LIBSVM text format)
  kernel.rs           — Kernel evaluation (dot, powi, k_function, Kernel struct)
  cache.rs            — LRU kernel cache (Qfloat = f32)
  qmatrix.rs          — QMatrix trait + SvcQ, OneClassQ, SvrQ implementations
  solver.rs           — SMO solver (Standard + Nu variants, WSS3, shrinking)
  train.rs            — svm_train, solve dispatchers, class grouping
  predict.rs          — predict, predict_values, predict_probability
  probability.rs      — Platt scaling, multiclass probability, CV-based estimation
  cross_validation.rs — svm_cross_validation (stratified + simple)
  metrics.rs          — accuracy_percentage, regression_metrics (MSE + R²)
  builder.rs          — SvmParameterBuilder (validates parameters at build time)
  util.rs             — Shared helpers (group_classes, parse_feature_index, PRNG)

bins/
  svm-train-rs/       — CLI matching C svm-train (all flags including -wi class weights)
  svm-predict-rs/     — CLI matching C svm-predict (-b probability, -q quiet)
  svm-scale-rs/       — CLI matching C svm-scale (3-pass, save/restore params)
```

### Workspace Layout

The project uses a Cargo workspace:
- `crates/libsvm/` — the library crate (`libsvm-rs` on crates.io)
- `bins/svm-train-rs/`, `bins/svm-predict-rs/`, `bins/svm-scale-rs/` — CLI binaries
- `vendor/libsvm/` — upstream C source (for reference and vendored tests)
- `data/` — test datasets (heart_scale, iris.scale, housing_scale, generated synthetics)
- `reference/` — verification artifacts (differential results, provenance, benchmarks)
- `scripts/` — verification and testing scripts

## Module Details

### types.rs — Core Data Types

Types matching LIBSVM's `svm.h`:

| Type | Description | C++ equivalent |
|------|-------------|----------------|
| `SvmType` | Enum: CSvc, NuSvc, OneClass, EpsilonSvr, NuSvr | `svm_type` int constants |
| `KernelType` | Enum: Linear, Polynomial, Rbf, Sigmoid, Precomputed | `kernel_type` int constants |
| `SvmNode` | Sparse feature `{index: i32, value: f64}` | `struct svm_node` |
| `SvmProblem` | Training data: `labels: Vec<f64>`, `instances: Vec<Vec<SvmNode>>` | `struct svm_problem` |
| `SvmParameter` | All training parameters (defaults match LIBSVM) | `struct svm_parameter` |
| `SvmModel` | Trained model with SVs, coefficients, rho | `struct svm_model` |

**Key design choice**: `SvmNode.index` is `i32` (not `usize`) to match LIBSVM's C `int` and preserve file format compatibility. Sentinel nodes (`index == -1`) used in C to terminate instance arrays are not stored — Rust's `Vec::len()` tracks instance length instead.

`SvmParameter::default()` produces the same defaults as LIBSVM: `svm_type = CSvc`, `kernel_type = Rbf`, `degree = 3`, `gamma = 0` (auto-detected as 1/num_features), `coef0 = 0`, `c = 1`, `nu = 0.5`, `eps = 0.001`, `cache_size = 100 MB`, `shrinking = true`.

### kernel.rs — Kernel Evaluation

Two evaluation paths exist because the training kernel needs swappable data references for shrinking, while prediction only needs stateless evaluation:

1. **`k_function(x, y, param)`** — Standalone stateless evaluation for prediction. Operates on sparse `&[SvmNode]` slices. Handles all 5 kernel types including precomputed (looks up `x[0].value` as the precomputed kernel value).

2. **`Kernel` struct** — Training evaluator. Stores `Vec<&'a [SvmNode]>` (borrowed slices into the original problem data) so the solver can swap data point references during shrinking without copying data. The C++ achieves the same via `const_cast` on a `const svm_node **x` pointer array.

**Supported kernels**:

| Type | Formula |
|------|---------|
| Linear | `dot(x, y)` |
| Polynomial | `(γ·dot(x,y) + coef0)^degree` |
| RBF | `exp(-γ·‖x-y‖²)` |
| Sigmoid | `tanh(γ·dot(x,y) + coef0)` |
| Precomputed | `x[j].value` (user supplies full kernel matrix) |

**RBF optimization**: `x_square[i] = dot(x[i], x[i])` is precomputed once during `Kernel::new()`, so the RBF kernel becomes:
```
K(i,j) = exp(-γ * (x_sq[i] + x_sq[j] - 2·dot(x[i], x[j])))
```
This avoids recomputing ‖x_i‖² on every kernel evaluation — a significant win since the kernel is evaluated millions of times during training.

**Sparse dot product**: The `dot()` function walks two sparse vectors simultaneously using a merge-join pattern on sorted indices, skipping non-overlapping features in O(nnz_x + nnz_y) time.

### cache.rs — LRU Kernel Cache

LRU cache storing rows of the Q matrix as `Qfloat` (`f32`) values. Using f32 instead of f64 halves memory usage while providing sufficient precision for the cached kernel values (the solver accumulates gradients in f64).

**Data structure**: Index-based circular doubly-linked list (no `unsafe`). Each training instance gets a cache node. Node `l` (where l = number of instances) is the sentinel head of the LRU list. Nodes not currently in the LRU list have `prev == NONE` (`usize::MAX`).

**Cache operations**:

- **`get_data(index, len)`**: Returns `(&mut [Qfloat], start)` where `start` is how many elements were already cached. The caller (QMatrix) fills `data[start..len]` with fresh kernel evaluations. If the row isn't cached, allocates space (evicting LRU rows if needed) and returns `start = 0`.

- **`swap_index(i, j)`**: Critical for solver shrinking. Three steps:
  1. Swap row data and LRU positions for rows i and j
  2. Iterate **all** cached rows and swap column entries at positions i,j
  3. If a row covers the lower index but not the higher, evict it ("give up")

  The column-swap loop in step 2 is essential for correctness — without it, the solver reads stale kernel values after shrinking swaps indices. This was a subtle bug during development that caused silent numerical drift.

### qmatrix.rs — Q Matrix Implementations

The `QMatrix` trait bridges the Kernel and Solver, abstracting how kernel values are computed and cached for different SVM formulations:

```rust
pub trait QMatrix {
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat];
    fn get_qd(&self) -> &[f64];
    fn swap_index(&mut self, i: usize, j: usize);
}
```

**Why `&mut self` on `get_q`**: The C++ uses `const` methods with `mutable` cache fields. Rust doesn't allow this pattern with `&self` — the cache needs mutation. The Solver therefore owns `Box<dyn QMatrix>` and copies Q row data into its own buffers before the borrow ends, avoiding lifetime conflicts.

#### SvcQ — Classification Q Matrix

For C-SVC and nu-SVC:
- `Q[i][j] = y[i] * y[j] * K(x[i], x[j])` stored as `Qfloat` (f32)
- `QD[i] = K(x[i], x[i])` — always 1.0 for RBF kernel
- `swap_index` swaps cache rows, kernel data references, label signs, and diagonal entries

#### OneClassQ — One-Class Q Matrix

For one-class SVM:
- `Q[i][j] = K(x[i], x[j])` — no label scaling (all samples are treated as positive)
- Same cache and swap logic as SvcQ, minus the label array

#### SvrQ — Regression Q Matrix

For epsilon-SVR and nu-SVR. The dual formulation has **2l variables** (l = number of training samples):
- Variables `0..l` correspond to `α_i` with `sign[k] = +1`, `index[k] = k`
- Variables `l..2l` correspond to `α*_i` with `sign[k+l] = -1`, `index[k+l] = k`

The kernel cache stores only l rows of actual kernel values (since `K(x[i], x[j])` doesn't depend on whether we're looking at α or α*). `get_q` fetches the real kernel row, then reorders and applies signs:
```
buf[j] = sign[i] * sign[j] * data[index[j]]
```

**Double buffer**: Two buffers are needed because the solver requests Q_i and Q_j simultaneously during the alpha update step. If both wrote to the same buffer, the second `get_q` call would overwrite the first. `swap_index` only swaps `sign`, `index`, and `qd` — the kernel cache maps to real data indices via `index[]`.

### solver.rs — SMO Solver

The computational core of the library, translating `Solver` and `Solver_NU` from svm.cpp (lines 362–1265, ~900 lines of C++).

**Solver variant**: `SolverVariant::Standard` vs `SolverVariant::Nu`. 95% of the code is shared — only 4 methods differ:

| Method | Standard | Nu |
|--------|----------|-----|
| `select_working_set` | Single I_up/I_low partition | Separate pos/neg class selection |
| `calculate_rho` | Average over free variables | `(r1 - r2) / 2` from class-separated free vars |
| `do_shrinking` | Single Gmax bound | Separate pos/neg Gmax bounds |
| `be_shrunk` | Check against global bound | Check against class-specific bound |

**Key constants**:

| Constant | Value | Purpose |
|----------|-------|---------|
| `TAU` | 1e-12 | Floor for `quad_coef` to prevent division by zero in WSS |
| `INF` | `f64::INFINITY` | Initial gradient bounds |
| `EPS` | From parameter | KKT violation tolerance (default 0.001) |

#### Initialization

1. Copy input arrays (`alpha`, `y`, `p`) into owned vectors
2. Compute `alpha_status` for each variable: `LowerBound` (α=0), `UpperBound` (α=C), or `Free`
3. Initialize gradient: `G[i] = p[i] + Σ_j(alpha[j] * Q[i][j])` for all non-lower-bound j
4. Initialize `G_bar`: `G_bar[i] = Σ_j(C_j * Q[i][j])` for all upper-bound j

`G_bar` accelerates gradient reconstruction during unshrinking — when a variable transitions from upper-bound to free, `G_bar` provides the full gradient contribution without re-scanning all variables.

#### Main Loop

```
while iter < max_iter:
    if counter expires: do_shrinking()
    (i, j) = select_working_set()       // returns None if optimal
    if None:
        reconstruct_gradient()           // unshrink all variables
        active_size = l                  // restore full problem
        (i, j) = select_working_set()    // retry with all variables
        if None: break                   // truly optimal
    update_alpha_pair(i, j)              // analytic 2-variable solve
```

- `max_iter`: `max(10_000_000, 100*l)` — same as C++
- Shrink counter: `min(l, 1000)` iterations between shrinking passes

#### Working Set Selection (WSS3)

Implements the WSS3 heuristic from Fan, Chen, and Lin (2005):

**Standard variant** — two-pass selection:
1. Find i: maximizes `-y_i * G[i]` among I_up (variables that can increase)
2. Find j: among I_low (variables that can decrease), minimizes the second-order approximation of objective decrease: `-(grad_diff²) / quad_coef` where `quad_coef = QD[i] + QD[j] - 2·y_i·y_j·Q_ij`

**Nu variant**: Partitions variables into positive and negative classes. Finds `i_p`/`j_p` (best positive pair) and `i_n`/`j_n` (best negative pair), then selects the pair with larger violation.

Returns `None` when the maximum KKT violation `Gmax_i + Gmax_j < eps`, indicating optimality.

#### Alpha Update

Analytic solution of the 2-variable sub-problem with box constraints [0, C]:

- **Different labels** (`y[i] ≠ y[j]`): constraint line `alpha[i] - alpha[j] = constant`
- **Same labels** (`y[i] = y[j]`): constraint line `alpha[i] + alpha[j] = constant`

Both branches clip alpha to the feasible box and update gradients for all active variables:
```
G[k] += Q_i[k] * Δα_i + Q_j[k] * Δα_j    for all k in active set
```

`G_bar` is updated incrementally when `alpha_status` changes (transition to/from upper bound).

#### Shrinking

Variables firmly at their bounds that are unlikely to change in subsequent iterations get "shrunk" — swapped to the end of the active set so they're excluded from WSS and gradient updates:

- `be_shrunk(i)` checks if `-y_i * G[i]` exceeds the current violation bound by a sufficient margin
- Active-from-back swap finds the first non-shrinkable variable from the end

**Unshrink trigger**: When `Gmax1 + Gmax2 <= eps * 10` (approaching convergence), the solver reconstructs the full gradient for all variables and resets `active_size = l`. This ensures the final optimality check considers all variables.

**Gradient reconstruction**: Uses `G_bar` to efficiently compute the full gradient contribution from upper-bound variables without re-evaluating kernel values.

### train.rs — Training Pipeline

Orchestrates the solver for all SVM types.

#### Solve Dispatchers

| Function | SVM Type | Solver Variant | Key Setup |
|----------|----------|----------------|-----------|
| `solve_c_svc` | C-SVC | Standard | `p[i] = -1` for all i; after solving, `alpha[i] *= y[i]` to get signed coefficients |
| `solve_nu_svc` | nu-SVC | Nu | Alpha initialized proportionally from nu budget; after solving, rescale by `1/r` |
| `solve_one_class` | One-class | Standard | `alpha = [1,1,...,frac,0,...,0]` where `nu*l` alphas are set to 1.0 |
| `solve_epsilon_svr` | epsilon-SVR | Standard | 2l variables; `p[i] = ε - y_i`, `p[i+l] = ε + y_i` |
| `solve_nu_svr` | nu-SVR | Nu | 2l variables; alpha initialized uniformly at `C·nu·l / 2` |

Each dispatcher sets up the dual problem (initial alpha values, gradient vector `p`, bounds `C_i`), calls `Solver::solve()`, then extracts the solution (nonzero alphas become support vector coefficients, rho becomes the bias).

#### Class Grouping

`svm_group_classes` builds:
- `label[]`: unique class labels in first-occurrence order (with -1/+1 swap to match C++ convention)
- `count[]`: number of samples per class
- `start[]`: cumulative start index for each class in the permuted array
- `perm[]`: permutation mapping grouped indices back to original dataset indices

This grouping enables efficient one-vs-one pair extraction without copying data.

#### svm_train — Main Entry Point

Two branches:

1. **Regression / one-class**: Single call to `svm_train_one()`, extract nonzero alphas as support vectors.

2. **Classification** (one-vs-one):
   - Group classes via `svm_group_classes`
   - Compute weighted C per class (base C × class weight)
   - For each of k*(k-1)/2 class pairs (i,j):
     - Build sub-problem with only classes i and j
     - Train binary classifier
     - Mark nonzero support vectors
   - Assemble `sv_coef` matrix: `sv_coef[j-1][nz_start[i]..] = class_i coefficients`

**Gamma auto-detection**: If `gamma == 0` (default), sets `gamma = 1/max_feature_index` where `max_feature_index` is the highest feature index seen in the training data.

**Probability training**: When `param.probability == true`, training invokes cross-validation internally:
- Binary classification: 5-fold CV to collect decision values, then fits a sigmoid (Platt scaling) via `sigmoid_train`
- Multiclass: pairwise probability estimation via `svm_binary_svc_probability`
- Regression: 5-fold CV to estimate sigma for Laplace correction
- One-class: 5-fold CV to estimate the positive density fraction

### predict.rs — Prediction

#### predict_values

Core prediction function for all SVM types:

- **Classification** (C-SVC, nu-SVC): Computes k*(k-1)/2 pairwise decision values `dec_value[p] = Σ sv_coef · K(x, sv) - rho[p]`, then runs one-vs-one voting. Returns the class with the most votes (ties broken by label ordering).

- **Regression** (epsilon-SVR, nu-SVR): `prediction = Σ sv_coef[i] * K(x, sv[i]) - rho[0]`

- **One-class**: Returns `+1` if `Σ sv_coef[i] * K(x, sv[i]) - rho[0] > 0`, else `-1`.

#### predict_probability

Wraps `predict_values` with probability calibration:

- **Binary classification**: Applies Platt sigmoid `P(y=1|f) = 1 / (1 + exp(A·f + B))` using probA/probB from the model.
- **Multiclass**: Computes pairwise probabilities via Platt sigmoid, then solves the coupled probability system via `multiclass_probability()` (iterative optimization matching LIBSVM's algorithm).
- **Regression**: Returns Laplace-corrected probability `P = 1 / (2·sigma) · exp(-|y-f|/sigma)`.
- **One-class**: Returns a density-based probability mark.

### probability.rs — Probability Estimation

#### sigmoid_train / sigmoid_predict

Fits a sigmoid `P(y=1|f) = 1 / (1 + exp(A·f + B))` to decision values using the algorithm from Platt (2000) with improvements from Lin, Lin, and Weng (2007). Uses Newton's method with backtracking line search to minimize the negative log-likelihood.

#### multiclass_probability

Given k*(k-1)/2 pairwise probabilities `r[i][j]`, solves for class probabilities `p[1]..p[k]` that satisfy:
```
p[i] = Σ_{j≠i} r[i][j] · p[j]  /  (r[i][j] · p[j] + r[j][i] · p[i])
```
Uses an iterative method (100 max iterations, convergence at 1e-5) matching LIBSVM's `multiclass_probability()`.

#### svm_binary_svc_probability

Runs internal 5-fold cross-validation, collects decision values for all training instances, then calls `sigmoid_train` to fit A and B parameters. Uses a deterministic PRNG seeded from LIBSVM's C `rand()` equivalent for fold shuffling.

### cross_validation.rs — Cross-Validation

`svm_cross_validation(problem, param, nr_fold)` splits data, trains on k-1 folds, predicts on the held-out fold:

- **Classification**: Stratified folding — each fold has approximately the same class proportions as the full dataset. Uses the same deterministic shuffling as C LIBSVM.
- **Regression / one-class**: Simple sequential splitting.

Returns a `Vec<f64>` of predicted values aligned with the original problem ordering, so the caller can compute accuracy (classification) or MSE (regression).

### io.rs — File I/O

#### Problem Files

LIBSVM sparse format: `label index:value index:value ...` per line. Features must have ascending indices. Missing features are implicitly zero (sparse representation).

For precomputed kernels: `label 0:sample_id 1:K(x,x1) 2:K(x,x2) ...` where index 0 holds the 1-based sample identifier.

#### Model Files

Two sections:
1. **Header**: Key-value pairs (`svm_type`, `kernel_type`, `nr_class`, `total_sv`, `rho`, `label`, `nr_sv`, optionally `probA`, `probB`)
2. **SV section**: After the `SV` marker, one line per support vector: `coef1 [coef2 ...] index:value index:value ...`

Float formatting uses `%.17g`-equivalent precision for model files (ensuring round-trip fidelity) and `%g`-equivalent for problem files. The `Gfmt` struct replicates C's `%g` format: dynamically chooses fixed vs scientific notation, strips trailing zeros, handles edge cases like `-0` formatting.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `SvmNode.index` is `i32` | Matches C `int` for format compatibility; sentinel `-1` not stored |
| `Qfloat = f32` | Matches original, halves cache memory; gradients accumulated in f64 |
| Ownership replaces manual memory | No `free_sv` flag; `SvmModel` owns its support vectors |
| `Kernel.x` is `Vec<&[SvmNode]>` | Swappable references during shrinking without data copies |
| Solver uses enum variant, not traits | 95% shared code; cleaner than trait objects for 4 method overrides |
| `QMatrix::get_q` takes `&mut self` | Replaces C++ `const` method + `mutable` cache fields |
| `%.17g` float formatting | Bit-exact model file compatibility with C LIBSVM |
| Minimal runtime deps | Default builds use only `thiserror`; `serde` and `rayon` are feature-gated |
| Manual CLI arg parsing | No `clap` dependency; handles `-wi` composite flags matching C behavior |
| Deterministic PRNG | Replicates C's `rand()` for identical fold shuffling in CV/probability |

## Numerical Equivalence Notes

- **Gradient accumulation**: Uses f64 for gradients (matching C++), f32 for cached Q values
- **Quad_coef floor**: `TAU = 1e-12` prevents division by zero in working set selection
- **Alpha comparison**: `alpha[i].abs() > 0.0` for SV detection (exact zero check, matching C++)
- **Class label cast**: `labels[i] as i32` matches C's `(int)prob->y[i]` truncation behavior
- **Float formatting**: `Gfmt` struct produces identical output to C's `printf("%g", x)` for all tested values
- **Target**: numerical equivalence, not bitwise identity. The differential suite accepts non-probability outputs within relative `1.5e-5` and absolute `1e-8` of upstream; [`reference/tolerance_policy.md`](../reference/tolerance_policy.md) lists every tolerance and warning rule
- **Known sources of drift**: floating-point accumulation order differences between compilers, shrinking heuristic timing, and gradient reconstruction precision
