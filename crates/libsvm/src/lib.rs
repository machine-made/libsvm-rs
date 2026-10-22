//! # libsvm-rs
//!
//! `libsvm-rs` is a pure Rust reimplementation of
//! [LIBSVM](https://github.com/cjlin1/libsvm) for training, prediction,
//! cross-validation, and LIBSVM text model/problem I/O without an FFI boundary.
//! The public API mirrors the original LIBSVM concepts while using owned Rust
//! data and `Result`-based error handling; see the
//! [migration guide](https://github.com/ricardofrantz/libsvm-rs/blob/main/docs/MIGRATION.md)
//! for a C-to-Rust mapping.
//!
//! ## LIBSVM parity
//!
//! The crate targets numerical equivalence and model-file compatibility with
//! upstream LIBSVM rather than bit-for-bit identity. Core counterparts include
//! [`train::svm_train`] for `svm_train`,
//! [`cross_validation::svm_cross_validation`] for `svm_cross_validation`,
//! [`predict::predict_values`] / [`predict::predict`] /
//! [`predict::predict_probability`] for the prediction APIs, and
//! [`io::save_model`] / [`io::load_model`] for LIBSVM model persistence.
//!
//! ## Quick example
//!
//! ```
//! use libsvm_rs::predict::predict;
//! use libsvm_rs::train::svm_train;
//! use libsvm_rs::{KernelType, SvmNode, SvmParameterBuilder, SvmProblem, SvmType};
//!
//! let problem = SvmProblem {
//!     labels: vec![-1.0, -1.0, 1.0, 1.0],
//!     instances: vec![
//!         vec![SvmNode { index: 1, value: -2.0 }],
//!         vec![SvmNode { index: 1, value: -1.0 }],
//!         vec![SvmNode { index: 1, value: 1.0 }],
//!         vec![SvmNode { index: 1, value: 2.0 }],
//!     ],
//! };
//! let param = SvmParameterBuilder::new()
//!     .svm_type(SvmType::CSvc)
//!     .kernel_type(KernelType::Linear)
//!     .build()?;
//!
//! let model = svm_train(&problem, &param);
//! let label = predict(&model, &[SvmNode { index: 1, value: 1.5 }]);
//! assert_eq!(label, 1.0);
//! # Ok::<(), libsvm_rs::SvmError>(())
//! ```
//!
//! ## Status
//!
//! Training works for all 5 SVM types (C-SVC, ν-SVC, one-class, ε-SVR,
//! ν-SVR). See [`train::svm_train`] for training, [`predict::predict`]
//! for inference, and [`predict::predict_probability`] for probabilistic
//! outputs. The project repository is
//! <https://github.com/ricardofrantz/libsvm-rs>.
//!
//! ## Trust Boundary
//!
//! Problem and model files are treated as untrusted text input by default.
//! The [`io`] loaders apply [`LoadOptions`] caps, reject malformed sparse
//! feature rows, and validate model-header consistency before allocating
//! support-vector storage. These checks bound parsing work and memory use; they
//! do not authenticate a model or prove that it is appropriate for a particular
//! deployment.
//!
//! ## Feature Flags
//!
//! - `rayon` — Enable parallel cross-validation (off by default), including
//!   probability-calibration CV folds for binary SVC models. Fold assignment
//!   remains serial and deterministic, then each fold trains on an
//!   independent worker. Per-fold training diagnostics are suppressed while the
//!   parallel workers run so output cannot interleave; use the default serial
//!   path if you need fold-internal progress text. With `k` parallel folds, peak
//!   memory can include up to `min(k, rayon_threads)` simultaneous kernel caches
//!   of `SvmParameter::cache_size` each; the cache size is never divided
//!   implicitly.
//! - `serde` — Enable `Serialize`/`Deserialize` for model and parameter
//!   types. `SvmType` and `KernelType` serialize as pinned LIBSVM integer
//!   codes (`0..4`). Deserializing `SvmModel` runs the same structural
//!   validation as the text model loader; LIBSVM text model files remain the
//!   C-compatible interchange format.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static QUIET_MODE: AtomicBool = AtomicBool::new(false);
static SUPPRESS_INFO_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Enable or disable quiet mode. When quiet, solver diagnostic messages
/// are suppressed (equivalent to LIBSVM's `-q` flag).
pub fn set_quiet(quiet: bool) {
    QUIET_MODE.store(quiet, Ordering::Relaxed);
}

/// Print an info message to stderr (suppressed in quiet mode).
pub(crate) fn info(msg: &str) {
    if !QUIET_MODE.load(Ordering::Relaxed) && SUPPRESS_INFO_DEPTH.load(Ordering::Relaxed) == 0 {
        eprint!("{}", msg);
    }
}

#[cfg(feature = "rayon")]
pub(crate) fn with_suppressed_info<T>(f: impl FnOnce() -> T) -> T {
    struct SuppressInfoGuard;

    impl Drop for SuppressInfoGuard {
        fn drop(&mut self) {
            SUPPRESS_INFO_DEPTH.fetch_sub(1, Ordering::Relaxed);
        }
    }

    SUPPRESS_INFO_DEPTH.fetch_add(1, Ordering::Relaxed);
    let _guard = SuppressInfoGuard;
    f()
}

/// Fluent builder for [`SvmParameter`](crate::SvmParameter) values.
pub mod builder;
/// Kernel-row cache used by the SMO solver.
pub mod cache;
/// Error types returned by fallible parsing, validation, and I/O APIs.
pub mod error;
/// LIBSVM problem/model text-format loading and saving.
pub mod io;
/// Kernel functions equivalent to LIBSVM's linear, polynomial, RBF, sigmoid, and precomputed kernels.
pub mod kernel;
/// Helpers for classification accuracy and regression error summaries.
pub mod metrics;
/// Q-matrix implementations consumed by the SMO solver.
pub mod qmatrix;
/// Sequential Minimal Optimization solver internals corresponding to LIBSVM's `Solver` and `Solver_NU`.
pub mod solver;
/// Training entry points corresponding to LIBSVM's `svm_train`.
pub mod train;
/// LIBSVM-compatible public data structures and C-style helper functions.
pub mod types;
/// Utility routines shared by training, parsing, and parity-oriented algorithms.
pub mod util;

/// Cross-validation entry point corresponding to LIBSVM's `svm_cross_validation`.
pub mod cross_validation;
/// Prediction entry points corresponding to LIBSVM's `svm_predict*` APIs.
pub mod predict;
/// Probability-estimation routines corresponding to LIBSVM's Platt-scaling and related helpers.
pub mod probability;

/// Fluent parameter builder with LIBSVM-compatible defaults.
pub use builder::SvmParameterBuilder;
/// Structured errors returned instead of LIBSVM integer/null failure codes.
pub use error::SvmError;
/// Resource caps for untrusted LIBSVM text input.
pub use io::LoadOptions;
/// Accuracy and regression metric helpers.
pub use metrics::{accuracy_percentage, regression_metrics};
/// Public LIBSVM-compatible types and helper functions.
pub use types::*;
