//! Core LIBSVM-compatible data structures.
//!
//! These types intentionally mirror LIBSVM concepts: sparse nodes, problem
//! rows, solver parameters, and trained model fields. Values produced by
//! [`crate::io::load_problem`] and [`crate::io::load_model`] have passed the
//! loader's text-format and resource-bound checks. Values constructed manually
//! by a caller are not automatically checked; call [`check_parameter`] before
//! training, and prefer the loader APIs for external model/problem files.

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "Tests unwrap fixture loading and checked assertions"
    )
)]

/// Type of SVM formulation.
///
/// Matches the integer constants in the original LIBSVM (`svm.h`):
/// `C_SVC=0, NU_SVC=1, ONE_CLASS=2, EPSILON_SVR=3, NU_SVR=4`.
///
/// # Example
///
/// ```
/// use libsvm_rs::SvmType;
/// let formulation = SvmType::EpsilonSvr;
/// assert_eq!(formulation as i32, 3);
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SvmType {
    /// C-Support Vector Classification.
    CSvc = 0,
    /// ν-Support Vector Classification.
    NuSvc = 1,
    /// One-class SVM (distribution estimation / novelty detection).
    OneClass = 2,
    /// ε-Support Vector Regression.
    EpsilonSvr = 3,
    /// ν-Support Vector Regression.
    NuSvr = 4,
}

/// Type of kernel function.
///
/// Matches the integer constants in the original LIBSVM (`svm.h`):
/// `LINEAR=0, POLY=1, RBF=2, SIGMOID=3, PRECOMPUTED=4`.
///
/// # Example
///
/// ```
/// use libsvm_rs::KernelType;
/// let kernel = KernelType::Rbf;
/// assert_eq!(kernel as i32, 2);
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum KernelType {
    /// `K(x,y) = x·y`
    Linear = 0,
    /// `K(x,y) = (γ·x·y + coef0)^degree`
    Polynomial = 1,
    /// `K(x,y) = exp(-γ·‖x-y‖²)`
    Rbf = 2,
    /// `K(x,y) = tanh(γ·x·y + coef0)`
    Sigmoid = 3,
    /// Kernel values supplied as a precomputed matrix.
    Precomputed = 4,
}

/// A single sparse feature: `index:value`.
///
/// In the original LIBSVM, a sentinel node with `index = -1` marks the end
/// of each instance. In this Rust port, instance length is tracked by
/// `Vec::len()` instead, so no sentinel is needed.
///
/// # Example
///
/// ```
/// use libsvm_rs::SvmNode;
/// let node = SvmNode { index: 1, value: 0.5 };
/// assert_eq!(node.index, 1);
/// assert_eq!(node.value.to_bits(), 0.5_f64.to_bits());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvmNode {
    /// 1-based feature index. Uses `i32` to match the original C `int` and
    /// preserve file-format compatibility.
    pub index: i32,
    /// Feature value.
    pub value: f64,
}

/// A training/test problem: a collection of labelled sparse instances.
///
/// `load_problem` validates that sparse feature indices are ascending and
/// within the configured [`crate::io::LoadOptions`] bounds. When constructing a
/// problem manually, keep `labels.len() == instances.len()` and use ascending
/// feature indices to match LIBSVM input assumptions.
///
/// # Example
///
/// ```
/// use libsvm_rs::{SvmNode, SvmProblem};
/// let problem = SvmProblem {
///     labels: vec![-1.0, 1.0],
///     instances: vec![vec![SvmNode { index: 1, value: -1.0 }],
///                     vec![SvmNode { index: 1, value: 1.0 }]],
/// };
/// assert_eq!(problem.labels.len(), problem.instances.len());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SvmProblem {
    /// Label (class for classification, target for regression) per instance.
    pub labels: Vec<f64>,
    /// Sparse feature vectors, one per instance.
    pub instances: Vec<Vec<SvmNode>>,
}

/// SVM parameters controlling the formulation, kernel, and solver.
///
/// Default values match the original LIBSVM defaults.
///
/// # Example
///
/// ```
/// use libsvm_rs::{KernelType, SvmParameter};
/// let param = SvmParameter { kernel_type: KernelType::Linear, c: 2.0, ..Default::default() };
/// param.validate()?;
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SvmParameter {
    /// SVM formulation type.
    pub svm_type: SvmType,
    /// Kernel function type.
    pub kernel_type: KernelType,
    /// Degree for polynomial kernel.
    pub degree: i32,
    /// γ parameter for RBF, polynomial, and sigmoid kernels.
    /// Set to `1/num_features` when 0.
    pub gamma: f64,
    /// Independent term in polynomial and sigmoid kernels.
    pub coef0: f64,
    /// Cache memory size in MB.
    pub cache_size: f64,
    /// Stopping tolerance for the solver.
    pub eps: f64,
    /// Cost parameter C (for C-SVC, ε-SVR, ν-SVR).
    pub c: f64,
    /// Per-class weight overrides: `(class_label, weight)` pairs.
    pub weight: Vec<(i32, f64)>,
    /// ν parameter (for ν-SVC, one-class SVM, ν-SVR).
    pub nu: f64,
    /// ε in the ε-insensitive loss function (ε-SVR).
    pub p: f64,
    /// Whether to use the shrinking heuristic.
    pub shrinking: bool,
    /// Whether to train for probability estimates.
    pub probability: bool,
}

impl Default for SvmParameter {
    fn default() -> Self {
        Self {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            degree: 3,
            gamma: 0.0, // means 1/num_features
            coef0: 0.0,
            cache_size: 100.0,
            eps: 0.001,
            c: 1.0,
            weight: Vec::new(),
            nu: 0.5,
            p: 0.1,
            shrinking: true,
            probability: false,
        }
    }
}

impl SvmParameter {
    /// Validate parameter values (independent of training data).
    ///
    /// This checks the same constraints as the original LIBSVM's
    /// `svm_check_parameter`, except for the ν-SVC feasibility check
    /// which requires the problem. Use [`check_parameter`] for the full check.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::SvmError::InvalidParameter`] when a parameter
    /// is outside the accepted range.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameter;
    /// SvmParameter::default().validate()?;
    /// let invalid = SvmParameter { eps: 0.0, ..Default::default() };
    /// assert!(invalid.validate().is_err());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn validate(&self) -> Result<(), crate::error::SvmError> {
        use crate::error::SvmError;

        // gamma must be non-negative for kernels that use it
        if matches!(
            self.kernel_type,
            KernelType::Polynomial | KernelType::Rbf | KernelType::Sigmoid
        ) && self.gamma < 0.0
        {
            return Err(SvmError::InvalidParameter("gamma < 0".into()));
        }

        // polynomial degree must be non-negative
        if self.kernel_type == KernelType::Polynomial && self.degree < 0 {
            return Err(SvmError::InvalidParameter(
                "degree of polynomial kernel < 0".into(),
            ));
        }

        if self.cache_size <= 0.0 {
            return Err(SvmError::InvalidParameter("cache_size <= 0".into()));
        }

        if self.eps <= 0.0 {
            return Err(SvmError::InvalidParameter("eps <= 0".into()));
        }

        // C > 0 for formulations that use it
        if matches!(
            self.svm_type,
            SvmType::CSvc | SvmType::EpsilonSvr | SvmType::NuSvr
        ) && self.c <= 0.0
        {
            return Err(SvmError::InvalidParameter("C <= 0".into()));
        }

        // nu ∈ (0, 1] for formulations that use it
        if matches!(
            self.svm_type,
            SvmType::NuSvc | SvmType::OneClass | SvmType::NuSvr
        ) && (self.nu <= 0.0 || self.nu > 1.0)
        {
            return Err(SvmError::InvalidParameter("nu <= 0 or nu > 1".into()));
        }

        // p >= 0 for epsilon-SVR
        if self.svm_type == SvmType::EpsilonSvr && self.p < 0.0 {
            return Err(SvmError::InvalidParameter("p < 0".into()));
        }

        Ok(())
    }
}

/// Checks the first node of a precomputed-kernel row, which must be
/// `0:sample_serial_number`.
///
/// The serial is an integer in `[1, upper]`, and never above `i32::MAX`,
/// because C LIBSVM casts it to an `int` wherever it uses it. The caller passes
/// the sample count as `upper`, or `usize::MAX` when it has none. On failure the
/// text names the value found and the limit, and continues the caller's message
/// after "must start with `0:sample_serial_number`".
pub(crate) fn check_sample_serial(first: &SvmNode, upper: usize) -> Result<(), String> {
    let limit = i32::try_from(upper).unwrap_or(i32::MAX);
    if first.index == 0
        && first.value.is_finite()
        && first.value >= 1.0
        && first.value <= f64::from(limit)
        && first.value.fract() == 0.0
    {
        Ok(())
    } else {
        Err(format!(
            "in [1, {limit}], got {}:{}",
            first.index, first.value
        ))
    }
}

/// Full parameter check including ν-SVC feasibility against training data.
///
/// Matches the original LIBSVM `svm_check_parameter()`.
///
/// # Errors
///
/// Returns [`crate::error::SvmError::InvalidParameter`] when the problem and
/// parameters do not match.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// libsvm_rs::check_parameter(&problem, &param)?;
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[expect(
    clippy::cast_possible_truncation,
    reason = "C truncates the same values at these points"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "C converts the same integer counts to double at these points"
)]
pub fn check_parameter(
    problem: &SvmProblem,
    param: &SvmParameter,
) -> Result<(), crate::error::SvmError> {
    use crate::error::SvmError;

    // Run the data-independent checks first
    param.validate()?;

    if problem.labels.len() != problem.instances.len() {
        return Err(SvmError::InvalidParameter(format!(
            "labels length ({}) does not match instance length ({})",
            problem.labels.len(),
            problem.instances.len()
        )));
    }

    if problem.labels.len() as u128 > (1u128 << 53) {
        return Err(SvmError::InvalidParameter(
            "row count exceeds 2^53; train on fewer rows".into(),
        ));
    }

    if problem.labels.is_empty() {
        return Err(SvmError::InvalidParameter(
            "problem has no instances".into(),
        ));
    }

    if param.kernel_type == KernelType::Precomputed {
        for (row, instance) in (1usize..).zip(&problem.instances) {
            let first = instance.first().ok_or_else(|| {
                SvmError::InvalidParameter(format!(
                    "precomputed kernel row {row} is missing 0:sample_serial_number"
                ))
            })?;
            check_sample_serial(first, problem.instances.len()).map_err(|why| {
                SvmError::InvalidParameter(format!(
                    "precomputed kernel row {row} must start with 0:sample_serial_number {why}"
                ))
            })?;
        }
    }

    // ν-SVC feasibility: for every pair of classes (i, j),
    // nu * (count_i + count_j) / 2 must be <= min(count_i, count_j)
    //
    // Note: LIBSVM casts labels to int for class grouping. We match this
    // behavior. Classification labels must be integers (non-integer labels
    // will be truncated, matching `(int)prob->y[i]` in the C code).
    if param.svm_type == SvmType::NuSvc {
        let mut class_counts: Vec<(i32, usize)> = Vec::new();
        for &y in &problem.labels {
            let label = y as i32;
            if let Some(entry) = class_counts.iter_mut().find(|(l, _)| *l == label) {
                entry.1 = entry.1.checked_add(1).ok_or_else(|| {
                    SvmError::InvalidParameter("class count overflow; train on fewer rows".into())
                })?;
            } else {
                class_counts.push((label, 1));
            }
        }

        for (i, &(_, n1)) in class_counts.iter().enumerate() {
            for &(_, n2) in class_counts.iter().skip(i).skip(1) {
                if param.nu
                    * n1.checked_add(n2).ok_or_else(|| {
                        SvmError::InvalidParameter(
                            "class count sum overflow; train on fewer rows".into(),
                        )
                    })? as f64
                    / 2.0
                    > n1.min(n2) as f64
                {
                    return Err(SvmError::InvalidParameter(
                        "specified nu is infeasible".into(),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// A trained SVM model.
///
/// Produced by training, or loaded from a LIBSVM model file.
///
/// `load_model` validates model-file shape contracts before returning this
/// type: class counts, support-vector counts, decision-function arrays, optional
/// probability metadata, and sparse support-vector rows must be internally
/// consistent. Manually constructed values bypass those checks, so callers that
/// accept external model text should prefer [`crate::io::load_model`] or
/// [`crate::io::load_model_from_reader`].
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// assert_eq!(model.nr_class, 2);
/// assert_eq!(model.n_sv.iter().sum::<usize>(), model.sv.len());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SvmModel {
    /// Parameters used during training.
    pub param: SvmParameter,
    /// Number of classes (2 for binary, >2 for multiclass, 2 for regression).
    pub nr_class: usize,
    /// Support vectors (sparse feature vectors).
    pub sv: Vec<Vec<SvmNode>>,
    /// Support vector coefficients. For k classes, this is a
    /// `(k-1) × num_sv` matrix stored as `Vec<Vec<f64>>`.
    pub sv_coef: Vec<Vec<f64>>,
    /// Bias terms (rho). One per class pair: `k*(k-1)/2` values.
    pub rho: Vec<f64>,
    /// Pairwise probability parameter A (Platt scaling). Empty if not trained
    /// with probability estimates.
    pub prob_a: Vec<f64>,
    /// Pairwise probability parameter B (Platt scaling). Empty if not trained
    /// with probability estimates.
    pub prob_b: Vec<f64>,
    /// Probability density marks (for one-class SVM).
    pub prob_density_marks: Vec<f64>,
    /// Original indices of support vectors in the training set (1-based).
    pub sv_indices: Vec<usize>,
    /// Class labels (in the order used internally).
    pub label: Vec<i32>,
    /// Number of support vectors per class.
    pub n_sv: Vec<usize>,
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl serde::Serialize for SvmType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_i32(*self as i32)
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> serde::Deserialize<'de> for SvmType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match <i32 as serde::Deserialize>::deserialize(deserializer)? {
            0 => Ok(SvmType::CSvc),
            1 => Ok(SvmType::NuSvc),
            2 => Ok(SvmType::OneClass),
            3 => Ok(SvmType::EpsilonSvr),
            4 => Ok(SvmType::NuSvr),
            code => Err(serde::de::Error::custom(format!(
                "unknown SvmType code {code}"
            ))),
        }
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl serde::Serialize for KernelType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_i32(*self as i32)
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> serde::Deserialize<'de> for KernelType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match <i32 as serde::Deserialize>::deserialize(deserializer)? {
            0 => Ok(KernelType::Linear),
            1 => Ok(KernelType::Polynomial),
            2 => Ok(KernelType::Rbf),
            3 => Ok(KernelType::Sigmoid),
            4 => Ok(KernelType::Precomputed),
            code => Err(serde::de::Error::custom(format!(
                "unknown KernelType code {code}"
            ))),
        }
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> serde::Deserialize<'de> for SvmModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawSvmModel {
            param: SvmParameter,
            nr_class: usize,
            sv: Vec<Vec<SvmNode>>,
            sv_coef: Vec<Vec<f64>>,
            rho: Vec<f64>,
            prob_a: Vec<f64>,
            prob_b: Vec<f64>,
            prob_density_marks: Vec<f64>,
            sv_indices: Vec<usize>,
            label: Vec<i32>,
            n_sv: Vec<usize>,
        }

        let raw = <RawSvmModel as serde::Deserialize>::deserialize(deserializer)?;
        let model = SvmModel {
            param: raw.param,
            nr_class: raw.nr_class,
            sv: raw.sv,
            sv_coef: raw.sv_coef,
            rho: raw.rho,
            prob_a: raw.prob_a,
            prob_b: raw.prob_b,
            prob_density_marks: raw.prob_density_marks,
            sv_indices: raw.sv_indices,
            label: raw.label,
            n_sv: raw.n_sv,
        };
        crate::io::validate_model(&model).map_err(serde::de::Error::custom)?;
        Ok(model)
    }
}

impl SvmModel {
    /// Return the SVM type used by the model.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::io::load_problem_from_reader;
    /// use libsvm_rs::{KernelType, SvmParameter, SvmType};
    /// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
    /// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
    /// let model = libsvm_rs::train::svm_train(&problem, &param);
    /// assert_eq!(model.svm_type(), SvmType::CSvc);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    #[must_use]
    pub fn svm_type(&self) -> SvmType {
        self.param.svm_type
    }

    /// Return number of classes.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::io::load_problem_from_reader;
    /// use libsvm_rs::{KernelType, SvmParameter, SvmType};
    /// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
    /// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
    /// let model = libsvm_rs::train::svm_train(&problem, &param);
    /// assert_eq!(model.class_count(), 2);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.nr_class
    }

    /// Return class labels in internal one-vs-one order.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::io::load_problem_from_reader;
    /// use libsvm_rs::{KernelType, SvmParameter, SvmType};
    /// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
    /// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
    /// let model = libsvm_rs::train::svm_train(&problem, &param);
    /// assert_eq!(model.labels(), &[1, -1]);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    #[must_use]
    pub fn labels(&self) -> &[i32] {
        &self.label
    }

    /// Return original 1-based support-vector indices.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::io::load_problem_from_reader;
    /// use libsvm_rs::{KernelType, SvmParameter, SvmType};
    /// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
    /// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
    /// let model = libsvm_rs::train::svm_train(&problem, &param);
    /// assert!(model.support_vector_indices().iter().all(|i| (1..=problem.labels.len()).contains(i)));
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    #[must_use]
    pub fn support_vector_indices(&self) -> &[usize] {
        &self.sv_indices
    }

    /// Return total number of support vectors.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::io::load_problem_from_reader;
    /// use libsvm_rs::{KernelType, SvmParameter, SvmType};
    /// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
    /// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
    /// let model = libsvm_rs::train::svm_train(&problem, &param);
    /// assert_eq!(model.support_vector_count(), model.sv.len());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    #[must_use]
    pub fn support_vector_count(&self) -> usize {
        self.sv.len()
    }

    /// Return SVR sigma when a probability-capable SVR model is available.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::io::load_problem_from_reader;
    /// use libsvm_rs::{KernelType, SvmParameter, SvmType};
    /// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
    /// let param = SvmParameter {
    ///     svm_type: SvmType::EpsilonSvr,
    ///     kernel_type: KernelType::Linear,
    ///     probability: true,
    ///     ..Default::default()
    /// };
    /// let model = libsvm_rs::train::svm_train(&problem, &param);
    /// let sigma = model.svr_probability()
    ///     .ok_or(libsvm_rs::SvmError::InvalidParameter("missing SVR scale".into()))?;
    /// assert!(sigma.is_finite() && sigma >= 0.0);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    #[must_use]
    pub fn svr_probability(&self) -> Option<f64> {
        match self.param.svm_type {
            SvmType::EpsilonSvr | SvmType::NuSvr => self.prob_a.first().copied(),
            _ => None,
        }
    }

    /// Check whether the model contains probability metadata.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::io::load_problem_from_reader;
    /// use libsvm_rs::{KernelType, SvmParameter, SvmType};
    /// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
    /// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
    /// let model = libsvm_rs::train::svm_train(&problem, &param);
    /// assert!(!model.has_probability_model());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    #[must_use]
    pub fn has_probability_model(&self) -> bool {
        match self.param.svm_type {
            SvmType::CSvc | SvmType::NuSvc => !self.prob_a.is_empty() && !self.prob_b.is_empty(),
            SvmType::EpsilonSvr | SvmType::NuSvr => !self.prob_a.is_empty(),
            SvmType::OneClass => !self.prob_density_marks.is_empty(),
        }
    }
}

/// C-API style helper matching LIBSVM's `svm_get_svm_type`.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// assert_eq!(libsvm_rs::svm_get_svm_type(&model), SvmType::CSvc);
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_get_svm_type(model: &SvmModel) -> SvmType {
    model.svm_type()
}

/// C-API style helper matching LIBSVM's `svm_get_nr_class`.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// assert_eq!(libsvm_rs::svm_get_nr_class(&model), 2);
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_get_nr_class(model: &SvmModel) -> usize {
    model.class_count()
}

/// C-API style helper matching LIBSVM's `svm_get_labels`.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// assert_eq!(libsvm_rs::svm_get_labels(&model), &[1, -1]);
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_get_labels(model: &SvmModel) -> &[i32] {
    model.labels()
}

/// C-API style helper matching LIBSVM's `svm_get_sv_indices`.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// assert_eq!(libsvm_rs::svm_get_sv_indices(&model), model.support_vector_indices());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_get_sv_indices(model: &SvmModel) -> &[usize] {
    model.support_vector_indices()
}

/// C-API style helper matching LIBSVM's `svm_get_nr_sv`.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// assert_eq!(libsvm_rs::svm_get_nr_sv(&model), model.sv.len());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_get_nr_sv(model: &SvmModel) -> usize {
    model.support_vector_count()
}

/// C-API style helper matching LIBSVM's `svm_get_svr_probability`.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter {
///     svm_type: SvmType::EpsilonSvr,
///     kernel_type: KernelType::Linear,
///     probability: true,
///     ..Default::default()
/// };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// let sigma = libsvm_rs::svm_get_svr_probability(&model)
///     .ok_or(libsvm_rs::SvmError::InvalidParameter("missing SVR scale".into()))?;
/// assert!(sigma.is_finite() && sigma >= 0.0);
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_get_svr_probability(model: &SvmModel) -> Option<f64> {
    model.svr_probability()
}

/// C-API style helper matching LIBSVM's `svm_check_probability_model`.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// assert!(!libsvm_rs::svm_check_probability_model(&model));
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_check_probability_model(model: &SvmModel) -> bool {
    model.has_probability_model()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::svm_train;
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    fn default_params_are_valid() {
        SvmParameter::default().validate().unwrap();
    }

    #[test]
    fn negative_gamma_rejected() {
        let p = SvmParameter {
            gamma: -1.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn zero_cache_rejected() {
        let p = SvmParameter {
            cache_size: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn zero_c_rejected() {
        let p = SvmParameter {
            c: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn nu_out_of_range_rejected() {
        let p = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 1.5,
            ..Default::default()
        };
        assert!(p.validate().is_err());

        let p2 = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 0.0,
            ..Default::default()
        };
        assert!(p2.validate().is_err());
    }

    #[test]
    fn negative_p_rejected_for_svr() {
        let p = SvmParameter {
            svm_type: SvmType::EpsilonSvr,
            p: -0.1,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn negative_poly_degree_rejected() {
        let p = SvmParameter {
            kernel_type: KernelType::Polynomial,
            degree: -1,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn check_parameter_rejects_empty_problem() {
        let problem = SvmProblem {
            labels: Vec::new(),
            instances: Vec::new(),
        };
        let err = check_parameter(&problem, &SvmParameter::default()).unwrap_err();
        assert!(format!("{err}").contains("problem has no instances"));
    }

    #[test]
    fn check_parameter_rejects_label_instance_length_mismatch() {
        let problem = SvmProblem {
            labels: vec![1.0],
            instances: Vec::new(),
        };
        let err = check_parameter(&problem, &SvmParameter::default()).unwrap_err();
        assert!(format!("{err}").contains("does not match instance length"));
    }

    #[test]
    fn check_parameter_rejects_precomputed_rows_without_sample_serial_number() {
        let problem = SvmProblem {
            labels: vec![1.0, -1.0],
            instances: vec![
                vec![],
                vec![SvmNode {
                    index: 0,
                    value: 2.0,
                }],
            ],
        };
        let param = SvmParameter {
            kernel_type: KernelType::Precomputed,
            ..Default::default()
        };
        let err = check_parameter(&problem, &param).unwrap_err();
        assert!(format!("{err}").contains("missing 0:sample_serial_number"));
    }

    fn precomputed_problem_with_serial(serial: f64) -> SvmProblem {
        SvmProblem {
            labels: vec![1.0, -1.0],
            instances: vec![
                vec![SvmNode {
                    index: 0,
                    value: 1.0,
                }],
                vec![SvmNode {
                    index: 0,
                    value: serial,
                }],
            ],
        }
    }

    #[test]
    fn check_parameter_rejects_precomputed_serial_above_i32_max() {
        // i32::MAX + 1. C LIBSVM casts the serial to an int.
        let problem = precomputed_problem_with_serial(2_147_483_648.0);
        let param = SvmParameter {
            kernel_type: KernelType::Precomputed,
            ..Default::default()
        };
        let err = check_parameter(&problem, &param).unwrap_err();
        assert!(matches!(err, crate::error::SvmError::InvalidParameter(_)));
        let msg = format!("{err}");
        assert!(msg.contains("row 2"), "unexpected error: {msg}");
        assert!(msg.contains("2147483648"), "unexpected error: {msg}");
        assert!(msg.contains("[1, 2]"), "unexpected error: {msg}");
    }

    #[test]
    fn check_parameter_rejects_precomputed_fractional_serial() {
        let problem = precomputed_problem_with_serial(1.5);
        let param = SvmParameter {
            kernel_type: KernelType::Precomputed,
            ..Default::default()
        };
        let err = check_parameter(&problem, &param).unwrap_err();
        assert!(matches!(err, crate::error::SvmError::InvalidParameter(_)));
        let msg = format!("{err}");
        assert!(msg.contains("must start with 0:sample_serial_number"));
        assert!(msg.contains("1.5"), "unexpected error: {msg}");
    }

    #[test]
    fn sample_serial_limit_never_exceeds_i32_max() {
        // A problem with more than i32::MAX rows cannot be built in a test,
        // so the check is called with the row count directly.
        let at_limit = SvmNode {
            index: 0,
            value: f64::from(i32::MAX),
        };
        let above_limit = SvmNode {
            index: 0,
            value: f64::from(i32::MAX) + 1.0,
        };
        check_sample_serial(&at_limit, usize::MAX).unwrap();
        let why = check_sample_serial(&above_limit, usize::MAX).unwrap_err();
        assert!(why.contains("[1, 2147483647]"), "unexpected reason: {why}");
        assert!(why.contains("2147483648"), "unexpected reason: {why}");
    }

    #[test]
    fn nu_svc_feasibility_check() {
        // 2 classes with 3 samples each: nu * (3+3)/2 <= 3  →  nu <= 1
        let problem = SvmProblem {
            labels: vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            instances: vec![vec![]; 6],
        };
        let ok_param = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 0.5,
            ..Default::default()
        };
        check_parameter(&problem, &ok_param).unwrap();

        // nu = 0.9: 0.9 * 6/2 = 2.7 <= 3 → feasible
        let borderline = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 0.9,
            ..Default::default()
        };
        check_parameter(&problem, &borderline).unwrap();
    }

    #[test]
    fn nu_svc_infeasible() {
        // 5 class-A, 1 class-B: nu*(5+1)/2 > min(5,1)=1  →  nu > 1/3
        let problem = SvmProblem {
            labels: vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            instances: vec![vec![]; 6],
        };
        let param = SvmParameter {
            svm_type: SvmType::NuSvc,
            nu: 0.5, // 0.5 * 6/2 = 1.5 > 1
            ..Default::default()
        };
        let err = check_parameter(&problem, &param);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("infeasible"));
    }

    #[test]
    fn c_api_style_model_helpers() {
        let problem = crate::io::load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            gamma: 1.0 / 13.0,
            ..Default::default()
        };
        let model = svm_train(&problem, &param);

        assert_eq!(svm_get_svm_type(&model), SvmType::CSvc);
        assert_eq!(svm_get_nr_class(&model), 2);
        assert_eq!(svm_get_nr_sv(&model), model.sv.len());
        assert_eq!(svm_get_labels(&model), model.label.as_slice());
        assert_eq!(svm_get_sv_indices(&model), model.sv_indices.as_slice());
        assert!(!svm_check_probability_model(&model));
        assert_eq!(svm_get_svr_probability(&model), None);
    }

    #[test]
    fn probability_helpers_by_svm_type() {
        let svm = vec![SvmNode {
            index: 1,
            value: 1.0,
        }];

        let csvc_model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::CSvc,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![svm.clone()],
            sv_coef: vec![vec![1.0]],
            rho: vec![0.0],
            prob_a: vec![1.0],
            prob_b: vec![-0.5],
            prob_density_marks: vec![],
            sv_indices: vec![1],
            label: vec![1, -1],
            n_sv: vec![1, 0],
        };
        assert!(csvc_model.has_probability_model());
        assert!(svm_check_probability_model(&csvc_model));
        assert_eq!(svm_get_svr_probability(&csvc_model), None);

        let eps_svr_model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::EpsilonSvr,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![svm.clone()],
            sv_coef: vec![vec![0.8]],
            rho: vec![0.0],
            prob_a: vec![0.123],
            prob_b: vec![],
            prob_density_marks: vec![],
            sv_indices: vec![1],
            label: vec![],
            n_sv: vec![],
        };
        assert!(eps_svr_model.has_probability_model());
        assert_eq!(svm_get_svr_probability(&eps_svr_model), Some(0.123));

        let one_class_model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::OneClass,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![svm],
            sv_coef: vec![vec![1.0]],
            rho: vec![0.0],
            prob_a: vec![],
            prob_b: vec![],
            prob_density_marks: vec![0.1; 10],
            sv_indices: vec![1],
            label: vec![],
            n_sv: vec![],
        };
        assert!(one_class_model.has_probability_model());
        assert_eq!(svm_get_svr_probability(&one_class_model), None);
    }
}
